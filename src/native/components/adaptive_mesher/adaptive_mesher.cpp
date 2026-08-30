#include "adaptive_mesher.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shrinkwrap {

namespace {

// ============================================================
// BASIC TYPES
// ============================================================

struct Point {
    double x;
    double y;
};

struct Rect {
    std::size_t x0;
    std::size_t y0;
    std::size_t x1;
    std::size_t y1;
};

struct ClippedPatch {
    std::vector<Point> polygon;
    std::vector<double> phi_values;
};

struct VertexKey {
    int side;
    std::int64_t x;
    std::int64_t y;

    bool operator==(
        const VertexKey& other
    ) const noexcept {
        return
            side == other.side &&
            x == other.x &&
            y == other.y;
    }
};

struct VertexKeyHash {
    std::size_t operator()(
        const VertexKey& key
    ) const noexcept {
        std::size_t h =
            std::hash<int>{}(
                key.side
            );

        auto combine =
            [&h](std::int64_t value) {
                h ^=
                    std::hash<std::int64_t>{}(
                        value
                    )
                    +
                    0x9e3779b97f4a7c15ULL
                    +
                    (h << 6)
                    +
                    (h >> 2);
            };

        combine(key.x);
        combine(key.y);

        return h;
    }
};

struct EdgeKey {
    std::int64_t ax;
    std::int64_t ay;
    std::int64_t bx;
    std::int64_t by;

    bool operator==(
        const EdgeKey& other
    ) const noexcept {
        return
            ax == other.ax &&
            ay == other.ay &&
            bx == other.bx &&
            by == other.by;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(
        const EdgeKey& key
    ) const noexcept {
        std::size_t h =
            std::hash<std::int64_t>{}(
                key.ax
            );

        auto combine =
            [&h](std::int64_t value) {
                h ^=
                    std::hash<std::int64_t>{}(
                        value
                    )
                    +
                    0x9e3779b97f4a7c15ULL
                    +
                    (h << 6)
                    +
                    (h >> 2);
            };

        combine(key.ay);
        combine(key.bx);
        combine(key.by);

        return h;
    }
};

struct BoundaryEdge {
    Point a;
    Point b;
};

constexpr double KEY_SCALE =
    1'000'000'000.0;


// ============================================================
// HELPERS
// ============================================================

inline std::int64_t quantize(
    const double value
) {
    return static_cast<std::int64_t>(
        std::llround(
            value * KEY_SCALE
        )
    );
}


inline Point zero_cross(
    const Point& p0,
    const double v0,
    const Point& p1,
    const double v1
) {
    const double denominator =
        v0 - v1;

    double t =
        std::abs(denominator) < 1e-12
        ? 0.5
        : v0 / denominator;

    t = std::clamp(
        t,
        0.0,
        1.0
    );

    return {
        p0.x + (p1.x - p0.x) * t,
        p0.y + (p1.y - p0.y) * t
    };
}


void clip_positive(
    const std::array<Point, 4>& points,
    const std::array<double, 4>& values,
    std::vector<Point>& output_points,
    std::vector<double>& output_values
) {
    output_points.clear();
    output_values.clear();

    output_points.reserve(6);
    output_values.reserve(6);

    for (
        std::size_t i = 0;
        i < 4;
        ++i
    ) {
        const std::size_t previous_index =
            (i + 3) % 4;

        const Point& current_p =
            points[i];

        const Point& previous_p =
            points[previous_index];

        const double current_v =
            values[i];

        const double previous_v =
            values[previous_index];

        const bool current_inside =
            current_v >= 0.0;

        const bool previous_inside =
            previous_v >= 0.0;

        if (current_inside) {

            if (!previous_inside) {
                output_points.push_back(
                    zero_cross(
                        previous_p,
                        previous_v,
                        current_p,
                        current_v
                    )
                );

                output_values.push_back(
                    0.0
                );
            }

            output_points.push_back(
                current_p
            );

            output_values.push_back(
                current_v
            );

        } else if (previous_inside) {

            output_points.push_back(
                zero_cross(
                    previous_p,
                    previous_v,
                    current_p,
                    current_v
                )
            );

            output_values.push_back(
                0.0
            );
        }
    }
}


EdgeKey make_edge_key(
    const Point& a,
    const Point& b
) {
    std::int64_t ax =
        quantize(a.x);

    std::int64_t ay =
        quantize(a.y);

    std::int64_t bx =
        quantize(b.x);

    std::int64_t by =
        quantize(b.y);

    if (
        bx < ax ||
        (
            bx == ax &&
            by < ay
        )
    ) {
        std::swap(ax, bx);
        std::swap(ay, by);
    }

    return {
        ax,
        ay,
        bx,
        by
    };
}


// ============================================================
// HEIGHT SAMPLING
// ============================================================

class SurfaceSampler {
public:
    SurfaceSampler(
        const float* top,
        const float* bottom,
        const std::size_t width,
        const std::size_t height
    )
        :
        top_(top),
        bottom_(bottom),
        width_(width),
        height_(height) {}

    double sample(
        const int side,
        double gx,
        double gy
    ) const {
        const float* data =
            side == 0
            ? top_
            : bottom_;

        gx = std::clamp(
            gx,
            0.0,
            static_cast<double>(
                width_ - 1
            )
        );

        gy = std::clamp(
            gy,
            0.0,
            static_cast<double>(
                height_ - 1
            )
        );

        const auto x0 =
            static_cast<std::size_t>(
                std::floor(gx)
            );

        const auto y0 =
            static_cast<std::size_t>(
                std::floor(gy)
            );

        const auto x1 =
            std::min(
                x0 + 1,
                width_ - 1
            );

        const auto y1 =
            std::min(
                y0 + 1,
                height_ - 1
            );

        const double tx =
            gx
            -
            static_cast<double>(x0);

        const double ty =
            gy
            -
            static_cast<double>(y0);

        const auto at =
            [&](std::size_t x,
                std::size_t y) {
                return static_cast<double>(
                    data[
                        y * width_ + x
                    ]
                );
            };

        return
            at(x0, y0)
            *
            (1.0 - tx)
            *
            (1.0 - ty)

            +

            at(x1, y0)
            *
            tx
            *
            (1.0 - ty)

            +

            at(x0, y1)
            *
            (1.0 - tx)
            *
            ty

            +

            at(x1, y1)
            *
            tx
            *
            ty;
    }

private:
    const float* top_;
    const float* bottom_;

    std::size_t width_;
    std::size_t height_;
};


// ============================================================
// ADAPTIVE ANALYSIS
// ============================================================

class AdaptiveBuilder {
public:
    AdaptiveBuilder(
        const float* top,
        const float* bottom,
        const float* phi,

        const std::size_t width,
        const std::size_t height,

        const double resolution,
        const double tolerance,
        const std::size_t max_span_cells
    )
        :
        top_(top),
        bottom_(bottom),
        phi_(phi),

        width_(width),
        height_(height),

        boundary_guard_mm_(
            std::max(
                resolution * 2.0,
                1e-8
            )
        ),

        tolerance_(
            std::max(
                tolerance,
                1e-8
            )
        ),

        max_span_cells_(
            std::max<std::size_t>(
                1,
                max_span_cells
            )
        ) {}

    void build() {
        recurse(
            {
                0,
                0,
                width_ - 1,
                height_ - 1
            }
        );
    }

    const std::vector<Rect>& leaves()
        const {
        return leaves_;
    }

    const std::vector<ClippedPatch>&
    clipped_patches() const {
        return clipped_;
    }

    std::size_t boundary_cells()
        const {
        return boundary_cells_;
    }

    std::size_t base_cells_saved()
        const {
        return base_cells_saved_;
    }

private:

    struct PhiRange {
        float minimum;
        float maximum;
    };


    PhiRange phi_range(
        const Rect& r
    ) const {
        float minimum =
            std::numeric_limits<float>::infinity();

        float maximum =
            -std::numeric_limits<float>::infinity();

        for (
            std::size_t y = r.y0;
            y <= r.y1;
            ++y
        ) {
            const std::size_t row =
                y * width_;

            for (
                std::size_t x = r.x0;
                x <= r.x1;
                ++x
            ) {
                const float value =
                    phi_[
                        row + x
                    ];

                minimum =
                    std::min(
                        minimum,
                        value
                    );

                maximum =
                    std::max(
                        maximum,
                        value
                    );
            }
        }

        return {
            minimum,
            maximum
        };
    }


    bool surface_within_tolerance(
        const Rect& r
    ) const {
        const std::size_t dx =
            r.x1 - r.x0;

        const std::size_t dy =
            r.y1 - r.y0;

        if (
            dx == 0 ||
            dy == 0
        ) {
            return true;
        }

        for (
            int side = 0;
            side < 2;
            ++side
        ) {
            const float* data =
                side == 0
                ? top_
                : bottom_;

            const auto at =
                [&](std::size_t x,
                    std::size_t y) {
                    return static_cast<double>(
                        data[
                            y * width_ + x
                        ]
                    );
                };

            const double z00 =
                at(r.x0, r.y0);

            const double z10 =
                at(r.x1, r.y0);

            const double z01 =
                at(r.x0, r.y1);

            const double z11 =
                at(r.x1, r.y1);

            for (
                std::size_t y = r.y0;
                y <= r.y1;
                ++y
            ) {
                const double v =
                    static_cast<double>(
                        y - r.y0
                    )
                    /
                    static_cast<double>(
                        dy
                    );

                for (
                    std::size_t x = r.x0;
                    x <= r.x1;
                    ++x
                ) {
                    const double u =
                        static_cast<double>(
                            x - r.x0
                        )
                        /
                        static_cast<double>(
                            dx
                        );

                    const double predicted =
                        z00
                        *
                        (1.0 - u)
                        *
                        (1.0 - v)

                        +

                        z10
                        *
                        u
                        *
                        (1.0 - v)

                        +

                        z01
                        *
                        (1.0 - u)
                        *
                        v

                        +

                        z11
                        *
                        u
                        *
                        v;

                    const double actual =
                        at(x, y);

                    if (
                        std::abs(
                            actual
                            -
                            predicted
                        )
                        >
                        tolerance_
                    ) {
                        return false;
                    }
                }
            }
        }

        return true;
    }


    void add_boundary_cell(
        const Rect& r
    ) {
        ++boundary_cells_;

        const std::size_t x =
            r.x0;

        const std::size_t y =
            r.y0;

        const std::array<Point, 4>
            points{{
                {
                    static_cast<double>(x),
                    static_cast<double>(y)
                },
                {
                    static_cast<double>(x + 1),
                    static_cast<double>(y)
                },
                {
                    static_cast<double>(x + 1),
                    static_cast<double>(y + 1)
                },
                {
                    static_cast<double>(x),
                    static_cast<double>(y + 1)
                }
            }};

        const std::array<double, 4>
            values{{
                static_cast<double>(
                    phi_[
                        y * width_ + x
                    ]
                ),

                static_cast<double>(
                    phi_[
                        y * width_ + x + 1
                    ]
                ),

                static_cast<double>(
                    phi_[
                        (y + 1) * width_
                        +
                        x + 1
                    ]
                ),

                static_cast<double>(
                    phi_[
                        (y + 1) * width_
                        +
                        x
                    ]
                )
            }};

        ClippedPatch patch;

        clip_positive(
            points,
            values,
            patch.polygon,
            patch.phi_values
        );

        if (
            patch.polygon.size()
            >= 3
        ) {
            clipped_.push_back(
                std::move(patch)
            );
        }
    }


    void split(
        const Rect& r
    ) {
        const std::size_t xm =
            r.x0
            +
            (r.x1 - r.x0) / 2;

        const std::size_t ym =
            r.y0
            +
            (r.y1 - r.y0) / 2;

        std::array<std::size_t, 3> xs{
            r.x0,
            xm,
            r.x1
        };

        std::array<std::size_t, 3> ys{
            r.y0,
            ym,
            r.y1
        };

        for (
            int yi = 0;
            yi < 2;
            ++yi
        ) {
            const std::size_t y0 =
                ys[yi];

            const std::size_t y1 =
                ys[yi + 1];

            if (y1 <= y0) {
                continue;
            }

            for (
                int xi = 0;
                xi < 2;
                ++xi
            ) {
                const std::size_t x0 =
                    xs[xi];

                const std::size_t x1 =
                    xs[xi + 1];

                if (x1 <= x0) {
                    continue;
                }

                recurse(
                    {
                        x0,
                        y0,
                        x1,
                        y1
                    }
                );
            }
        }
    }


    void recurse(
        const Rect& r
    ) {
        const std::size_t cells_x =
            r.x1 - r.x0;

        const std::size_t cells_y =
            r.y1 - r.y0;

        if (
            cells_x == 0 ||
            cells_y == 0
        ) {
            return;
        }

        const PhiRange range =
            phi_range(r);

        // Completely outside.
        if (
            range.maximum < 0.0f
        ) {
            return;
        }

        const bool base_cell =
            cells_x == 1
            &&
            cells_y == 1;

        const bool completely_inside =
            range.minimum > 0.0f;

        if (!completely_inside) {

            if (base_cell) {
                add_boundary_cell(r);
                return;
            }

            split(r);
            return;
        }

        //
        // Keep a very thin uniform band next to Edge_Cuts /
        // holes. This gives clipped boundary cells a conforming
        // base-grid neighbour and prevents T-junction cracks.
        //
        const bool near_boundary =
            static_cast<double>(
                range.minimum
            )
            <=
            boundary_guard_mm_;

        if (
            near_boundary
            &&
            !base_cell
        ) {
            split(r);
            return;
        }

        const bool span_ok =
            cells_x <= max_span_cells_
            &&
            cells_y <= max_span_cells_;

        if (
            span_ok
            &&
            surface_within_tolerance(r)
        ) {
            leaves_.push_back(r);

            const std::size_t represented =
                cells_x * cells_y;

            if (represented > 1) {
                base_cells_saved_ +=
                    represented - 1;
            }

            return;
        }

        if (base_cell) {
            leaves_.push_back(r);
            return;
        }

        split(r);
    }


    const float* top_;
    const float* bottom_;
    const float* phi_;

    std::size_t width_;
    std::size_t height_;

    double boundary_guard_mm_;
    double tolerance_;
    std::size_t max_span_cells_;

    std::vector<Rect> leaves_;
    std::vector<ClippedPatch> clipped_;

    std::size_t boundary_cells_ = 0;
    std::size_t base_cells_saved_ = 0;
};


// ============================================================
// FINAL MESH BUILDER
// ============================================================

class MeshWriter {
public:
    MeshWriter(
        const SurfaceSampler& sampler,
        const double min_x,
        const double min_y,
        const double resolution
    )
        :
        sampler_(sampler),
        min_x_(min_x),
        min_y_(min_y),
        resolution_(resolution) {}


    std::int64_t vertex(
        const double gx,
        const double gy,
        const int side
    ) {
        const VertexKey key{
            side,
            quantize(gx),
            quantize(gy)
        };

        const auto found =
            cache_.find(key);

        if (
            found != cache_.end()
        ) {
            return found->second;
        }

        const double x =
            min_x_
            +
            gx * resolution_;

        const double y =
            min_y_
            +
            gy * resolution_;

        const double z =
            sampler_.sample(
                side,
                gx,
                gy
            );

        const std::int64_t index =
            static_cast<std::int64_t>(
                vertices.size() / 3
            );

        vertices.push_back(
            static_cast<float>(x)
        );

        vertices.push_back(
            static_cast<float>(y)
        );

        vertices.push_back(
            static_cast<float>(z)
        );

        cache_.emplace(
            key,
            index
        );

        return index;
    }


    void triangle(
        const std::int64_t a,
        const std::int64_t b,
        const std::int64_t c
    ) {
        if (
            a == b ||
            b == c ||
            a == c
        ) {
            return;
        }

        faces.push_back(a);
        faces.push_back(b);
        faces.push_back(c);
    }


    std::vector<float> vertices;
    std::vector<std::int64_t> faces;

private:
    const SurfaceSampler& sampler_;

    double min_x_;
    double min_y_;
    double resolution_;

    std::unordered_map<
        VertexKey,
        std::int64_t,
        VertexKeyHash
    > cache_;
};


// ============================================================
// RECTANGLE EDGE CUTS
// ============================================================

void add_cut(
    std::map<
        std::size_t,
        std::vector<std::size_t>
    >& cuts,
    const std::size_t line,
    const std::size_t position
) {
    cuts[line].push_back(
        position
    );
}


void normalize_cuts(
    std::map<
        std::size_t,
        std::vector<std::size_t>
    >& cuts
) {
    for (auto& entry : cuts) {
        auto& values =
            entry.second;

        std::sort(
            values.begin(),
            values.end()
        );

        values.erase(
            std::unique(
                values.begin(),
                values.end()
            ),
            values.end()
        );
    }
}


std::vector<std::size_t>
cuts_in_range(
    const std::map<
        std::size_t,
        std::vector<std::size_t>
    >& cuts,

    const std::size_t line,
    const std::size_t low,
    const std::size_t high
) {
    std::vector<std::size_t> result;

    const auto found =
        cuts.find(line);

    if (found == cuts.end()) {
        result.push_back(low);

        if (high != low) {
            result.push_back(high);
        }

        return result;
    }

    const auto& values =
        found->second;

    auto begin =
        std::lower_bound(
            values.begin(),
            values.end(),
            low
        );

    auto end =
        std::upper_bound(
            values.begin(),
            values.end(),
            high
        );

    result.assign(
        begin,
        end
    );

    if (
        result.empty()
        ||
        result.front() != low
    ) {
        result.insert(
            result.begin(),
            low
        );
    }

    if (
        result.back() != high
    ) {
        result.push_back(
            high
        );
    }

    return result;
}


// ============================================================
// RECTANGLE PERIMETER
// ============================================================

std::vector<Point> rect_perimeter(
    const Rect& r,

    const std::map<
        std::size_t,
        std::vector<std::size_t>
    >& horizontal,

    const std::map<
        std::size_t,
        std::vector<std::size_t>
    >& vertical
) {
    std::vector<Point> polygon;

    // Bottom: left -> right
    {
        auto xs =
            cuts_in_range(
                horizontal,
                r.y0,
                r.x0,
                r.x1
            );

        for (const auto x : xs) {
            polygon.push_back({
                static_cast<double>(x),
                static_cast<double>(r.y0)
            });
        }
    }

    // Right: bottom -> top, without first corner
    {
        auto ys =
            cuts_in_range(
                vertical,
                r.x1,
                r.y0,
                r.y1
            );

        for (
            std::size_t i = 1;
            i < ys.size();
            ++i
        ) {
            polygon.push_back({
                static_cast<double>(r.x1),
                static_cast<double>(ys[i])
            });
        }
    }

    // Top: right -> left, without first corner
    {
        auto xs =
            cuts_in_range(
                horizontal,
                r.y1,
                r.x0,
                r.x1
            );

        if (!xs.empty()) {
            for (
                std::size_t i =
                    xs.size() - 1;
                i-- > 0;
            ) {
                polygon.push_back({
                    static_cast<double>(
                        xs[i]
                    ),
                    static_cast<double>(
                        r.y1
                    )
                });
            }
        }
    }

    // Left: top -> bottom, excluding both corners
    {
        auto ys =
            cuts_in_range(
                vertical,
                r.x0,
                r.y0,
                r.y1
            );

        if (ys.size() > 2) {
            for (
                std::size_t i =
                    ys.size() - 1;
                i-- > 1;
            ) {
                polygon.push_back({
                    static_cast<double>(
                        r.x0
                    ),
                    static_cast<double>(
                        ys[i]
                    )
                });
            }
        }
    }

    return polygon;
}


// ============================================================
// MAIN
// ============================================================

} // namespace


AdaptiveMeshResult adaptive_envelope_to_mesh(
    const float* top,
    const float* bottom,
    const float* phi,

    const std::size_t width,
    const std::size_t height,

    const double min_x,
    const double min_y,
    const double resolution,

    const double surface_tolerance,
    const std::size_t max_span_cells
) {
    if (
        top == nullptr ||
        bottom == nullptr ||
        phi == nullptr
    ) {
        throw std::invalid_argument(
            "Null pointer passed to adaptive mesher."
        );
    }

    if (
        width < 2 ||
        height < 2
    ) {
        throw std::invalid_argument(
            "Grid must be at least 2x2."
        );
    }

    if (
        !std::isfinite(resolution) ||
        resolution <= 0.0
    ) {
        throw std::invalid_argument(
            "Resolution must be finite and > 0."
        );
    }

    AdaptiveBuilder adaptive(
        top,
        bottom,
        phi,

        width,
        height,

        resolution,
        surface_tolerance,
        max_span_cells
    );

    adaptive.build();

    const auto& leaves =
        adaptive.leaves();

    const auto& clipped =
        adaptive.clipped_patches();

    // ========================================================
    // BUILD CONFORMING EDGE CUT MAP
    // ========================================================

    std::map<
        std::size_t,
        std::vector<std::size_t>
    > horizontal;

    std::map<
        std::size_t,
        std::vector<std::size_t>
    > vertical;

    for (const Rect& r : leaves) {

        add_cut(
            horizontal,
            r.y0,
            r.x0
        );

        add_cut(
            horizontal,
            r.y0,
            r.x1
        );

        add_cut(
            horizontal,
            r.y1,
            r.x0
        );

        add_cut(
            horizontal,
            r.y1,
            r.x1
        );

        add_cut(
            vertical,
            r.x0,
            r.y0
        );

        add_cut(
            vertical,
            r.x0,
            r.y1
        );

        add_cut(
            vertical,
            r.x1,
            r.y0
        );

        add_cut(
            vertical,
            r.x1,
            r.y1
        );
    }

    //
    // Boundary patches were not part of the adaptive leaf list,
    // so their integer grid corners were previously invisible
    // to the conforming edge map. That could create T-junctions
    // where a large adaptive rectangle met base-grid boundary
    // cells.
    //
    for (
        const ClippedPatch& patch :
        clipped
    ) {
        for (
            const Point& point :
            patch.polygon
        ) {
            const double rounded_x =
                std::round(point.x);

            const double rounded_y =
                std::round(point.y);

            const bool integer_x =
                std::abs(
                    point.x
                    -
                    rounded_x
                )
                <=
                1e-9;

            const bool integer_y =
                std::abs(
                    point.y
                    -
                    rounded_y
                )
                <=
                1e-9;

            //
            // Only true grid corners matter for adaptive
            // rectangle subdivision. Fractional phi=0 crossings
            // belong to the physical boundary itself.
            //
            if (
                integer_x
                &&
                integer_y
                &&
                rounded_x >= 0.0
                &&
                rounded_y >= 0.0
                &&
                rounded_x
                    <
                    static_cast<double>(width)
                &&
                rounded_y
                    <
                    static_cast<double>(height)
            ) {
                const auto gx =
                    static_cast<std::size_t>(
                        rounded_x
                    );

                const auto gy =
                    static_cast<std::size_t>(
                        rounded_y
                    );

                add_cut(
                    horizontal,
                    gy,
                    gx
                );

                add_cut(
                    vertical,
                    gx,
                    gy
                );
            }
        }
    }

    normalize_cuts(horizontal);
    normalize_cuts(vertical);

    SurfaceSampler sampler(
        top,
        bottom,
        width,
        height
    );

    MeshWriter writer(
        sampler,
        min_x,
        min_y,
        resolution
    );

    std::unordered_map<
        EdgeKey,
        BoundaryEdge,
        EdgeKeyHash
    > boundary_edges;

    // ========================================================
    // ADAPTIVE INTERIOR PATCHES
    // ========================================================

    for (const Rect& r : leaves) {

        std::vector<Point> polygon =
            rect_perimeter(
                r,
                horizontal,
                vertical
            );

        if (polygon.size() < 3) {
            continue;
        }

        const double center_x =
            (
                static_cast<double>(r.x0)
                +
                static_cast<double>(r.x1)
            )
            *
            0.5;

        const double center_y =
            (
                static_cast<double>(r.y0)
                +
                static_cast<double>(r.y1)
            )
            *
            0.5;

        const std::int64_t top_center =
            writer.vertex(
                center_x,
                center_y,
                0
            );

        const std::int64_t bottom_center =
            writer.vertex(
                center_x,
                center_y,
                1
            );

        const std::size_t count =
            polygon.size();

        for (
            std::size_t i = 0;
            i < count;
            ++i
        ) {
            const std::size_t j =
                (i + 1) % count;

            const std::int64_t ta =
                writer.vertex(
                    polygon[i].x,
                    polygon[i].y,
                    0
                );

            const std::int64_t tb =
                writer.vertex(
                    polygon[j].x,
                    polygon[j].y,
                    0
                );

            const std::int64_t ba =
                writer.vertex(
                    polygon[i].x,
                    polygon[i].y,
                    1
                );

            const std::int64_t bb =
                writer.vertex(
                    polygon[j].x,
                    polygon[j].y,
                    1
                );

            // top CCW
            writer.triangle(
                top_center,
                ta,
                tb
            );

            // bottom reversed
            writer.triangle(
                bottom_center,
                bb,
                ba
            );
        }
    }

    // ========================================================
    // EXACT BOUNDARY CELLS
    // ========================================================

    for (
        const ClippedPatch& patch :
        clipped
    ) {
        const std::size_t count =
            patch.polygon.size();

        if (count < 3) {
            continue;
        }

        std::vector<std::int64_t>
            top_indices;

        std::vector<std::int64_t>
            bottom_indices;

        top_indices.reserve(count);
        bottom_indices.reserve(count);

        for (
            const Point& point :
            patch.polygon
        ) {
            top_indices.push_back(
                writer.vertex(
                    point.x,
                    point.y,
                    0
                )
            );

            bottom_indices.push_back(
                writer.vertex(
                    point.x,
                    point.y,
                    1
                )
            );
        }

        for (
            std::size_t i = 1;
            i + 1 < count;
            ++i
        ) {
            writer.triangle(
                top_indices[0],
                top_indices[i],
                top_indices[i + 1]
            );

            writer.triangle(
                bottom_indices[0],
                bottom_indices[i + 1],
                bottom_indices[i]
            );
        }

        // phi=0 polygon edges become vertical walls.
        for (
            std::size_t i = 0;
            i < count;
            ++i
        ) {
            const std::size_t j =
                (i + 1) % count;

            if (
                std::abs(
                    patch.phi_values[i]
                ) <= 1e-10
                &&
                std::abs(
                    patch.phi_values[j]
                ) <= 1e-10
            ) {
                const Point& a =
                    patch.polygon[i];

                const Point& b =
                    patch.polygon[j];

                boundary_edges[
                    make_edge_key(a, b)
                ] = {
                    a,
                    b
                };
            }
        }
    }

    // ========================================================
    // OUTER / HOLE WALLS
    // ========================================================

    for (
        const auto& entry :
        boundary_edges
    ) {
        const Point& a =
            entry.second.a;

        const Point& b =
            entry.second.b;

        const auto ta =
            writer.vertex(
                a.x,
                a.y,
                0
            );

        const auto tb =
            writer.vertex(
                b.x,
                b.y,
                0
            );

        const auto ba =
            writer.vertex(
                a.x,
                a.y,
                1
            );

        const auto bb =
            writer.vertex(
                b.x,
                b.y,
                1
            );

        writer.triangle(
            ta,
            ba,
            bb
        );

        writer.triangle(
            ta,
            bb,
            tb
        );
    }

    if (
        writer.vertices.empty()
        ||
        writer.faces.empty()
    ) {
        throw std::runtime_error(
            "Adaptive mesher generated empty geometry."
        );
    }

    AdaptiveMeshResult result;

    result.vertices =
        std::move(writer.vertices);

    result.faces =
        std::move(writer.faces);

    result.adaptive_patches =
        leaves.size();

    result.boundary_cells =
        adaptive.boundary_cells();

    result.base_cells_saved =
        adaptive.base_cells_saved();

    return result;
}

} // namespace shrinkwrap
