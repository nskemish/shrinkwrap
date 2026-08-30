#include "mesher.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shrinkwrap {

namespace {

struct Point {
    double x;
    double y;
};

struct VertexKey {
    int side;
    std::int64_t x;
    std::int64_t y;

    bool operator==(const VertexKey& other) const noexcept {
        return
            side == other.side &&
            x == other.x &&
            y == other.y;
    }
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& key) const noexcept {
        std::size_t h = std::hash<int>{}(key.side);

        h ^= std::hash<std::int64_t>{}(key.x)
             + 0x9e3779b97f4a7c15ULL
             + (h << 6)
             + (h >> 2);

        h ^= std::hash<std::int64_t>{}(key.y)
             + 0x9e3779b97f4a7c15ULL
             + (h << 6)
             + (h >> 2);

        return h;
    }
};

struct EdgeKey {
    std::int64_t ax;
    std::int64_t ay;
    std::int64_t bx;
    std::int64_t by;

    bool operator==(const EdgeKey& other) const noexcept {
        return
            ax == other.ax &&
            ay == other.ay &&
            bx == other.bx &&
            by == other.by;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const noexcept {
        std::size_t h = std::hash<std::int64_t>{}(key.ax);

        auto combine = [&h](std::int64_t value) {
            h ^= std::hash<std::int64_t>{}(value)
                 + 0x9e3779b97f4a7c15ULL
                 + (h << 6)
                 + (h >> 2);
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

constexpr double KEY_SCALE = 1'000'000'000.0;

inline std::int64_t quantize(const double value) {
    return static_cast<std::int64_t>(
        std::llround(value * KEY_SCALE)
    );
}

inline Point zero_cross(
    const Point& p0,
    const double v0,
    const Point& p1,
    const double v1
) {
    const double denominator = v0 - v1;

    double t;

    if (std::abs(denominator) < 1e-12) {
        t = 0.5;
    } else {
        t = v0 / denominator;
    }

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

    constexpr std::size_t count = 4;

    for (std::size_t i = 0; i < count; ++i) {
        const Point& current_p = points[i];
        const double current_v = values[i];

        const std::size_t previous_index =
            (i + count - 1) % count;

        const Point& previous_p =
            points[previous_index];

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

                output_values.push_back(0.0);
            }

            output_points.push_back(current_p);
            output_values.push_back(current_v);

        } else if (previous_inside) {
            output_points.push_back(
                zero_cross(
                    previous_p,
                    previous_v,
                    current_p,
                    current_v
                )
            );

            output_values.push_back(0.0);
        }
    }
}

EdgeKey make_edge_key(
    const Point& a,
    const Point& b
) {
    std::int64_t ax = quantize(a.x);
    std::int64_t ay = quantize(a.y);

    std::int64_t bx = quantize(b.x);
    std::int64_t by = quantize(b.y);

    if (
        bx < ax ||
        (bx == ax && by < ay)
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

class MeshBuilder {
public:
    MeshBuilder(
        const float* top,
        const float* bottom,
        const std::size_t width,
        const std::size_t height,
        const double min_x,
        const double min_y,
        const double resolution
    )
        : top_(top),
          bottom_(bottom),
          width_(width),
          height_(height),
          min_x_(min_x),
          min_y_(min_y),
          resolution_(resolution) {}

    std::int64_t get_vertex(
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
            vertex_cache_.find(key);

        if (found != vertex_cache_.end()) {
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

        const float* data =
            side == 0
            ? top_
            : bottom_;

        const double z =
            sample_height(
                data,
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

        vertex_cache_.emplace(
            key,
            index
        );

        return index;
    }

    void add_face(
        const std::int64_t a,
        const std::int64_t b,
        const std::int64_t c
    ) {
        faces.push_back(a);
        faces.push_back(b);
        faces.push_back(c);
    }

    std::vector<float> vertices;
    std::vector<std::int64_t> faces;

private:
    double sample_height(
        const float* data,
        double gx,
        double gy
    ) const {
        gx = std::clamp(
            gx,
            0.0,
            static_cast<double>(width_ - 1)
        );

        gy = std::clamp(
            gy,
            0.0,
            static_cast<double>(height_ - 1)
        );

        const auto x0 =
            static_cast<std::size_t>(
                std::floor(gx)
            );

        const auto y0 =
            static_cast<std::size_t>(
                std::floor(gy)
            );

        const std::size_t x1 =
            std::min(
                x0 + 1,
                width_ - 1
            );

        const std::size_t y1 =
            std::min(
                y0 + 1,
                height_ - 1
            );

        const double tx =
            gx - static_cast<double>(x0);

        const double ty =
            gy - static_cast<double>(y0);

        const auto at =
            [this, data](
                const std::size_t x,
                const std::size_t y
            ) -> double {
                return data[
                    y * width_ + x
                ];
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

    const float* top_;
    const float* bottom_;

    std::size_t width_;
    std::size_t height_;

    double min_x_;
    double min_y_;
    double resolution_;

    std::unordered_map<
        VertexKey,
        std::int64_t,
        VertexKeyHash
    > vertex_cache_;
};

} // namespace


MeshResult envelope_to_mesh(
    const float* top,
    const float* bottom,
    const float* phi,

    const std::size_t width,
    const std::size_t height,

    const double min_x,
    const double min_y,
    const double resolution
) {
    if (
        top == nullptr ||
        bottom == nullptr ||
        phi == nullptr
    ) {
        throw std::invalid_argument(
            "Null pointer passed to envelope_to_mesh."
        );
    }

    if (
        width < 2 ||
        height < 2
    ) {
        throw std::invalid_argument(
            "Envelope grid must be at least 2x2."
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

    MeshBuilder builder(
        top,
        bottom,
        width,
        height,
        min_x,
        min_y,
        resolution
    );

    std::unordered_map<
        EdgeKey,
        BoundaryEdge,
        EdgeKeyHash
    > boundary_edges;

    std::vector<Point> polygon;
    std::vector<double> polygon_values;

    std::size_t active_cells = 0;
    std::size_t boundary_cells = 0;

    for (
        std::size_t y = 0;
        y < height - 1;
        ++y
    ) {
        for (
            std::size_t x = 0;
            x < width - 1;
            ++x
        ) {
            const std::array<Point, 4> points{{
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

            const std::array<double, 4> values{{
                phi[
                    y * width + x
                ],

                phi[
                    y * width + (x + 1)
                ],

                phi[
                    (y + 1) * width + (x + 1)
                ],

                phi[
                    (y + 1) * width + x
                ]
            }};

            const double maximum =
                std::max({
                    values[0],
                    values[1],
                    values[2],
                    values[3]
                });

            if (maximum < 0.0) {
                continue;
            }

            clip_positive(
                points,
                values,
                polygon,
                polygon_values
            );

            if (polygon.size() < 3) {
                continue;
            }

            ++active_cells;

            const bool fully_inside =
                values[0] > 0.0 &&
                values[1] > 0.0 &&
                values[2] > 0.0 &&
                values[3] > 0.0;

            if (!fully_inside) {
                ++boundary_cells;
            }

            std::vector<std::int64_t>
                top_indices;

            std::vector<std::int64_t>
                bottom_indices;

            top_indices.reserve(
                polygon.size()
            );

            bottom_indices.reserve(
                polygon.size()
            );

            for (const Point& point : polygon) {
                top_indices.push_back(
                    builder.get_vertex(
                        point.x,
                        point.y,
                        0
                    )
                );

                bottom_indices.push_back(
                    builder.get_vertex(
                        point.x,
                        point.y,
                        1
                    )
                );
            }

            for (
                std::size_t i = 1;
                i + 1 < polygon.size();
                ++i
            ) {
                builder.add_face(
                    top_indices[0],
                    top_indices[i],
                    top_indices[i + 1]
                );

                builder.add_face(
                    bottom_indices[0],
                    bottom_indices[i + 1],
                    bottom_indices[i]
                );
            }

            const std::size_t count =
                polygon.size();

            for (
                std::size_t i = 0;
                i < count;
                ++i
            ) {
                const std::size_t j =
                    (i + 1) % count;

                if (
                    std::abs(
                        polygon_values[i]
                    ) <= 1e-10
                    &&
                    std::abs(
                        polygon_values[j]
                    ) <= 1e-10
                ) {
                    const Point& a =
                        polygon[i];

                    const Point& b =
                        polygon[j];

                    boundary_edges[
                        make_edge_key(a, b)
                    ] = {
                        a,
                        b
                    };
                }
            }
        }
    }

    for (
        const auto& entry :
        boundary_edges
    ) {
        const Point& a =
            entry.second.a;

        const Point& b =
            entry.second.b;

        const std::int64_t ta =
            builder.get_vertex(
                a.x,
                a.y,
                0
            );

        const std::int64_t tb =
            builder.get_vertex(
                b.x,
                b.y,
                0
            );

        const std::int64_t ba =
            builder.get_vertex(
                a.x,
                a.y,
                1
            );

        const std::int64_t bb =
            builder.get_vertex(
                b.x,
                b.y,
                1
            );

        builder.add_face(
            ta,
            ba,
            bb
        );

        builder.add_face(
            ta,
            bb,
            tb
        );
    }

    if (builder.vertices.empty()) {
        throw std::runtime_error(
            "Generated mesh contains no vertices."
        );
    }

    if (builder.faces.empty()) {
        throw std::runtime_error(
            "Generated mesh contains no faces."
        );
    }

    MeshResult result;

    result.vertices =
        std::move(builder.vertices);

    result.faces =
        std::move(builder.faces);

    result.active_cells =
        active_cells;

    result.boundary_cells =
        boundary_cells;

    return result;
}

} // namespace shrinkwrap
