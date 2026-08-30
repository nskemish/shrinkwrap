#include "rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace shrinkwrap {

namespace {

inline void rasterize_triangle(
    const double* p0,
    const double* p1,
    const double* p2,

    float* top,
    float* bottom,

    const std::size_t width,
    const std::size_t height,

    const double min_x,
    const double min_y,
    const double resolution
) {
    const double x0 = p0[0];
    const double y0 = p0[1];
    const double z0 = p0[2];

    const double x1 = p1[0];
    const double y1 = p1[1];
    const double z1 = p1[2];

    const double x2 = p2[0];
    const double y2 = p2[1];
    const double z2 = p2[2];

    const double denominator =
        (y1 - y2) * (x0 - x2)
        +
        (x2 - x1) * (y0 - y2);

    //
    // Vertikalan triangle nema XY površinu.
    //
    if (std::abs(denominator) < 1e-12) {
        return;
    }

    const double triangle_min_x =
        std::min({x0, x1, x2});

    const double triangle_max_x =
        std::max({x0, x1, x2});

    const double triangle_min_y =
        std::min({y0, y1, y2});

    const double triangle_max_y =
        std::max({y0, y1, y2});

    std::int64_t ix0 =
        static_cast<std::int64_t>(
            std::floor(
                (triangle_min_x - min_x)
                /
                resolution
            )
        );

    std::int64_t ix1 =
        static_cast<std::int64_t>(
            std::ceil(
                (triangle_max_x - min_x)
                /
                resolution
            )
        );

    std::int64_t iy0 =
        static_cast<std::int64_t>(
            std::floor(
                (triangle_min_y - min_y)
                /
                resolution
            )
        );

    std::int64_t iy1 =
        static_cast<std::int64_t>(
            std::ceil(
                (triangle_max_y - min_y)
                /
                resolution
            )
        );

    ix0 = std::max<std::int64_t>(
        0,
        ix0
    );

    iy0 = std::max<std::int64_t>(
        0,
        iy0
    );

    ix1 = std::min<std::int64_t>(
        static_cast<std::int64_t>(width) - 1,
        ix1
    );

    iy1 = std::min<std::int64_t>(
        static_cast<std::int64_t>(height) - 1,
        iy1
    );

    if (
        ix0 > ix1
        ||
        iy0 > iy1
    ) {
        return;
    }

    //
    // Isto kao Python verzija.
    //
    constexpr double epsilon = -1e-7;

    for (
        std::int64_t iy = iy0;
        iy <= iy1;
        ++iy
    ) {
        const double y =
            min_y
            +
            static_cast<double>(iy)
            *
            resolution;

        for (
            std::int64_t ix = ix0;
            ix <= ix1;
            ++ix
        ) {
            const double x =
                min_x
                +
                static_cast<double>(ix)
                *
                resolution;

            const double a =
                (
                    (y1 - y2)
                    *
                    (x - x2)
                    +
                    (x2 - x1)
                    *
                    (y - y2)
                )
                /
                denominator;

            const double b =
                (
                    (y2 - y0)
                    *
                    (x - x2)
                    +
                    (x0 - x2)
                    *
                    (y - y2)
                )
                /
                denominator;

            const double c =
                1.0
                -
                a
                -
                b;

            if (
                a < epsilon
                ||
                b < epsilon
                ||
                c < epsilon
            ) {
                continue;
            }

            const double z =
                a * z0
                +
                b * z1
                +
                c * z2;

            const std::size_t index =
                static_cast<std::size_t>(iy)
                *
                width
                +
                static_cast<std::size_t>(ix);

            if (
                z
                >
                static_cast<double>(top[index])
            ) {
                top[index] =
                    static_cast<float>(z);
            }

            if (
                z
                <
                static_cast<double>(bottom[index])
            ) {
                bottom[index] =
                    static_cast<float>(z);
            }
        }
    }
}

} // namespace


void project_mesh(
    const double* vertices,
    const std::size_t vertex_count,

    const std::int64_t* faces,
    const std::size_t face_count,

    float* top,
    float* bottom,

    const std::size_t width,
    const std::size_t height,

    const double min_x,
    const double min_y,
    const double resolution
) {
    if (
        vertices == nullptr
        ||
        faces == nullptr
        ||
        top == nullptr
        ||
        bottom == nullptr
    ) {
        throw std::invalid_argument(
            "Null pointer passed to project_mesh."
        );
    }

    if (
        width == 0
        ||
        height == 0
    ) {
        throw std::invalid_argument(
            "Projection grid cannot be empty."
        );
    }

    if (
        resolution <= 0.0
        ||
        !std::isfinite(resolution)
    ) {
        throw std::invalid_argument(
            "Resolution must be finite and > 0."
        );
    }

    for (
        std::size_t face_index = 0;
        face_index < face_count;
        ++face_index
    ) {
        const std::int64_t i0 =
            faces[
                face_index * 3 + 0
            ];

        const std::int64_t i1 =
            faces[
                face_index * 3 + 1
            ];

        const std::int64_t i2 =
            faces[
                face_index * 3 + 2
            ];

        if (
            i0 < 0
            ||
            i1 < 0
            ||
            i2 < 0
            ||
            static_cast<std::size_t>(i0) >= vertex_count
            ||
            static_cast<std::size_t>(i1) >= vertex_count
            ||
            static_cast<std::size_t>(i2) >= vertex_count
        ) {
            throw std::out_of_range(
                "Mesh face references an invalid vertex."
            );
        }

        const double* p0 =
            vertices
            +
            static_cast<std::size_t>(i0)
            *
            3;

        const double* p1 =
            vertices
            +
            static_cast<std::size_t>(i1)
            *
            3;

        const double* p2 =
            vertices
            +
            static_cast<std::size_t>(i2)
            *
            3;

        rasterize_triangle(
            p0,
            p1,
            p2,

            top,
            bottom,

            width,
            height,

            min_x,
            min_y,
            resolution
        );
    }
}

} // namespace shrinkwrap
