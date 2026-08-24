#include "regularizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace shrinkwrap {

namespace {

bool solve_3x3(
    double a[3][3],
    double b[3],
    double x[3]
) {
    for (int column = 0; column < 3; ++column) {

        int pivot = column;

        for (
            int row = column + 1;
            row < 3;
            ++row
        ) {
            if (
                std::abs(a[row][column])
                >
                std::abs(a[pivot][column])
            ) {
                pivot = row;
            }
        }

        if (
            std::abs(a[pivot][column])
            <
            1e-14
        ) {
            return false;
        }

        if (pivot != column) {
            for (int k = 0; k < 3; ++k) {
                std::swap(
                    a[pivot][k],
                    a[column][k]
                );
            }

            std::swap(
                b[pivot],
                b[column]
            );
        }

        const double divisor =
            a[column][column];

        for (
            int k = column;
            k < 3;
            ++k
        ) {
            a[column][k] /=
                divisor;
        }

        b[column] /=
            divisor;

        for (int row = 0; row < 3; ++row) {

            if (row == column) {
                continue;
            }

            const double factor =
                a[row][column];

            for (
                int k = column;
                k < 3;
                ++k
            ) {
                a[row][k] -=
                    factor
                    *
                    a[column][k];
            }

            b[row] -=
                factor
                *
                b[column];
        }
    }

    x[0] = b[0];
    x[1] = b[1];
    x[2] = b[2];

    return true;
}


bool fit_local_plane(
    const std::vector<float>& surface,
    const std::uint8_t* mask,

    const std::size_t width,
    const std::size_t height,

    const std::size_t cx,
    const std::size_t cy,

    const int radius,
    const double resolution,

    double& predicted_center,
    double& rms_error
) {
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;

    double sx = 0.0;
    double sy = 0.0;

    double sxz = 0.0;
    double syz = 0.0;
    double sz = 0.0;

    std::size_t count = 0;

    const int min_x =
        std::max(
            0,
            static_cast<int>(cx) - radius
        );

    const int max_x =
        std::min(
            static_cast<int>(width) - 1,
            static_cast<int>(cx) + radius
        );

    const int min_y =
        std::max(
            0,
            static_cast<int>(cy) - radius
        );

    const int max_y =
        std::min(
            static_cast<int>(height) - 1,
            static_cast<int>(cy) + radius
        );

    for (int y = min_y; y <= max_y; ++y) {

        for (int x = min_x; x <= max_x; ++x) {

            const std::size_t index =
                static_cast<std::size_t>(y)
                *
                width
                +
                static_cast<std::size_t>(x);

            if (!mask[index]) {
                continue;
            }

            const double px =
                (
                    static_cast<double>(x)
                    -
                    static_cast<double>(cx)
                )
                *
                resolution;

            const double py =
                (
                    static_cast<double>(y)
                    -
                    static_cast<double>(cy)
                )
                *
                resolution;

            const double z =
                static_cast<double>(
                    surface[index]
                );

            sxx += px * px;
            syy += py * py;
            sxy += px * py;

            sx += px;
            sy += py;

            sxz += px * z;
            syz += py * z;
            sz += z;

            ++count;
        }
    }

    if (count < 6) {
        return false;
    }

    double matrix[3][3] = {
        {
            sxx,
            sxy,
            sx
        },
        {
            sxy,
            syy,
            sy
        },
        {
            sx,
            sy,
            static_cast<double>(count)
        }
    };

    double rhs[3] = {
        sxz,
        syz,
        sz
    };

    double solution[3];

    if (
        !solve_3x3(
            matrix,
            rhs,
            solution
        )
    ) {
        return false;
    }

    const double a =
        solution[0];

    const double b =
        solution[1];

    const double c =
        solution[2];

    predicted_center = c;

    double squared_error = 0.0;

    for (int y = min_y; y <= max_y; ++y) {

        for (int x = min_x; x <= max_x; ++x) {

            const std::size_t index =
                static_cast<std::size_t>(y)
                *
                width
                +
                static_cast<std::size_t>(x);

            if (!mask[index]) {
                continue;
            }

            const double px =
                (
                    static_cast<double>(x)
                    -
                    static_cast<double>(cx)
                )
                *
                resolution;

            const double py =
                (
                    static_cast<double>(y)
                    -
                    static_cast<double>(cy)
                )
                *
                resolution;

            const double predicted =
                a * px
                +
                b * py
                +
                c;

            const double error =
                static_cast<double>(
                    surface[index]
                )
                -
                predicted;

            squared_error +=
                error * error;
        }
    }

    rms_error =
        std::sqrt(
            squared_error
            /
            static_cast<double>(count)
        );

    return true;
}

} // namespace


void regularize_steep_surface(
    float* surface,
    const float* original,
    const std::uint8_t* pcb_mask,

    const std::size_t width,
    const std::size_t height,

    const double resolution,
    const double angle_threshold_deg,
    const double correction_tolerance_mm,

    const int radius,
    const int iterations
) {
    if (
        surface == nullptr
        ||
        original == nullptr
        ||
        pcb_mask == nullptr
        ||
        width < 3
        ||
        height < 3
        ||
        resolution <= 0.0
        ||
        radius < 1
        ||
        iterations < 1
    ) {
        return;
    }

    const double angle_threshold =
        std::clamp(
            angle_threshold_deg,
            0.0,
            89.9
        );

    const double max_correction =
        std::max(
            correction_tolerance_mm,
            0.0
        );

    const double planar_rms_limit =
        std::max(
            max_correction * 4.0,
            resolution * 0.25
        );

    std::vector<float> current(
        surface,
        surface + width * height
    );

    std::vector<float> next =
        current;

    for (
        int iteration = 0;
        iteration < iterations;
        ++iteration
    ) {
        next = current;

        for (
            std::size_t y = 1;
            y + 1 < height;
            ++y
        ) {
            for (
                std::size_t x = 1;
                x + 1 < width;
                ++x
            ) {
                const std::size_t index =
                    y * width + x;

                if (!pcb_mask[index]) {
                    continue;
                }

                //
                // Real component geometry is locked.
                //
                if (
                    original[index]
                    >
                    1e-5f
                ) {
                    continue;
                }

                const std::size_t left =
                    index - 1;

                const std::size_t right =
                    index + 1;

                const std::size_t down =
                    index - width;

                const std::size_t up =
                    index + width;

                if (
                    !pcb_mask[left]
                    ||
                    !pcb_mask[right]
                    ||
                    !pcb_mask[down]
                    ||
                    !pcb_mask[up]
                ) {
                    continue;
                }

                const double dzdx =
                    (
                        static_cast<double>(
                            current[right]
                        )
                        -
                        static_cast<double>(
                            current[left]
                        )
                    )
                    /
                    (2.0 * resolution);

                const double dzdy =
                    (
                        static_cast<double>(
                            current[up]
                        )
                        -
                        static_cast<double>(
                            current[down]
                        )
                    )
                    /
                    (2.0 * resolution);

                const double gradient =
                    std::sqrt(
                        dzdx * dzdx
                        +
                        dzdy * dzdy
                    );

                const double angle =
                    std::atan(
                        gradient
                    )
                    *
                    180.0
                    /
                    std::acos(-1.0);

                if (
                    angle
                    <
                    angle_threshold
                ) {
                    continue;
                }

                double predicted = 0.0;
                double rms_error = 0.0;

                if (
                    !fit_local_plane(
                        current,
                        pcb_mask,

                        width,
                        height,

                        x,
                        y,

                        radius,
                        resolution,

                        predicted,
                        rms_error
                    )
                ) {
                    continue;
                }

                //
                // Strong curvature should remain curvature.
                // We only regularize regions which are already
                // reasonably close to some local plane.
                //
                if (
                    rms_error
                    >
                    planar_rms_limit
                ) {
                    continue;
                }

                const double value =
                    static_cast<double>(
                        current[index]
                    );

                double correction =
                    predicted
                    -
                    value;

                correction =
                    std::clamp(
                        correction,
                        -max_correction,
                        max_correction
                    );

                double corrected =
                    value
                    +
                    correction;

                //
                // Cloth may NEVER go through the actual
                // component geometry.
                //
                corrected =
                    std::max(
                        corrected,
                        static_cast<double>(
                            original[index]
                        )
                    );

                corrected =
                    std::max(
                        corrected,
                        0.0
                    );

                next[index] =
                    static_cast<float>(
                        corrected
                    );
            }
        }

        current.swap(
            next
        );
    }

    std::copy(
        current.begin(),
        current.end(),
        surface
    );
}

} // namespace shrinkwrap
