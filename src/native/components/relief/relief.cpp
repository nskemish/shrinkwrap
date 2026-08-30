#include "relief.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace shrinkwrap {

void propagate_relief(
    float* relief,
    const std::uint8_t* pcb_mask,
    const std::size_t width,
    const std::size_t height,
    const float drop_per_pixel
) {
    if (
        relief == nullptr
        ||
        pcb_mask == nullptr
    ) {
        throw std::invalid_argument(
            "Null pointer passed to propagate_relief."
        );
    }

    if (
        width == 0
        ||
        height == 0
    ) {
        return;
    }

    // LEFT -> RIGHT + RIGHT -> LEFT
    for (
        std::size_t y = 0;
        y < height;
        ++y
    ) {
        const std::size_t row = y * width;

        for (
            std::size_t x = 1;
            x < width;
            ++x
        ) {
            const std::size_t i = row + x;
            const std::size_t previous = i - 1;

            if (
                !pcb_mask[i]
                ||
                !pcb_mask[previous]
            ) {
                continue;
            }

            const float candidate =
                std::max(
                    0.0f,
                    relief[previous]
                    -
                    drop_per_pixel
                );

            if (candidate > relief[i]) {
                relief[i] = candidate;
            }
        }

        if (width >= 2) {
            for (
                std::size_t x = width - 1;
                x-- > 0;
            ) {
                const std::size_t i = row + x;
                const std::size_t next = i + 1;

                if (
                    !pcb_mask[i]
                    ||
                    !pcb_mask[next]
                ) {
                    continue;
                }

                const float candidate =
                    std::max(
                        0.0f,
                        relief[next]
                        -
                        drop_per_pixel
                    );

                if (candidate > relief[i]) {
                    relief[i] = candidate;
                }
            }
        }
    }

    // BOTTOM -> TOP + TOP -> BOTTOM
    for (
        std::size_t x = 0;
        x < width;
        ++x
    ) {
        for (
            std::size_t y = 1;
            y < height;
            ++y
        ) {
            const std::size_t i =
                y * width
                +
                x;

            const std::size_t previous =
                (y - 1) * width
                +
                x;

            if (
                !pcb_mask[i]
                ||
                !pcb_mask[previous]
            ) {
                continue;
            }

            const float candidate =
                std::max(
                    0.0f,
                    relief[previous]
                    -
                    drop_per_pixel
                );

            if (candidate > relief[i]) {
                relief[i] = candidate;
            }
        }

        if (height >= 2) {
            for (
                std::size_t y = height - 1;
                y-- > 0;
            ) {
                const std::size_t i =
                    y * width
                    +
                    x;

                const std::size_t next =
                    (y + 1) * width
                    +
                    x;

                if (
                    !pcb_mask[i]
                    ||
                    !pcb_mask[next]
                ) {
                    continue;
                }

                const float candidate =
                    std::max(
                        0.0f,
                        relief[next]
                        -
                        drop_per_pixel
                    );

                if (candidate > relief[i]) {
                    relief[i] = candidate;
                }
            }
        }
    }
}

} // namespace shrinkwrap
