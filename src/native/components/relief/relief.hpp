#pragma once

#include <cstddef>
#include <cstdint>

namespace shrinkwrap {

void propagate_relief(
    float* relief,
    const std::uint8_t* pcb_mask,
    std::size_t width,
    std::size_t height,
    float drop_per_pixel
);

} // namespace shrinkwrap
