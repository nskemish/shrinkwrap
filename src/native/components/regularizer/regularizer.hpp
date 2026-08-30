#pragma once

#include <cstddef>
#include <cstdint>

namespace shrinkwrap {

void regularize_steep_surface(
    float* surface,
    const float* original,
    const std::uint8_t* pcb_mask,

    std::size_t width,
    std::size_t height,

    double resolution,
    double angle_threshold_deg,
    double correction_tolerance_mm,

    int radius,
    int iterations
);

} // namespace shrinkwrap
