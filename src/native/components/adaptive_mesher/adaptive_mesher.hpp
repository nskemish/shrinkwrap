#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shrinkwrap {

struct AdaptiveMeshResult {
    std::vector<float> vertices;
    std::vector<std::int64_t> faces;

    std::size_t adaptive_patches = 0;
    std::size_t boundary_cells = 0;
    std::size_t base_cells_saved = 0;
};

AdaptiveMeshResult adaptive_envelope_to_mesh(
    const float* top,
    const float* bottom,
    const float* phi,

    std::size_t width,
    std::size_t height,

    double min_x,
    double min_y,
    double resolution,

    double surface_tolerance,
    std::size_t max_span_cells
);

} // namespace shrinkwrap
