#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shrinkwrap {

struct MeshResult {
    std::vector<float> vertices;
    std::vector<std::int64_t> faces;

    std::size_t active_cells = 0;
    std::size_t boundary_cells = 0;
};

MeshResult envelope_to_mesh(
    const float* top,
    const float* bottom,
    const float* phi,

    std::size_t width,
    std::size_t height,

    double min_x,
    double min_y,
    double resolution
);

} // namespace shrinkwrap
