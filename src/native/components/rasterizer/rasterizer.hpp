#pragma once

#include <cstddef>
#include <cstdint>

namespace shrinkwrap {

void project_mesh(
    const double* vertices,
    std::size_t vertex_count,

    const std::int64_t* faces,
    std::size_t face_count,

    float* top,
    float* bottom,

    std::size_t width,
    std::size_t height,

    double min_x,
    double min_y,
    double resolution
);

} // namespace shrinkwrap
