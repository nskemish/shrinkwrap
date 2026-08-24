#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shrinkwrap {

struct QEMSimplifyResult {
    std::vector<float> vertices;
    std::vector<std::int64_t> faces;

    std::size_t input_faces = 0;
    std::size_t output_faces = 0;
    std::size_t collapsed_edges = 0;
};

QEMSimplifyResult simplify_qem(
    const float* vertices,
    std::size_t vertex_count,

    const std::int64_t* faces,
    std::size_t face_count,

    double target_ratio,
    bool preserve_normals
);

} // namespace shrinkwrap
