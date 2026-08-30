#include "qem_simplifier.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/boost/graph/iterator.h>

#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_filter.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_plane_and_line_policies.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>


namespace shrinkwrap {

namespace SMS =
    CGAL::Surface_mesh_simplification;

using Kernel =
    CGAL::Simple_cartesian<double>;

using Point =
    Kernel::Point_3;

using SurfaceMesh =
    CGAL::Surface_mesh<Point>;

using VertexDescriptor =
    SurfaceMesh::Vertex_index;

using GHPolicies =
    SMS::GarlandHeckbert_plane_and_line_policies<
        SurfaceMesh,
        Kernel
    >;


QEMSimplifyResult simplify_qem(
    const float* vertices,
    const std::size_t vertex_count,

    const std::int64_t* faces,
    const std::size_t face_count,

    const double target_ratio,
    const bool preserve_normals
) {
    if (
        vertices == nullptr
        ||
        faces == nullptr
    ) {
        throw std::invalid_argument(
            "Null mesh pointer."
        );
    }

    if (
        !std::isfinite(
            target_ratio
        )
        ||
        target_ratio <= 0.0
        ||
        target_ratio > 1.0
    ) {
        throw std::invalid_argument(
            "target_ratio must be in (0, 1]."
        );
    }

    SurfaceMesh mesh;

    std::vector<VertexDescriptor>
        vertex_map;

    vertex_map.reserve(
        vertex_count
    );

    // ========================================================
    // VERTICES
    // ========================================================

    for (
        std::size_t i = 0;
        i < vertex_count;
        ++i
    ) {
        const float* vertex =
            vertices
            +
            i * 3;

        vertex_map.push_back(
            mesh.add_vertex(
                Point(
                    static_cast<double>(
                        vertex[0]
                    ),
                    static_cast<double>(
                        vertex[1]
                    ),
                    static_cast<double>(
                        vertex[2]
                    )
                )
            )
        );
    }

    // ========================================================
    // FACES
    // ========================================================

    for (
        std::size_t i = 0;
        i < face_count;
        ++i
    ) {
        const std::int64_t i0 =
            faces[
                i * 3 + 0
            ];

        const std::int64_t i1 =
            faces[
                i * 3 + 1
            ];

        const std::int64_t i2 =
            faces[
                i * 3 + 2
            ];

        if (
            i0 < 0
            ||
            i1 < 0
            ||
            i2 < 0
            ||
            static_cast<std::size_t>(
                i0
            ) >= vertex_count
            ||
            static_cast<std::size_t>(
                i1
            ) >= vertex_count
            ||
            static_cast<std::size_t>(
                i2
            ) >= vertex_count
        ) {
            throw std::out_of_range(
                "Face references invalid vertex."
            );
        }

        const auto face =
            mesh.add_face(
                vertex_map[
                    static_cast<std::size_t>(
                        i0
                    )
                ],
                vertex_map[
                    static_cast<std::size_t>(
                        i1
                    )
                ],
                vertex_map[
                    static_cast<std::size_t>(
                        i2
                    )
                ]
            );

        if (
            face
            ==
            SurfaceMesh::null_face()
        ) {
            throw std::runtime_error(
                "CGAL rejected a face. "
                "Input mesh must be indexed, manifold, "
                "non-degenerate and consistently oriented."
            );
        }
    }

    const std::size_t input_faces =
        mesh.number_of_faces();

    // ========================================================
    // GARLAND-HECKBERT PLANE-AND-LINE QEM
    // ========================================================

    SMS::Edge_count_ratio_stop_predicate<
        SurfaceMesh
    > stop(
        target_ratio
    );

    GHPolicies policies(
        mesh
    );

    using GHCost =
        typename GHPolicies::Get_cost;

    using GHPlacement =
        typename GHPolicies::Get_placement;

    const GHCost& cost =
        policies.get_cost();

    const GHPlacement& placement =
        policies.get_placement();

    std::size_t collapsed = 0;

    if (preserve_normals) {

        SMS::Bounded_normal_change_filter<>
            normal_filter;

        collapsed =
            static_cast<std::size_t>(
                SMS::edge_collapse(
                    mesh,
                    stop,

                    CGAL::parameters::
                        get_cost(
                            cost
                        )
                        .get_placement(
                            placement
                        )
                        .filter(
                            normal_filter
                        )
                )
            );

    } else {

        collapsed =
            static_cast<std::size_t>(
                SMS::edge_collapse(
                    mesh,
                    stop,

                    CGAL::parameters::
                        get_cost(
                            cost
                        )
                        .get_placement(
                            placement
                        )
                )
            );
    }

    mesh.collect_garbage();

    // ========================================================
    // OUTPUT
    // ========================================================

    QEMSimplifyResult result;

    result.input_faces =
        input_faces;

    result.output_faces =
        mesh.number_of_faces();

    result.collapsed_edges =
        collapsed;

    std::unordered_map<
        std::size_t,
        std::int64_t
    > output_index;

    output_index.reserve(
        mesh.number_of_vertices()
    );

    // ========================================================
    // OUTPUT VERTICES
    // ========================================================

    for (
        const auto vertex :
        mesh.vertices()
    ) {
        const Point& point =
            mesh.point(
                vertex
            );

        const std::int64_t index =
            static_cast<std::int64_t>(
                result.vertices.size()
                /
                3
            );

        output_index.emplace(
            static_cast<std::size_t>(
                vertex.idx()
            ),
            index
        );

        result.vertices.push_back(
            static_cast<float>(
                point.x()
            )
        );

        result.vertices.push_back(
            static_cast<float>(
                point.y()
            )
        );

        result.vertices.push_back(
            static_cast<float>(
                point.z()
            )
        );
    }

    // ========================================================
    // OUTPUT FACES
    // ========================================================

    for (
        const auto face :
        mesh.faces()
    ) {
        const auto halfedge =
            mesh.halfedge(
                face
            );

        std::int64_t indices[3] = {
            -1,
            -1,
            -1
        };

        std::size_t count = 0;

        for (
            const auto vertex :
            CGAL::vertices_around_face(
                halfedge,
                mesh
            )
        ) {
            if (count >= 3) {
                throw std::runtime_error(
                    "CGAL output contains "
                    "a non-triangular face."
                );
            }

            indices[count] =
                output_index.at(
                    static_cast<std::size_t>(
                        vertex.idx()
                    )
                );

            ++count;
        }

        if (count != 3) {
            throw std::runtime_error(
                "CGAL output contains "
                "a non-triangular face."
            );
        }

        result.faces.push_back(
            indices[0]
        );

        result.faces.push_back(
            indices[1]
        );

        result.faces.push_back(
            indices[2]
        );
    }

    return result;
}

} // namespace shrinkwrap
