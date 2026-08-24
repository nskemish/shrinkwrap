#include "rasterizer.hpp"
#include "relief.hpp"
#include "mesher.hpp"
#include "adaptive_mesher.hpp"
#include "qem_simplifier.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace {

void project_mesh_binding(
    py::array_t<
        double,
        py::array::c_style
    > vertices,

    py::array_t<
        std::int64_t,
        py::array::c_style
    > faces,

    py::array_t<
        float,
        py::array::c_style
    > top,

    py::array_t<
        float,
        py::array::c_style
    > bottom,

    const double min_x,
    const double min_y,
    const double resolution
) {
    if (vertices.ndim() != 2) {
        throw std::invalid_argument(
            "vertices must be a 2D array."
        );
    }

    if (vertices.shape(1) != 3) {
        throw std::invalid_argument(
            "vertices must have shape (N, 3)."
        );
    }

    if (faces.ndim() != 2) {
        throw std::invalid_argument(
            "faces must be a 2D array."
        );
    }

    if (faces.shape(1) != 3) {
        throw std::invalid_argument(
            "faces must have shape (M, 3)."
        );
    }

    if (
        top.ndim() != 2
        ||
        bottom.ndim() != 2
    ) {
        throw std::invalid_argument(
            "top and bottom must be 2D arrays."
        );
    }

    if (
        top.shape(0) != bottom.shape(0)
        ||
        top.shape(1) != bottom.shape(1)
    ) {
        throw std::invalid_argument(
            "top and bottom shapes must match."
        );
    }

    if (
        !top.writeable()
        ||
        !bottom.writeable()
    ) {
        throw std::invalid_argument(
            "top and bottom arrays must be writable."
        );
    }

    const auto vertex_count =
        static_cast<std::size_t>(
            vertices.shape(0)
        );

    const auto face_count =
        static_cast<std::size_t>(
            faces.shape(0)
        );

    const auto height =
        static_cast<std::size_t>(
            top.shape(0)
        );

    const auto width =
        static_cast<std::size_t>(
            top.shape(1)
        );

    const auto* vertex_ptr =
        vertices.data();

    const auto* face_ptr =
        faces.data();

    auto* top_ptr =
        top.mutable_data();

    auto* bottom_ptr =
        bottom.mutable_data();

    //
    // Tokom teškog C++ rada Python GIL nam nije potreban.
    //
    py::gil_scoped_release release;

    shrinkwrap::project_mesh(
        vertex_ptr,
        vertex_count,

        face_ptr,
        face_count,

        top_ptr,
        bottom_ptr,

        width,
        height,

        min_x,
        min_y,
        resolution
    );
}

} // namespace



void propagate_relief_binding(
    py::array_t<
        float,
        py::array::c_style
    > relief,

    py::array_t<
        bool,
        py::array::c_style
    > pcb_mask,

    const float drop_per_pixel
) {
    if (
        relief.ndim() != 2
        ||
        pcb_mask.ndim() != 2
    ) {
        throw std::invalid_argument(
            "relief and pcb_mask must be 2D arrays."
        );
    }

    if (
        relief.shape(0) != pcb_mask.shape(0)
        ||
        relief.shape(1) != pcb_mask.shape(1)
    ) {
        throw std::invalid_argument(
            "relief and pcb_mask shapes must match."
        );
    }

    if (!relief.writeable()) {
        throw std::invalid_argument(
            "relief must be writable."
        );
    }

    const auto height =
        static_cast<std::size_t>(
            relief.shape(0)
        );

    const auto width =
        static_cast<std::size_t>(
            relief.shape(1)
        );

    auto* relief_ptr =
        relief.mutable_data();

    const auto* mask_ptr =
        reinterpret_cast<const std::uint8_t*>(
            pcb_mask.data()
        );

    py::gil_scoped_release release;

    shrinkwrap::propagate_relief(
        relief_ptr,
        mask_ptr,
        width,
        height,
        drop_per_pixel
    );
}



py::tuple envelope_to_mesh_binding(
    py::array_t<
        float,
        py::array::c_style
    > top,

    py::array_t<
        float,
        py::array::c_style
    > bottom,

    py::array_t<
        float,
        py::array::c_style
    > phi,

    const double min_x,
    const double min_y,
    const double resolution
) {
    if (
        top.ndim() != 2 ||
        bottom.ndim() != 2 ||
        phi.ndim() != 2
    ) {
        throw std::invalid_argument(
            "top, bottom and phi must be 2D."
        );
    }

    if (
        top.shape(0) != bottom.shape(0) ||
        top.shape(1) != bottom.shape(1) ||
        top.shape(0) != phi.shape(0) ||
        top.shape(1) != phi.shape(1)
    ) {
        throw std::invalid_argument(
            "top, bottom and phi shapes must match."
        );
    }

    const auto height =
        static_cast<std::size_t>(
            top.shape(0)
        );

    const auto width =
        static_cast<std::size_t>(
            top.shape(1)
        );

    shrinkwrap::MeshResult result;

    {
        py::gil_scoped_release release;

        result =
            shrinkwrap::envelope_to_mesh(
                top.data(),
                bottom.data(),
                phi.data(),

                width,
                height,

                min_x,
                min_y,
                resolution
            );
    }

    const std::size_t vertex_count =
        result.vertices.size() / 3;

    const std::size_t face_count =
        result.faces.size() / 3;

    py::array_t<float> vertices({
        static_cast<py::ssize_t>(vertex_count),
        static_cast<py::ssize_t>(3)
    });

    py::array_t<std::int64_t> faces({
        static_cast<py::ssize_t>(face_count),
        static_cast<py::ssize_t>(3)
    });

    std::copy(
        result.vertices.begin(),
        result.vertices.end(),
        vertices.mutable_data()
    );

    std::copy(
        result.faces.begin(),
        result.faces.end(),
        faces.mutable_data()
    );

    return py::make_tuple(
        vertices,
        faces,
        result.active_cells,
        result.boundary_cells
    );
}



py::tuple adaptive_envelope_to_mesh_binding(
    py::array_t<
        float,
        py::array::c_style
    > top,

    py::array_t<
        float,
        py::array::c_style
    > bottom,

    py::array_t<
        float,
        py::array::c_style
    > phi,

    const double min_x,
    const double min_y,
    const double resolution,
    const double surface_tolerance,
    const std::size_t max_span_cells
) {
    if (
        top.ndim() != 2 ||
        bottom.ndim() != 2 ||
        phi.ndim() != 2
    ) {
        throw std::invalid_argument(
            "top, bottom and phi must be 2D."
        );
    }

    if (
        top.shape(0) != bottom.shape(0) ||
        top.shape(1) != bottom.shape(1) ||
        top.shape(0) != phi.shape(0) ||
        top.shape(1) != phi.shape(1)
    ) {
        throw std::invalid_argument(
            "top, bottom and phi shapes must match."
        );
    }

    const auto height =
        static_cast<std::size_t>(
            top.shape(0)
        );

    const auto width =
        static_cast<std::size_t>(
            top.shape(1)
        );

    shrinkwrap::AdaptiveMeshResult result;

    {
        py::gil_scoped_release release;

        result =
            shrinkwrap::adaptive_envelope_to_mesh(
                top.data(),
                bottom.data(),
                phi.data(),

                width,
                height,

                min_x,
                min_y,
                resolution,

                surface_tolerance,
                max_span_cells
            );
    }

    const std::size_t vertex_count =
        result.vertices.size() / 3;

    const std::size_t face_count =
        result.faces.size() / 3;

    py::array_t<float> vertices({
        static_cast<py::ssize_t>(
            vertex_count
        ),
        static_cast<py::ssize_t>(3)
    });

    py::array_t<std::int64_t> faces({
        static_cast<py::ssize_t>(
            face_count
        ),
        static_cast<py::ssize_t>(3)
    });

    std::copy(
        result.vertices.begin(),
        result.vertices.end(),
        vertices.mutable_data()
    );

    std::copy(
        result.faces.begin(),
        result.faces.end(),
        faces.mutable_data()
    );

    return py::make_tuple(
        vertices,
        faces,
        result.adaptive_patches,
        result.boundary_cells,
        result.base_cells_saved
    );
}



py::tuple simplify_qem_binding(
    py::array_t<
        float,
        py::array::c_style
    > vertices,

    py::array_t<
        std::int64_t,
        py::array::c_style
    > faces,

    const double target_ratio,
    const bool preserve_normals
) {
    if (
        vertices.ndim() != 2
        ||
        vertices.shape(1) != 3
    ) {
        throw std::invalid_argument(
            "vertices must have shape (N, 3)."
        );
    }

    if (
        faces.ndim() != 2
        ||
        faces.shape(1) != 3
    ) {
        throw std::invalid_argument(
            "faces must have shape (M, 3)."
        );
    }

    shrinkwrap::QEMSimplifyResult result;

    {
        py::gil_scoped_release release;

        result =
            shrinkwrap::simplify_qem(
                vertices.data(),
                static_cast<std::size_t>(
                    vertices.shape(0)
                ),

                faces.data(),
                static_cast<std::size_t>(
                    faces.shape(0)
                ),

                target_ratio,
                preserve_normals
            );
    }

    py::array_t<float> out_vertices({
        static_cast<py::ssize_t>(
            result.vertices.size()
            /
            3
        ),
        static_cast<py::ssize_t>(3)
    });

    py::array_t<std::int64_t> out_faces({
        static_cast<py::ssize_t>(
            result.faces.size()
            /
            3
        ),
        static_cast<py::ssize_t>(3)
    });

    std::copy(
        result.vertices.begin(),
        result.vertices.end(),
        out_vertices.mutable_data()
    );

    std::copy(
        result.faces.begin(),
        result.faces.end(),
        out_faces.mutable_data()
    );

    return py::make_tuple(
        out_vertices,
        out_faces,
        result.input_faces,
        result.output_faces,
        result.collapsed_edges
    );
}


PYBIND11_MODULE(
    _native,
    module
) {
    module.doc() =
        "Native C++ compute engine for ShrinkWrap.";

    module.def(
        "project_mesh",
        &project_mesh_binding,

        py::arg("vertices"),
        py::arg("faces"),
        py::arg("top"),
        py::arg("bottom"),
        py::arg("min_x"),
        py::arg("min_y"),
        py::arg("resolution"),

        R"doc(
Project a triangle mesh onto TOP/BOTTOM XY height maps.

Arrays:
    vertices : float64 [N, 3]
    faces    : int64   [M, 3]
    top      : float32 [H, W]
    bottom   : float32 [H, W]

top and bottom are modified in-place.
)doc"
    );

    module.def(
        "propagate_relief",
        &propagate_relief_binding,

        py::arg("relief"),
        py::arg("pcb_mask"),
        py::arg("drop_per_pixel"),

        R"doc(
Propagate relief slope inside the PCB footprint.

relief is modified in-place.
)doc"
    );


    module.def(
        "envelope_to_mesh",
        &envelope_to_mesh_binding,

        py::arg("top"),
        py::arg("bottom"),
        py::arg("phi"),
        py::arg("min_x"),
        py::arg("min_y"),
        py::arg("resolution"),

        "Generate envelope mesh using native C++ mesher."
    );


    module.def(
        "adaptive_envelope_to_mesh",
        &adaptive_envelope_to_mesh_binding,

        py::arg("top"),
        py::arg("bottom"),
        py::arg("phi"),

        py::arg("min_x"),
        py::arg("min_y"),
        py::arg("resolution"),

        py::arg("surface_tolerance") = 0.01,
        py::arg("max_span_cells") = 128,

        "Generate an adaptive conforming envelope mesh."
    );


    module.def(
        "simplify_qem",
        &simplify_qem_binding,

        py::arg("vertices"),
        py::arg("faces"),
        py::arg("target_ratio") = 0.25,
        py::arg("preserve_normals") = true,

        "Simplify using Garland-Heckbert QEM within a physical geometric tolerance."
    );

}
