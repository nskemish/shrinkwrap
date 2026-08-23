from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import trimesh

from core import _native
from core.heightmap import _project_mesh_python


def main():
    mesh = trimesh.creation.box(
        extents=(
            20.0,
            15.0,
            1.6,
        )
    )

    resolution = 0.1
    min_x = -11.0
    min_y = -8.5

    width = 221
    height = 171

    py_top = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32,
    )

    py_bottom = np.full(
        (height, width),
        np.inf,
        dtype=np.float32,
    )

    cpp_top = py_top.copy()
    cpp_bottom = py_bottom.copy()

    _project_mesh_python(
        mesh=mesh,
        top=py_top,
        bottom=py_bottom,
        min_x=min_x,
        min_y=min_y,
        resolution=resolution,
    )

    vertices = np.ascontiguousarray(
        mesh.vertices,
        dtype=np.float64,
    )

    faces = np.ascontiguousarray(
        mesh.faces,
        dtype=np.int64,
    )

    _native.project_mesh(
        vertices,
        faces,
        cpp_top,
        cpp_bottom,
        min_x,
        min_y,
        resolution,
    )

    py_top_hit = np.isfinite(py_top)
    cpp_top_hit = np.isfinite(cpp_top)

    py_bottom_hit = np.isfinite(py_bottom)
    cpp_bottom_hit = np.isfinite(cpp_bottom)

    print()
    print("=== MASK ===")

    print(
        "top equal:",
        np.array_equal(
            py_top_hit,
            cpp_top_hit,
        ),
    )

    print(
        "bottom equal:",
        np.array_equal(
            py_bottom_hit,
            cpp_bottom_hit,
        ),
    )

    top_valid = (
        py_top_hit
        &
        cpp_top_hit
    )

    bottom_valid = (
        py_bottom_hit
        &
        cpp_bottom_hit
    )

    top_error = (
        np.max(
            np.abs(
                py_top[top_valid]
                -
                cpp_top[top_valid]
            )
        )
        if np.any(top_valid)
        else 0.0
    )

    bottom_error = (
        np.max(
            np.abs(
                py_bottom[bottom_valid]
                -
                cpp_bottom[bottom_valid]
            )
        )
        if np.any(bottom_valid)
        else 0.0
    )

    print()
    print("=== ERROR ===")

    print(
        "top max error:",
        top_error,
    )

    print(
        "bottom max error:",
        bottom_error,
    )

    assert np.array_equal(
        py_top_hit,
        cpp_top_hit,
    )

    assert np.array_equal(
        py_bottom_hit,
        cpp_bottom_hit,
    )

    assert top_error < 1e-5
    assert bottom_error < 1e-5

    print()
    print("PASS: C++ rasterizer matches Python.")


if __name__ == "__main__":
    main()
