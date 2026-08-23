from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.envelope import (
    Envelope,
    envelope_to_mesh,
    _envelope_to_mesh_python,
)


def main():
    resolution = 0.25

    h = 180
    w = 240

    pcb_mask = np.zeros(
        (h, w),
        dtype=bool,
    )

    pcb_mask[
        15:h - 15,
        20:w - 20
    ] = True

    top = np.full(
        (h, w),
        0.8,
        dtype=np.float32,
    )

    bottom = np.full(
        (h, w),
        -0.8,
        dtype=np.float32,
    )

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    bump = np.exp(
        -(
            (xx - w * 0.5) ** 2
            +
            (yy - h * 0.5) ** 2
        )
        /
        500.0
    )

    top += (
        bump * 6.0
    ).astype(
        np.float32
    )

    env = Envelope(
        top=top,
        bottom=bottom,
        pcb_mask=pcb_mask,
        holes=tuple(),

        min_x=0.0,
        min_y=0.0,
        resolution=resolution,

        pcb_top_z=0.8,
        pcb_bottom_z=-0.8,
        pcb_thickness=1.6,
    )

    start = time.perf_counter()

    py_mesh = _envelope_to_mesh_python(
        env
    )

    py_time = (
        time.perf_counter()
        -
        start
    )

    start = time.perf_counter()

    cpp_mesh = envelope_to_mesh(
        env
    )

    cpp_time = (
        time.perf_counter()
        -
        start
    )

    print()
    print("=== COUNTS ===")

    print(
        "Python:",
        len(py_mesh.vertices),
        "vertices /",
        len(py_mesh.faces),
        "faces",
    )

    print(
        "C++:",
        len(cpp_mesh.vertices),
        "vertices /",
        len(cpp_mesh.faces),
        "faces",
    )

    print()
    print("=== BOUNDS ===")

    print(
        "Python:",
        py_mesh.bounds,
    )

    print(
        "C++:",
        cpp_mesh.bounds,
    )

    print()
    print("=== GEOMETRY ===")

    print(
        "Python watertight:",
        py_mesh.is_watertight,
    )

    print(
        "C++ watertight:",
        cpp_mesh.is_watertight,
    )

    print(
        "Python area:",
        py_mesh.area,
    )

    print(
        "C++ area:",
        cpp_mesh.area,
    )

    print(
        "Python volume:",
        py_mesh.volume,
    )

    print(
        "C++ volume:",
        cpp_mesh.volume,
    )

    print()
    print("=== PERFORMANCE ===")

    print(
        f"Python: {py_time:.4f}s"
    )

    print(
        f"C++:    {cpp_time:.4f}s"
    )

    print(
        f"Speedup: {py_time / cpp_time:.1f}x"
    )

    assert (
        len(py_mesh.vertices)
        ==
        len(cpp_mesh.vertices)
    )

    assert (
        len(py_mesh.faces)
        ==
        len(cpp_mesh.faces)
    )

    assert np.allclose(
        py_mesh.bounds,
        cpp_mesh.bounds,
        atol=1e-5,
    )

    assert np.isclose(
        py_mesh.area,
        cpp_mesh.area,
        rtol=1e-5,
    )

    assert np.isclose(
        abs(py_mesh.volume),
        abs(cpp_mesh.volume),
        rtol=1e-5,
    )

    print()
    print(
        "PASS: C++ mesher matches Python geometry."
    )


if __name__ == "__main__":
    main()
