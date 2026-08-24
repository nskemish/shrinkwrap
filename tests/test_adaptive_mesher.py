from pathlib import Path
import sys
import time

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

import numpy as np

from core import _native

from core.envelope import (
    Envelope,
    envelope_to_mesh,
    envelope_to_mesh_adaptive,
)


def main():

    resolution = 0.05

    h = 800
    w = 600

    pcb_mask = np.zeros(
        (h, w),
        dtype=bool,
    )

    pcb_mask[
        20:h - 20,
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

    # ========================================================
    # Smooth curved component
    # ========================================================

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    cx = w * 0.5
    cy = h * 0.5

    radius = 80.0

    d2 = (
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
    )

    bump = np.exp(
        -d2
        /
        (
            2.0
            *
            radius
            *
            radius
        )
    )

    top += (
        bump * 8.0
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

    # ========================================================
    # UNIFORM
    # ========================================================

    start = time.perf_counter()

    uniform = envelope_to_mesh(
        env
    )

    uniform_time = (
        time.perf_counter()
        -
        start
    )

    # ========================================================
    # ADAPTIVE
    # ========================================================

    start = time.perf_counter()

    adaptive = (
        envelope_to_mesh_adaptive(
            env,
            surface_tolerance=0.01,
            max_span_cells=128,
        )
    )

    adaptive_time = (
        time.perf_counter()
        -
        start
    )

    print()
    print("=" * 72)
    print("COUNTS")
    print("=" * 72)

    print(
        f"Uniform:  "
        f"{len(uniform.vertices):,} vertices / "
        f"{len(uniform.faces):,} triangles"
    )

    print(
        f"Adaptive: "
        f"{len(adaptive.vertices):,} vertices / "
        f"{len(adaptive.faces):,} triangles"
    )

    reduction = (
        1.0
        -
        len(adaptive.faces)
        /
        len(uniform.faces)
    ) * 100.0

    print(
        f"Triangle reduction: "
        f"{reduction:.2f}%"
    )

    print()
    print("=" * 72)
    print("GEOMETRY")
    print("=" * 72)

    print(
        "Uniform bounds:"
    )
    print(
        uniform.bounds
    )

    print(
        "Adaptive bounds:"
    )
    print(
        adaptive.bounds
    )

    print(
        "Uniform area:",
        uniform.area,
    )

    print(
        "Adaptive area:",
        adaptive.area,
    )

    print(
        "Uniform volume:",
        uniform.volume,
    )

    print(
        "Adaptive volume:",
        adaptive.volume,
    )

    area_error = (
        abs(
            adaptive.area
            -
            uniform.area
        )
        /
        uniform.area
        *
        100.0
    )

    volume_error = (
        abs(
            abs(adaptive.volume)
            -
            abs(uniform.volume)
        )
        /
        abs(uniform.volume)
        *
        100.0
    )

    print(
        f"Area error:   "
        f"{area_error:.5f}%"
    )

    print(
        f"Volume error: "
        f"{volume_error:.5f}%"
    )

    # ========================================================
    # QEM SIMPLIFICATION
    # ========================================================

    print()
    print("=" * 72)
    print("QEM")
    print("=" * 72)

    qem_vertices = np.ascontiguousarray(
        adaptive.vertices,
        dtype=np.float32,
    )

    qem_faces = np.ascontiguousarray(
        adaptive.faces,
        dtype=np.int64,
    )

    qem_start = time.perf_counter()

    (
        simplified_vertices,
        simplified_faces,
        qem_input_faces,
        qem_output_faces,
        collapsed_edges,
    ) = _native.simplify_qem(
        qem_vertices,
        qem_faces,
        0.25,
        False,
    )

    qem_time = (
        time.perf_counter()
        -
        qem_start
    )

    import trimesh

    simplified = trimesh.Trimesh(
        vertices=simplified_vertices,
        faces=simplified_faces,
        process=False,
    )

    qem_reduction = (
        1.0
        -
        qem_output_faces
        /
        qem_input_faces
    ) * 100.0

    print(
        f"Input:     "
        f"{qem_input_faces:,} triangles"
    )

    print(
        f"Output:    "
        f"{qem_output_faces:,} triangles"
    )

    print(
        f"Reduction: "
        f"{qem_reduction:.2f}%"
    )

    print(
        f"Collapsed: "
        f"{collapsed_edges:,} edges"
    )

    print(
        f"QEM time:  "
        f"{qem_time:.4f}s"
    )

    print(
        "Watertight:",
        simplified.is_watertight,
    )

    print(
        "Bounds:"
    )

    print(
        simplified.bounds
    )

    print()
    print("=" * 72)
    print("PERFORMANCE")
    print("=" * 72)

    print(
        f"Uniform:  "
        f"{uniform_time:.4f}s"
    )

    print(
        f"Adaptive: "
        f"{adaptive_time:.4f}s"
    )

    print()
    print("=" * 72)
    print("TOPOLOGY")
    print("=" * 72)

    print(
        "Checking adaptive watertightness only..."
    )

    print(
        "Adaptive watertight:",
        adaptive.is_watertight,
    )

    assert np.allclose(
        uniform.bounds,
        adaptive.bounds,
        atol=resolution,
    )

    assert volume_error < 1.0

    # Adaptive mesher is allowed to match the uniform mesh
    # in difficult/high-detail cases. Final reduction is now
    # also handled by QEM.
    assert (
        len(adaptive.faces)
        <=
        len(uniform.faces)
    )

    assert (
        qem_output_faces
        <
        len(adaptive.faces)
    )

    assert simplified.is_watertight

    assert adaptive.is_watertight

    print()
    print(
        "PASS: adaptive mesher."
    )


if __name__ == "__main__":
    main()
