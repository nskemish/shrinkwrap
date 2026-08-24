from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import trimesh

from core import _native
from core.envelope import (
    Envelope,
    envelope_to_mesh_adaptive,
)


def make_reference_mesh(
    resolution: float = 0.05,
):
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

    mesh = envelope_to_mesh_adaptive(
        env,
        surface_tolerance=0.01,
        max_span_cells=128,
    )

    return mesh


def simplify(
    vertices: np.ndarray,
    faces: np.ndarray,
    ratio: float,
):
    start = time.perf_counter()

    (
        out_vertices,
        out_faces,
        _,
        _,
        _,
    ) = _native.simplify_qem(
        vertices,
        faces,
        float(ratio),
        False,
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    return (
        np.ascontiguousarray(
            out_vertices,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            out_faces,
            dtype=np.int64,
        ),
        elapsed,
    )


def sampled_error(
    reference: trimesh.Trimesh,
    simplified: trimesh.Trimesh,
    samples: int,
):
    np.random.seed(1234)

    ref_points, _ = (
        trimesh.sample.sample_surface(
            reference,
            samples,
        )
    )

    np.random.seed(5678)

    simp_points, _ = (
        trimesh.sample.sample_surface(
            simplified,
            samples,
        )
    )

    _, d_ref_to_simp, _ = (
        trimesh.proximity.closest_point_naive(
            simplified,
            ref_points,
        )
    )

    _, d_simp_to_ref, _ = (
        trimesh.proximity.closest_point_naive(
            reference,
            simp_points,
        )
    )

    d_ref_to_simp = np.asarray(
        d_ref_to_simp,
        dtype=np.float64,
    )

    d_simp_to_ref = np.asarray(
        d_simp_to_ref,
        dtype=np.float64,
    )

    maximum = max(
        float(
            np.max(
                d_ref_to_simp
            )
        ),
        float(
            np.max(
                d_simp_to_ref
            )
        ),
    )

    p99 = max(
        float(
            np.percentile(
                d_ref_to_simp,
                99.0,
            )
        ),
        float(
            np.percentile(
                d_simp_to_ref,
                99.0,
            )
        ),
    )

    rms = float(
        np.sqrt(
            (
                np.mean(
                    d_ref_to_simp ** 2
                )
                +
                np.mean(
                    d_simp_to_ref ** 2
                )
            )
            /
            2.0
        )
    )

    return {
        "max": maximum,
        "p99": p99,
        "rms": rms,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    print(
        "Generating reference adaptive mesh..."
    )

    reference = make_reference_mesh()

    print(
        f"Reference: "
        f"{len(reference.vertices):,} vertices / "
        f"{len(reference.faces):,} triangles"
    )

    print(
        "Reference watertight:",
        reference.is_watertight,
    )

    vertices = np.ascontiguousarray(
        reference.vertices,
        dtype=np.float32,
    )

    faces = np.ascontiguousarray(
        reference.faces,
        dtype=np.int64,
    )

    low = float(
        args.min_ratio
    )

    high = float(
        args.max_ratio
    )

    best = None

    print()
    print("=" * 88)
    print("QEM TOLERANCE SEARCH")
    print("=" * 88)

    print(
        f"Tolerance: {args.tolerance:.6f} mm"
    )

    print(
        f"Samples:   {args.samples:,}"
    )

    print()

    for iteration in range(
        args.iterations
    ):
        ratio = (
            low
            +
            high
        ) * 0.5

        out_v, out_f, qem_time = simplify(
            vertices,
            faces,
            ratio,
        )

        simplified = trimesh.Trimesh(
            vertices=out_v,
            faces=out_f,
            process=False,
        )

        check_start = (
            time.perf_counter()
        )

        error = sampled_error(
            reference,
            simplified,
            args.samples,
        )

        check_time = (
            time.perf_counter()
            -
            check_start
        )

        passed = (
            error["max"]
            <=
            args.tolerance
        )

        reduction = (
            1.0
            -
            len(out_f)
            /
            len(faces)
        ) * 100.0

        print(
            f"[{iteration + 1:02d}] "
            f"ratio={ratio:.6f} | "
            f"faces={len(out_f):,} | "
            f"reduction={reduction:6.2f}% | "
            f"max={error['max']:.6f} mm | "
            f"p99={error['p99']:.6f} mm | "
            f"rms={error['rms']:.6f} mm | "
            f"QEM={qem_time:.3f}s | "
            f"check={check_time:.3f}s | "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if passed:
            best = {
                "ratio": ratio,
                "vertices": out_v,
                "faces": out_f,
                "error": error,
            }

            # Smaller ratio = more aggressive.
            high = ratio

        else:
            low = ratio

    print()
    print("=" * 88)
    print("RESULT")
    print("=" * 88)

    if best is None:
        print(
            "No tested ratio met tolerance."
        )
        raise SystemExit(1)

    result = trimesh.Trimesh(
        vertices=best["vertices"],
        faces=best["faces"],
        process=False,
    )

    reduction = (
        1.0
        -
        len(best["faces"])
        /
        len(faces)
    ) * 100.0

    print(
        f"Ratio:      {best['ratio']:.6f}"
    )

    print(
        f"Faces:      "
        f"{len(faces):,} -> "
        f"{len(best['faces']):,}"
    )

    print(
        f"Reduction:  {reduction:.2f}%"
    )

    print(
        f"Max error:  "
        f"{best['error']['max']:.6f} mm"
    )

    print(
        f"P99 error:  "
        f"{best['error']['p99']:.6f} mm"
    )

    print(
        f"RMS error:  "
        f"{best['error']['rms']:.6f} mm"
    )

    print(
        f"Watertight: {result.is_watertight}"
    )

    result.export(
        "qem_tolerance_test.stl"
    )

    print(
        "Saved: qem_tolerance_test.stl"
    )


if __name__ == "__main__":
    main()
