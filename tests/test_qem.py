from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import numpy as np
import trimesh

from core import _native


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mesh",
        type=Path,
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    mesh = trimesh.load_mesh(
        args.mesh,
        process=False,
    )

    #
    # STL stores every triangle independently and therefore
    # normally contains three vertex records per face.
    #
    # QEM edge collapse requires indexed/shared topology.
    #
    # Weld coincident STL vertices before handing the mesh
    # to CGAL.
    #
    print(
        f"Raw STL: "
        f"{len(mesh.vertices):,} vertices / "
        f"{len(mesh.faces):,} triangles"
    )

    mesh.merge_vertices()

    mesh.remove_unreferenced_vertices()

    print(
        f"Welded:  "
        f"{len(mesh.vertices):,} vertices / "
        f"{len(mesh.faces):,} triangles"
    )

    vertices = np.ascontiguousarray(
        mesh.vertices,
        dtype=np.float32,
    )

    faces = np.ascontiguousarray(
        mesh.faces,
        dtype=np.int64,
    )

    print(
        f"Input: "
        f"{len(vertices):,} vertices / "
        f"{len(faces):,} triangles"
    )

    start = time.perf_counter()

    (
        out_vertices,
        out_faces,
        input_faces,
        output_faces,
        collapsed,
    ) = _native.simplify_qem(
        vertices,
        faces,
        args.ratio,
        True,
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    result = trimesh.Trimesh(
        vertices=out_vertices,
        faces=out_faces,
        process=False,
    )

    reduction = (
        1.0
        -
        output_faces
        /
        input_faces
    ) * 100.0

    print()
    print(
        f"Output: "
        f"{len(out_vertices):,} vertices / "
        f"{len(out_faces):,} triangles"
    )

    print(
        f"Reduction: {reduction:.2f}%"
    )

    print(
        f"Collapsed edges: {collapsed:,}"
    )

    print(
        f"Time: {elapsed:.3f}s"
    )

    print(
        f"Watertight: {result.is_watertight}"
    )

    print(
        "Bounds:"
    )

    print(
        result.bounds
    )

    output = Path(
        "qem_test.stl"
    )

    result.export(
        output
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
