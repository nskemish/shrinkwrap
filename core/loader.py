from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTENSIONS = {
    ".stl",
    ".obj",
    ".ply",
    ".glb",
    ".gltf",
}


class MeshLoadError(RuntimeError):
    pass


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)

    if not path.exists():
        raise MeshLoadError(
            f"File does not exist: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise MeshLoadError(
            f"Unsupported file format: {path.suffix}"
        )

    try:
        loaded = trimesh.load(
            path,
            force="scene",
        )
    except Exception as exc:
        raise MeshLoadError(
            f"Could not load mesh:\n{exc}"
        ) from exc

    mesh = _scene_to_mesh(loaded)

    if mesh.is_empty:
        raise MeshLoadError(
            "Loaded mesh is empty."
        )

    mesh.remove_unreferenced_vertices()

    return mesh


def _scene_to_mesh(
    loaded,
) -> trimesh.Trimesh:

    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()

    if not isinstance(loaded, trimesh.Scene):
        raise MeshLoadError(
            "Loaded object is not a triangle mesh."
        )

    meshes = []

    #
    # Scene može sadržati više komponenti.
    #
    # dump(concatenate=False) primenjuje transformacije
    # scene graph-a, što je bitno kod GLB/GLTF fajlova.
    #
    try:
        dumped = loaded.dump(
            concatenate=False
        )
    except Exception:
        dumped = list(
            loaded.geometry.values()
        )

    for geometry in dumped:

        if not isinstance(
            geometry,
            trimesh.Trimesh,
        ):
            continue

        if geometry.is_empty:
            continue

        meshes.append(
            geometry.copy()
        )

    if not meshes:
        raise MeshLoadError(
            "Scene does not contain any triangle meshes."
        )

    if len(meshes) == 1:
        return meshes[0]

    return trimesh.util.concatenate(
        meshes
    )


def get_mesh_info(
    mesh: trimesh.Trimesh,
) -> dict:

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )

    size = bounds[1] - bounds[0]

    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),

        "width": float(size[0]),
        "depth": float(size[1]),
        "height": float(size[2]),

        "min": bounds[0].copy(),
        "max": bounds[1].copy(),

        "center": np.asarray(
            mesh.bounding_box.centroid,
            dtype=np.float64,
        ),
    }