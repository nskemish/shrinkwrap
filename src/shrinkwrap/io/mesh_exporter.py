from pathlib import Path

import trimesh


SUPPORTED_EXPORTS = {
    ".stl",
    ".obj",
    ".ply",
    ".glb",
}


def export_mesh(
    mesh: trimesh.Trimesh,
    path: str | Path,
):
    path = Path(path)

    extension = (
        path.suffix.lower()
    )

    if extension not in SUPPORTED_EXPORTS:
        raise ValueError(
            f"Unsupported export format: {extension}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh.export(
        path
    )