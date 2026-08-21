from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class HeightMap:
    top: np.ndarray
    bottom: np.ndarray

    min_x: float
    min_y: float

    resolution: float

    base_bottom: float
    base_top: float

    @property
    def z(self):
        # compatibility sa starijim kodom
        return self.top

    @property
    def width(self) -> int:
        return self.top.shape[1]

    @property
    def height(self) -> int:
        return self.top.shape[0]

    @property
    def max_x(self) -> float:
        return (
            self.min_x
            + (self.width - 1)
            * self.resolution
        )

    @property
    def max_y(self) -> float:
        return (
            self.min_y
            + (self.height - 1)
            * self.resolution
        )


def generate_heightmap(
    mesh: trimesh.Trimesh,
    resolution: float,
) -> HeightMap:

    if resolution <= 0:
        raise ValueError(
            "Resolution must be > 0"
        )

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )

    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]

    width = max(
        2,
        int(
            np.ceil(
                (max_x - min_x)
                / resolution
            )
        ) + 1,
    )

    height = max(
        2,
        int(
            np.ceil(
                (max_y - min_y)
                / resolution
            )
        ) + 1,
    )

    top = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32,
    )

    bottom = np.full(
        (height, width),
        np.inf,
        dtype=np.float32,
    )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    for face in faces:

        tri = vertices[face]

        _rasterize_triangle(
            tri=tri,
            top=top,
            bottom=bottom,
            min_x=min_x,
            min_y=min_y,
            resolution=resolution,
        )

    top[
        ~np.isfinite(top)
    ] = np.nan

    bottom[
        ~np.isfinite(bottom)
    ] = np.nan

    return HeightMap(
        top=top,
        bottom=bottom,

        min_x=float(min_x),
        min_y=float(min_y),

        resolution=float(
            resolution
        ),

        base_bottom=float(min_z),
        base_top=float(max_z),
    )


def _rasterize_triangle(
    tri: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    min_x: float,
    min_y: float,
    resolution: float,
):

    p0 = tri[0]
    p1 = tri[1]
    p2 = tri[2]

    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    denominator = (
        (y1 - y2) * (x0 - x2)
        +
        (x2 - x1) * (y0 - y2)
    )

    #
    # Vertikalni trougao nema površinu
    # kada se projektuje na XY.
    #
    if abs(denominator) < 1e-12:
        return

    tri_min_x = min(
        x0,
        x1,
        x2,
    )

    tri_max_x = max(
        x0,
        x1,
        x2,
    )

    tri_min_y = min(
        y0,
        y1,
        y2,
    )

    tri_max_y = max(
        y0,
        y1,
        y2,
    )

    ix0 = max(
        0,
        int(
            np.floor(
                (tri_min_x - min_x)
                / resolution
            )
        ),
    )

    ix1 = min(
        top.shape[1] - 1,
        int(
            np.ceil(
                (tri_max_x - min_x)
                / resolution
            )
        ),
    )

    iy0 = max(
        0,
        int(
            np.floor(
                (tri_min_y - min_y)
                / resolution
            )
        ),
    )

    iy1 = min(
        top.shape[0] - 1,
        int(
            np.ceil(
                (tri_max_y - min_y)
                / resolution
            )
        ),
    )

    for iy in range(
        iy0,
        iy1 + 1,
    ):

        y = (
            min_y
            + iy * resolution
        )

        for ix in range(
            ix0,
            ix1 + 1,
        ):

            x = (
                min_x
                + ix * resolution
            )

            a = (
                (y1 - y2)
                * (x - x2)
                +
                (x2 - x1)
                * (y - y2)
            ) / denominator

            b = (
                (y2 - y0)
                * (x - x2)
                +
                (x0 - x2)
                * (y - y2)
            ) / denominator

            c = (
                1.0
                - a
                - b
            )

            epsilon = -1e-6

            if (
                a >= epsilon
                and
                b >= epsilon
                and
                c >= epsilon
            ):

                z = (
                    a * z0
                    +
                    b * z1
                    +
                    c * z2
                )

                #
                # Gornja koža
                #
                if (
                    not np.isfinite(
                        top[iy, ix]
                    )
                    or
                    z > top[iy, ix]
                ):
                    top[
                        iy,
                        ix
                    ] = z

                #
                # Donja koža
                #
                if (
                    not np.isfinite(
                        bottom[iy, ix]
                    )
                    or
                    z < bottom[iy, ix]
                ):
                    bottom[
                        iy,
                        ix
                    ] = z