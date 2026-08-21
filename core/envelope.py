import numpy as np
import trimesh

from scipy import ndimage

from core.heightmap import HeightMap


def build_envelope(
    heightmap: HeightMap,
    smoothing_mm: float = 0.5,
    clearance_mm: float = 0.0,
    bridge_mm: float = 6.0,
    slope_degrees: float = 45.0,
) -> HeightMap:

    resolution = (
        heightmap.resolution
    )

    top_raw = np.asarray(
        heightmap.top,
        dtype=np.float32,
    )

    bottom_raw = np.asarray(
        heightmap.bottom,
        dtype=np.float32,
    )

    top_valid = np.isfinite(
        top_raw
    )

    bottom_valid = np.isfinite(
        bottom_raw
    )

    if not np.any(top_valid):
        raise ValueError(
            "Top height map contains no geometry."
        )

    if not np.any(bottom_valid):
        raise ValueError(
            "Bottom height map contains no geometry."
        )

    #
    # Važno:
    #
    # Prazna mesta na TOP strani se popunjavaju
    # najnižim realnim top nivoom.
    #
    # Prazna mesta na BOTTOM strani se popunjavaju
    # najvišim realnim bottom nivoom.
    #
    # Tako zadržavamo debljinu sklopa.
    #

    top_base = float(
        np.nanmin(
            top_raw
        )
    )

    bottom_base = float(
        np.nanmax(
            bottom_raw
        )
    )

    top_original = np.where(
        top_valid,
        top_raw,
        top_base,
    ).astype(
        np.float32
    )

    bottom_original = np.where(
        bottom_valid,
        bottom_raw,
        bottom_base,
    ).astype(
        np.float32
    )

    #
    # TOP cloth
    #

    top = _build_top_cloth(
        top_original,
        resolution=resolution,
        bridge_mm=bridge_mm,
        smoothing_mm=smoothing_mm,
        slope_degrees=slope_degrees,
    )

    #
    # BOTTOM cloth
    #
    # Pretvorimo donju stranu u pozitivni problem:
    # -Z → napravimo isti cloth → vratimo znak.
    #

    bottom_inverted = (
        -bottom_original
    )

    bottom_cloth_inverted = (
        _build_top_cloth(
            bottom_inverted,
            resolution=resolution,
            bridge_mm=bridge_mm,
            smoothing_mm=smoothing_mm,
            slope_degrees=slope_degrees,
        )
    )

    bottom = (
        -bottom_cloth_inverted
    )

    #
    # Clearance mora da širi model
    # na OBE strane.
    #

    if clearance_mm > 0.0:

        top += clearance_mm

        bottom -= clearance_mm

    #
    # Bezbednost:
    #
    # top nikada ne sme biti ispod originala,
    # bottom nikada ne sme biti iznad originala.
    #

    top = np.maximum(
        top,
        top_original,
    )

    bottom = np.minimum(
        bottom,
        bottom_original,
    )

    return HeightMap(
        top=top.astype(
            np.float32
        ),

        bottom=bottom.astype(
            np.float32
        ),

        min_x=heightmap.min_x,
        min_y=heightmap.min_y,

        resolution=(
            heightmap.resolution
        ),

        base_bottom=float(
            np.min(bottom)
        ),

        base_top=float(
            np.max(top)
        ),
    )


def _build_top_cloth(
    height: np.ndarray,
    resolution: float,
    bridge_mm: float,
    smoothing_mm: float,
    slope_degrees: float,
) -> np.ndarray:

    original = height.copy()

    cloth = original.copy()

    # ==========================================================
    # BRIDGE
    # ==========================================================

    if bridge_mm > 0.0:

        radius = max(
            1,
            int(
                round(
                    bridge_mm
                    / resolution
                )
            ),
        )

        size = (
            radius * 2
            + 1
        )

        cloth = (
            ndimage.grey_closing(
                cloth,
                size=(
                    size,
                    size,
                ),
                mode="nearest",
            )
        )

    # ==========================================================
    # SLOPE
    # ==========================================================

    slope_degrees = float(
        np.clip(
            slope_degrees,
            2.0,
            88.0,
        )
    )

    slope = np.tan(
        np.radians(
            slope_degrees
        )
    )

    drop_per_pixel = (
        slope
        * resolution
    )

    cloth = _apply_slope_envelope(
        cloth,
        drop_per_pixel,
    )

    # ==========================================================
    # SMOOTH
    # ==========================================================

    if smoothing_mm > 0.0:

        sigma = (
            smoothing_mm
            / resolution
        )

        smoothed = (
            ndimage.gaussian_filter(
                cloth,
                sigma=sigma,
                mode="nearest",
            )
        )

        cloth = np.maximum(
            original,
            smoothed,
        )

    #
    # Platno nikada kroz objekat.
    #

    cloth = np.maximum(
        cloth,
        original,
    )

    return cloth


def _apply_slope_envelope(
    height: np.ndarray,
    drop_per_pixel: float,
) -> np.ndarray:

    result = height.copy()

    h, w = result.shape

    #
    # X+
    #

    for y in range(h):

        for x in range(
            1,
            w,
        ):

            result[y, x] = max(
                result[y, x],
                result[y, x - 1]
                - drop_per_pixel,
            )

        #
        # X-
        #

        for x in range(
            w - 2,
            -1,
            -1,
        ):

            result[y, x] = max(
                result[y, x],
                result[y, x + 1]
                - drop_per_pixel,
            )

    #
    # Y+
    #

    for x in range(w):

        for y in range(
            1,
            h,
        ):

            result[y, x] = max(
                result[y, x],
                result[y - 1, x]
                - drop_per_pixel,
            )

        #
        # Y-
        #

        for y in range(
            h - 2,
            -1,
            -1,
        ):

            result[y, x] = max(
                result[y, x],
                result[y + 1, x]
                - drop_per_pixel,
            )

    return result


def envelope_to_mesh(
    envelope: HeightMap,
) -> trimesh.Trimesh:
    """
    Generiše ZATVOREN solid mesh:

        top surface
        bottom surface
        front wall
        back wall
        left wall
        right wall
    """

    top = np.asarray(
        envelope.top,
        dtype=np.float32,
    )

    bottom = np.asarray(
        envelope.bottom,
        dtype=np.float32,
    )

    h, w = top.shape

    resolution = (
        envelope.resolution
    )

    vertex_count_per_side = (
        h * w
    )

    #
    # TOP + BOTTOM vertices
    #

    vertices = np.zeros(
        (
            vertex_count_per_side * 2,
            3,
        ),
        dtype=np.float32,
    )

    for y in range(h):

        py = (
            envelope.min_y
            + y * resolution
        )

        for x in range(w):

            px = (
                envelope.min_x
                + x * resolution
            )

            i = (
                y * w
                + x
            )

            #
            # top vertex
            #

            vertices[i] = (
                px,
                py,
                top[y, x],
            )

            #
            # bottom vertex
            #

            vertices[
                vertex_count_per_side
                + i
            ] = (
                px,
                py,
                bottom[y, x],
            )

    faces = []

    offset = (
        vertex_count_per_side
    )

    # ==========================================================
    # TOP
    # ==========================================================

    for y in range(
        h - 1
    ):

        for x in range(
            w - 1
        ):

            a = y * w + x
            b = a + 1

            c = (
                (y + 1) * w
                + x
            )

            d = c + 1

            faces.append(
                [a, b, d]
            )

            faces.append(
                [a, d, c]
            )

    # ==========================================================
    # BOTTOM
    #
    # Obrnut winding.
    # ==========================================================

    for y in range(
        h - 1
    ):

        for x in range(
            w - 1
        ):

            a = (
                offset
                + y * w
                + x
            )

            b = a + 1

            c = (
                offset
                + (y + 1) * w
                + x
            )

            d = c + 1

            faces.append(
                [a, d, b]
            )

            faces.append(
                [a, c, d]
            )

    # ==========================================================
    # FRONT + BACK WALL
    # ==========================================================

    for x in range(
        w - 1
    ):

        #
        # y = 0
        #

        t0 = x
        t1 = x + 1

        b0 = (
            offset + x
        )

        b1 = (
            offset + x + 1
        )

        faces.append(
            [t0, b1, t1]
        )

        faces.append(
            [t0, b0, b1]
        )

        #
        # y = h - 1
        #

        row = (
            (h - 1)
            * w
        )

        t0 = (
            row + x
        )

        t1 = (
            row + x + 1
        )

        b0 = (
            offset + t0
        )

        b1 = (
            offset + t1
        )

        faces.append(
            [t0, t1, b1]
        )

        faces.append(
            [t0, b1, b0]
        )

    # ==========================================================
    # LEFT + RIGHT WALL
    # ==========================================================

    for y in range(
        h - 1
    ):

        #
        # x = 0
        #

        t0 = (
            y * w
        )

        t1 = (
            (y + 1) * w
        )

        b0 = (
            offset + t0
        )

        b1 = (
            offset + t1
        )

        faces.append(
            [t0, t1, b1]
        )

        faces.append(
            [t0, b1, b0]
        )

        #
        # x = w - 1
        #

        t0 = (
            y * w
            + (w - 1)
        )

        t1 = (
            (y + 1) * w
            + (w - 1)
        )

        b0 = (
            offset + t0
        )

        b1 = (
            offset + t1
        )

        faces.append(
            [t0, b1, t1]
        )

        faces.append(
            [t0, b0, b1]
        )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(
            faces,
            dtype=np.int64,
        ),
        process=False,
    )

    mesh.remove_unreferenced_vertices()

    mesh.fix_normals()

    return mesh