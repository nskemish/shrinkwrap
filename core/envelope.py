from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

from core.drill import DrillHole
from core.heightmap import HeightMap


# ============================================================
# DATA
# ============================================================


@dataclass
class Envelope:
    top: np.ndarray
    bottom: np.ndarray

    #
    # Finalna XY oblast materijala:
    #
    #     PCB
    #     minus NPTH
    #
    solid_mask: np.ndarray

    min_x: float
    min_y: float
    resolution: float

    pcb_top_z: float
    pcb_bottom_z: float
    pcb_thickness: float


# ============================================================
# PUBLIC
# ============================================================


def build_envelope(
    heightmap: HeightMap,
    smoothing_mm: float = 0.4,
    clearance_mm: float = 0.0,
    bridge_mm: float = 4.0,
    slope_degrees: float = 55.0,
    holes: list[DrillHole] | None = None,
    hole_keepout_mm: float = 1.5,
) -> Envelope:
    """
    Finalni model se sastoji od:

        RIGID PCB
        +
        TOP COMPONENT RELIEF
        +
        BOTTOM COMPONENT RELIEF
        -
        NPTH HOLES

    PCB baza nikada ne prolazi kroz cloth algoritam.

    Cloth radi samo nad pozitivnim relief-om iznad
    pcb_top_z i ispod pcb_bottom_z.
    """

    holes = list(
        holes or []
    )

    raw_top = np.asarray(
        heightmap.raw_top,
        dtype=np.float32,
    )

    raw_bottom = np.asarray(
        heightmap.raw_bottom,
        dtype=np.float32,
    )

    pcb_mask = np.asarray(
        heightmap.pcb_mask,
        dtype=bool,
    )

    if not np.any(
        pcb_mask
    ):
        raise ValueError(
            "PCB mask is empty."
        )

    pcb_top_z = float(
        heightmap.pcb_top_z
    )

    pcb_bottom_z = float(
        heightmap.pcb_bottom_z
    )

    #
    # ========================================================
    # COMPONENT RELIEF
    # ========================================================
    #
    # Na mestima gde projection nije pogodio geometriju
    # relief je nula.
    #
    # PCB sam po sebi takođe daje relief = 0.
    #

    top_relief = np.zeros_like(
        raw_top,
        dtype=np.float32,
    )

    bottom_relief = np.zeros_like(
        raw_bottom,
        dtype=np.float32,
    )

    valid_top = np.isfinite(
        raw_top
    )

    valid_bottom = np.isfinite(
        raw_bottom
    )

    #
    # Sve iznad gornje PCB ravni.
    #
    top_component = (
        pcb_mask
        &
        valid_top
        &
        (
            raw_top
            >
            pcb_top_z
            +
            1e-4
        )
    )

    top_relief[
        top_component
    ] = (
        raw_top[
            top_component
        ]
        -
        pcb_top_z
    )

    #
    # Sve ispod donje PCB ravni.
    #
    bottom_component = (
        pcb_mask
        &
        valid_bottom
        &
        (
            raw_bottom
            <
            pcb_bottom_z
            -
            1e-4
        )
    )

    bottom_relief[
        bottom_component
    ] = (
        pcb_bottom_z
        -
        raw_bottom[
            bottom_component
        ]
    )

    #
    # ========================================================
    # CLOTH
    # ========================================================
    #

    top_cloth = _build_relief_cloth(
        relief=top_relief,
        pcb_mask=pcb_mask,

        resolution=heightmap.resolution,

        bridge_mm=bridge_mm,
        smoothing_mm=smoothing_mm,
        slope_degrees=slope_degrees,
    )

    bottom_cloth = _build_relief_cloth(
        relief=bottom_relief,
        pcb_mask=pcb_mask,

        resolution=heightmap.resolution,

        bridge_mm=bridge_mm,
        smoothing_mm=smoothing_mm,
        slope_degrees=slope_degrees,
    )

    #
    # ========================================================
    # RIGID PCB EDGE
    # ========================================================
    #
    # Sam boundary PCB-a mora ostati upravo PCB top/bottom.
    #
    # Znači component relief ne sme da napravi visoki
    # vertikalni zid na samoj ivici.
    #
    boundary = _mask_boundary(
        pcb_mask
    )

    top_cloth[
        boundary
    ] = 0.0

    bottom_cloth[
        boundary
    ] = 0.0

    #
    # ========================================================
    # NPTH KEEPOUT
    # ========================================================
    #
    # Screw/mounting zona:
    #
    # cloth se uklanja u:
    #
    #     hole radius + keepout
    #
    # ali PCB visina ostaje tačno pcb_top / pcb_bottom.
    #
    if holes and hole_keepout_mm > 0.0:

        keepout_mask = _make_hole_mask(
            heightmap=heightmap,
            holes=holes,
            extra_radius_mm=hole_keepout_mm,
        )

        top_cloth[
            keepout_mask
        ] = 0.0

        bottom_cloth[
            keepout_mask
        ] = 0.0

    #
    # ========================================================
    # FINAL SURFACES
    # ========================================================
    #

    top = (
        pcb_top_z
        +
        top_cloth
    )

    bottom = (
        pcb_bottom_z
        -
        bottom_cloth
    )

    #
    # Clearance se primenjuje SAMO tamo gde postoji relief.
    #
    # Ne povećavamo samu rigidnu PCB debljinu.
    #
    if clearance_mm > 0.0:

        top_has_relief = (
            top_cloth
            >
            1e-5
        )

        bottom_has_relief = (
            bottom_cloth
            >
            1e-5
        )

        top[
            top_has_relief
        ] += clearance_mm

        bottom[
            bottom_has_relief
        ] -= clearance_mm

    #
    # ========================================================
    # NPTH HOLES
    # ========================================================
    #

    npth_mask = _make_hole_mask(
        heightmap=heightmap,
        holes=holes,
        extra_radius_mm=0.0,
    )

    solid_mask = (
        pcb_mask
        &
        ~npth_mask
    )

    print(
        "[envelope] "
        f"top relief max={float(np.max(top_cloth)):.3f} mm | "
        f"bottom relief max={float(np.max(bottom_cloth)):.3f} mm | "
        f"NPTH={len(holes)}"
    )

    return Envelope(
        top=top.astype(
            np.float32
        ),

        bottom=bottom.astype(
            np.float32
        ),

        solid_mask=solid_mask,

        min_x=heightmap.min_x,
        min_y=heightmap.min_y,

        resolution=heightmap.resolution,

        pcb_top_z=pcb_top_z,
        pcb_bottom_z=pcb_bottom_z,

        pcb_thickness=(
            pcb_top_z
            -
            pcb_bottom_z
        ),
    )


# ============================================================
# CLOTH ON RELIEF ONLY
# ============================================================


def _build_relief_cloth(
    relief: np.ndarray,
    pcb_mask: np.ndarray,
    resolution: float,
    bridge_mm: float,
    smoothing_mm: float,
    slope_degrees: float,
) -> np.ndarray:
    """
    Cloth algoritam nikada ne vidi apsolutni PCB Z.

    On vidi samo:

        0 mm   = gola PCB površina
        >0 mm  = komponenta iznad PCB-a

    Time PCB više ne može slučajno da dobije slope.
    """

    original = np.asarray(
        relief,
        dtype=np.float32,
    )

    result = original.copy()

    #
    # Van PCB-a ništa ne postoji.
    #
    result[
        ~pcb_mask
    ] = 0.0

    # --------------------------------------------------------
    # BRIDGE
    # --------------------------------------------------------

    if bridge_mm > 0.0:

        radius_px = max(
            1,
            int(
                round(
                    bridge_mm
                    /
                    resolution
                )
            ),
        )

        size = (
            radius_px * 2
            +
            1
        )

        bridged = ndimage.grey_closing(
            result,

            size=(
                size,
                size,
            ),

            mode="constant",
            cval=0.0,
        )

        result[
            pcb_mask
        ] = bridged[
            pcb_mask
        ]

    # --------------------------------------------------------
    # SLOPE
    # --------------------------------------------------------

    slope_degrees = float(
        np.clip(
            slope_degrees,
            2.0,
            88.0,
        )
    )

    drop_per_pixel = (
        np.tan(
            np.radians(
                slope_degrees
            )
        )
        *
        resolution
    )

    result = _propagate_relief(
        relief=result,

        pcb_mask=pcb_mask,

        drop_per_pixel=drop_per_pixel,
    )

    # --------------------------------------------------------
    # SMOOTHING
    # --------------------------------------------------------

    if smoothing_mm > 0.0:

        sigma = (
            smoothing_mm
            /
            resolution
        )

        #
        # Mask-aware smoothing:
        #
        # sprečava zero prostor VAN PCB-a da vuče
        # samu PCB ivicu naniže.
        #
        values = ndimage.gaussian_filter(
            result
            *
            pcb_mask.astype(
                np.float32
            ),

            sigma=sigma,

            mode="constant",
            cval=0.0,
        )

        weights = ndimage.gaussian_filter(
            pcb_mask.astype(
                np.float32
            ),

            sigma=sigma,

            mode="constant",
            cval=0.0,
        )

        smooth = np.zeros_like(
            result
        )

        valid = (
            weights
            >
            1e-6
        )

        smooth[
            valid
        ] = (
            values[
                valid
            ]
            /
            weights[
                valid
            ]
        )

        #
        # Cloth nikada ne sme proći kroz originalni
        # component relief.
        #
        result[
            pcb_mask
        ] = np.maximum(
            original[
                pcb_mask
            ],

            smooth[
                pcb_mask
            ],
        )

    result[
        ~pcb_mask
    ] = 0.0

    result = np.maximum(
        result,
        original,
    )

    return result


def _propagate_relief(
    relief: np.ndarray,
    pcb_mask: np.ndarray,
    drop_per_pixel: float,
) -> np.ndarray:
    """
    Slope propagation SAMO unutar PCB footprint-a.

    Ne može preći preko PCB edge-a.
    """

    result = relief.copy()

    h, w = (
        result.shape
    )

    #
    # LEFT -> RIGHT
    #
    for y in range(h):

        for x in range(
            1,
            w,
        ):

            if not (
                pcb_mask[y, x]
                and
                pcb_mask[y, x - 1]
            ):
                continue

            candidate = max(
                0.0,

                result[
                    y,
                    x - 1
                ]
                -
                drop_per_pixel,
            )

            if candidate > result[
                y,
                x
            ]:

                result[
                    y,
                    x
                ] = candidate

        #
        # RIGHT -> LEFT
        #
        for x in range(
            w - 2,
            -1,
            -1,
        ):

            if not (
                pcb_mask[y, x]
                and
                pcb_mask[y, x + 1]
            ):
                continue

            candidate = max(
                0.0,

                result[
                    y,
                    x + 1
                ]
                -
                drop_per_pixel,
            )

            if candidate > result[
                y,
                x
            ]:

                result[
                    y,
                    x
                ] = candidate

    #
    # BOTTOM -> TOP
    #
    for x in range(w):

        for y in range(
            1,
            h,
        ):

            if not (
                pcb_mask[y, x]
                and
                pcb_mask[y - 1, x]
            ):
                continue

            candidate = max(
                0.0,

                result[
                    y - 1,
                    x
                ]
                -
                drop_per_pixel,
            )

            if candidate > result[
                y,
                x
            ]:

                result[
                    y,
                    x
                ] = candidate

        #
        # TOP -> BOTTOM
        #
        for y in range(
            h - 2,
            -1,
            -1,
        ):

            if not (
                pcb_mask[y, x]
                and
                pcb_mask[y + 1, x]
            ):
                continue

            candidate = max(
                0.0,

                result[
                    y + 1,
                    x
                ]
                -
                drop_per_pixel,
            )

            if candidate > result[
                y,
                x
            ]:

                result[
                    y,
                    x
                ] = candidate

    return result


# ============================================================
# MASKS
# ============================================================


def _mask_boundary(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Tačno jedan raster sloj unutrašnje PCB ivice.
    """

    eroded = ndimage.binary_erosion(
        mask,

        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),

        border_value=0,
    )

    return (
        mask
        &
        ~eroded
    )


def _make_hole_mask(
    heightmap: HeightMap,
    holes: list[DrillHole],
    extra_radius_mm: float,
) -> np.ndarray:
    """
    Rasterizuje samo eksplicitne NPTH drill rupe.
    """

    h, w = (
        heightmap.raw_top.shape
    )

    mask = np.zeros(
        (h, w),
        dtype=bool,
    )

    if not holes:
        return mask

    for hole in holes:

        radius = max(
            0.0,

            hole.diameter
            *
            0.5
            +
            extra_radius_mm,
        )

        ix0 = max(
            0,

            int(
                np.floor(
                    (
                        hole.x
                        -
                        radius
                        -
                        heightmap.min_x
                    )
                    /
                    heightmap.resolution
                )
            ),
        )

        ix1 = min(
            w - 1,

            int(
                np.ceil(
                    (
                        hole.x
                        +
                        radius
                        -
                        heightmap.min_x
                    )
                    /
                    heightmap.resolution
                )
            ),
        )

        iy0 = max(
            0,

            int(
                np.floor(
                    (
                        hole.y
                        -
                        radius
                        -
                        heightmap.min_y
                    )
                    /
                    heightmap.resolution
                )
            ),
        )

        iy1 = min(
            h - 1,

            int(
                np.ceil(
                    (
                        hole.y
                        +
                        radius
                        -
                        heightmap.min_y
                    )
                    /
                    heightmap.resolution
                )
            ),
        )

        if (
            ix0 > ix1
            or
            iy0 > iy1
        ):
            continue

        xs = (
            heightmap.min_x
            +
            np.arange(
                ix0,
                ix1 + 1,
                dtype=np.float64,
            )
            *
            heightmap.resolution
        )

        ys = (
            heightmap.min_y
            +
            np.arange(
                iy0,
                iy1 + 1,
                dtype=np.float64,
            )
            *
            heightmap.resolution
        )

        dx = (
            xs[
                None,
                :
            ]
            -
            hole.x
        )

        dy = (
            ys[
                :,
                None
            ]
            -
            hole.y
        )

        local = (
            dx * dx
            +
            dy * dy
            <=
            radius * radius
        )

        target = mask[
            iy0:
            iy1 + 1,
            ix0:
            ix1 + 1,
        ]

        target[
            local
        ] = True

    return mask


# ============================================================
# MESH
# ============================================================


def envelope_to_mesh(
    envelope: Envelope,
) -> trimesh.Trimesh:
    """
    Jednostavna i deterministička triangulacija.

    Nema:
      - boolean-a
      - Triangle-a
      - Earcut-a
      - zipper ring-a
      - mesh cleanup-a

    solid_mask određuje:

        PCB outline
        i
        NPTH rupe

    Svaka boundary ćelija automatski dobija vertikalni zid.
    """

    top = np.asarray(
        envelope.top,
        dtype=np.float32,
    )

    bottom = np.asarray(
        envelope.bottom,
        dtype=np.float32,
    )

    mask = np.asarray(
        envelope.solid_mask,
        dtype=bool,
    )

    if (
        top.shape
        !=
        bottom.shape
        or
        top.shape
        !=
        mask.shape
    ):
        raise ValueError(
            "Envelope arrays have incompatible shapes."
        )

    h, w = (
        mask.shape
    )

    #
    # Ćelija postoji samo ako sva četiri corner sample-a
    # imaju materijal.
    #
    cell_mask = (
        mask[:-1, :-1]
        &
        mask[:-1, 1:]
        &
        mask[1:, :-1]
        &
        mask[1:, 1:]
    )

    #
    # Potrebni grid vertices.
    #
    needed = np.zeros(
        (h, w),
        dtype=bool,
    )

    needed[:-1, :-1] |= cell_mask
    needed[:-1, 1:] |= cell_mask
    needed[1:, :-1] |= cell_mask
    needed[1:, 1:] |= cell_mask

    vertices = []

    top_index = np.full(
        (h, w),
        -1,
        dtype=np.int64,
    )

    bottom_index = np.full(
        (h, w),
        -1,
        dtype=np.int64,
    )

    #
    # --------------------------------------------------------
    # VERTICES
    # --------------------------------------------------------
    #

    for y in range(h):

        py = (
            envelope.min_y
            +
            y
            *
            envelope.resolution
        )

        for x in range(w):

            if not needed[
                y,
                x
            ]:
                continue

            px = (
                envelope.min_x
                +
                x
                *
                envelope.resolution
            )

            top_index[
                y,
                x
            ] = len(
                vertices
            )

            vertices.append(
                [
                    px,
                    py,
                    float(
                        top[y, x]
                    ),
                ]
            )

            bottom_index[
                y,
                x
            ] = len(
                vertices
            )

            vertices.append(
                [
                    px,
                    py,
                    float(
                        bottom[y, x]
                    ),
                ]
            )

    faces = []

    #
    # --------------------------------------------------------
    # TOP + BOTTOM
    # --------------------------------------------------------
    #

    for y in range(
        h - 1
    ):

        for x in range(
            w - 1
        ):

            if not cell_mask[
                y,
                x
            ]:
                continue

            ta = top_index[
                y,
                x
            ]

            tb = top_index[
                y,
                x + 1
            ]

            tc = top_index[
                y + 1,
                x
            ]

            td = top_index[
                y + 1,
                x + 1
            ]

            ba = bottom_index[
                y,
                x
            ]

            bb = bottom_index[
                y,
                x + 1
            ]

            bc = bottom_index[
                y + 1,
                x
            ]

            bd = bottom_index[
                y + 1,
                x + 1
            ]

            #
            # TOP
            #
            faces.append(
                [
                    ta,
                    tb,
                    td,
                ]
            )

            faces.append(
                [
                    ta,
                    td,
                    tc,
                ]
            )

            #
            # BOTTOM
            #
            faces.append(
                [
                    ba,
                    bd,
                    bb,
                ]
            )

            faces.append(
                [
                    ba,
                    bc,
                    bd,
                ]
            )

    #
    # --------------------------------------------------------
    # BOUNDARY WALLS
    # --------------------------------------------------------
    #
    # Ista logika zatvara:
    #
    #   - spoljašnji PCB edge
    #   - NPTH hole edge
    #
    # Bez ikakvog posebnog slučaja.
    #

    def add_wall(
        t1: int,
        t2: int,
        b1: int,
        b2: int,
        reverse: bool,
    ):

        if min(
            t1,
            t2,
            b1,
            b2,
        ) < 0:
            return

        if reverse:

            faces.append(
                [
                    t1,
                    t2,
                    b2,
                ]
            )

            faces.append(
                [
                    t1,
                    b2,
                    b1,
                ]
            )

        else:

            faces.append(
                [
                    t1,
                    b2,
                    t2,
                ]
            )

            faces.append(
                [
                    t1,
                    b1,
                    b2,
                ]
            )

    for y in range(
        h - 1
    ):

        for x in range(
            w - 1
        ):

            if not cell_mask[
                y,
                x
            ]:
                continue

            ta = top_index[y, x]
            tb = top_index[y, x + 1]
            tc = top_index[y + 1, x]
            td = top_index[y + 1, x + 1]

            ba = bottom_index[y, x]
            bb = bottom_index[y, x + 1]
            bc = bottom_index[y + 1, x]
            bd = bottom_index[y + 1, x + 1]

            #
            # -Y
            #
            if (
                y == 0
                or
                not cell_mask[
                    y - 1,
                    x
                ]
            ):

                add_wall(
                    ta,
                    tb,
                    ba,
                    bb,
                    reverse=True,
                )

            #
            # +Y
            #
            if (
                y == h - 2
                or
                not cell_mask[
                    y + 1,
                    x
                ]
            ):

                add_wall(
                    tc,
                    td,
                    bc,
                    bd,
                    reverse=False,
                )

            #
            # -X
            #
            if (
                x == 0
                or
                not cell_mask[
                    y,
                    x - 1
                ]
            ):

                add_wall(
                    tc,
                    ta,
                    bc,
                    ba,
                    reverse=False,
                )

            #
            # +X
            #
            if (
                x == w - 2
                or
                not cell_mask[
                    y,
                    x + 1
                ]
            ):

                add_wall(
                    tb,
                    td,
                    bb,
                    bd,
                    reverse=False,
                )

    if not vertices:
        raise ValueError(
            "Generated mesh contains no vertices."
        )

    if not faces:
        raise ValueError(
            "Generated mesh contains no faces."
        )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            vertices,
            dtype=np.float32,
        ),

        faces=np.asarray(
            faces,
            dtype=np.int64,
        ),

        process=False,
    )

    mesh.remove_unreferenced_vertices()

    print(
        "[mesh] "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.faces):,} triangles"
    )

    return mesh
