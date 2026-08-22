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
    # Rigidni PCB footprint PRE oduzimanja NPTH.
    #
    pcb_mask: np.ndarray

    #
    # Raster preview finalnog solid-a.
    #
    # Finalni mesher više NE koristi njegov pixel edge
    # direktno za geometriju granice.
    #
    solid_mask: np.ndarray

    #
    # Tačne NPTH geometrije iz Excellon-a.
    #
    holes: tuple[DrillHole, ...]

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

        keepout_weight = _make_hole_keepout_weight(
            heightmap=heightmap,
            holes=holes,
            keepout_mm=hole_keepout_mm,

            #
            # Mali anti-alias / transition pojas.
            # Ne menja nominalnu keepout dimenziju,
            # samo uklanja raster stepenice.
            #
            feather_mm=max(
                0.15,
                heightmap.resolution * 1.5,
            ),
        )

        #
        # weight:
        #
        #   0.0 = rigid PCB, nema cloth-a
        #   1.0 = puni cloth
        #
        top_cloth *= keepout_weight
        bottom_cloth *= keepout_weight

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

        pcb_mask=pcb_mask.copy(),

        solid_mask=solid_mask,

        holes=tuple(holes),

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


def _make_hole_keepout_weight(
    heightmap: HeightMap,
    holes: list[DrillHole],
    keepout_mm: float,
    feather_mm: float,
) -> np.ndarray:
    """
    Analitički, sub-pixel NPTH keepout.

    Za svaku rupu:

        inner_radius =
            hole_radius + keepout_mm

    Unutar inner_radius:
        cloth = 0

    U feather pojasu:
        smoothstep 0 -> 1

    Van njega:
        cloth ostaje netaknut.

    Ovo ne rasterizuje krug kao bool masku, pa keepout
    više nema oštar pixel-step prelaz.
    """

    h, w = heightmap.raw_top.shape

    result = np.ones(
        (h, w),
        dtype=np.float32,
    )

    if not holes:
        return result

    xs = (
        heightmap.min_x
        +
        np.arange(
            w,
            dtype=np.float64,
        )
        *
        heightmap.resolution
    )

    ys = (
        heightmap.min_y
        +
        np.arange(
            h,
            dtype=np.float64,
        )
        *
        heightmap.resolution
    )

    xx = xs[None, :]
    yy = ys[:, None]

    feather_mm = max(
        1e-6,
        float(feather_mm),
    )

    for hole in holes:

        radius = (
            float(hole.diameter)
            *
            0.5
        )

        inner_radius = (
            radius
            +
            float(keepout_mm)
        )

        distance = np.sqrt(
            (
                xx
                -
                float(hole.x)
            )
            ** 2
            +
            (
                yy
                -
                float(hole.y)
            )
            ** 2
        )

        #
        # t:
        #
        # <= 0     unutra
        # 0..1     feather
        # >= 1     puni cloth
        #
        t = (
            distance
            -
            inner_radius
        ) / feather_mm

        t = np.clip(
            t,
            0.0,
            1.0,
        )

        #
        # smoothstep
        #
        weight = (
            t * t
            *
            (
                3.0
                -
                2.0 * t
            )
        )

        #
        # Ako ima više rupa, najjači keepout pobeđuje.
        #
        result = np.minimum(
            result,
            weight.astype(
                np.float32
            ),
        )

    return result


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
    boundary_quality: int = 1,
) -> trimesh.Trimesh:
    """
    High-resolution final mesher.

    VAŽNO:

        envelope.resolution

    ostaje rezolucija RAY/HEIGHT simulacije.

    Finalni mesh koristi:

        mesh_resolution =
            envelope.resolution / boundary_quality

    Na primer:

        ray resolution     = 0.20 mm
        boundary_quality   = 4
        mesh resolution    = 0.05 mm

    Height, cloth i PCB detection se NE računaju ponovo.
    Samo se finalna implicitna geometrija supersampluje.

    PCB edge:
        signed-distance iz pcb_mask

    NPTH:
        exact analitički krug iz Excellon-a
    """

    boundary_quality = max(
        1,
        int(
            boundary_quality
        ),
    )

    top_coarse = np.asarray(
        envelope.top,
        dtype=np.float64,
    )

    bottom_coarse = np.asarray(
        envelope.bottom,
        dtype=np.float64,
    )

    pcb_mask = np.asarray(
        envelope.pcb_mask,
        dtype=bool,
    )

    if (
        top_coarse.shape
        !=
        bottom_coarse.shape
        or
        top_coarse.shape
        !=
        pcb_mask.shape
    ):
        raise ValueError(
            "Envelope arrays have incompatible shapes."
        )

    coarse_h, coarse_w = (
        pcb_mask.shape
    )

    coarse_resolution = float(
        envelope.resolution
    )

    mesh_resolution = (
        coarse_resolution
        /
        boundary_quality
    )

    #
    # ========================================================
    # COARSE PCB SIGNED DISTANCE
    # ========================================================
    #
    # Positive = PCB
    # Negative = outside
    #

    inside_distance = (
        ndimage.distance_transform_edt(
            pcb_mask
        )
        *
        coarse_resolution
    )

    outside_distance = (
        ndimage.distance_transform_edt(
            ~pcb_mask
        )
        *
        coarse_resolution
    )

    pcb_phi_coarse = (
        inside_distance
        -
        outside_distance
    ).astype(
        np.float64
    )

    #
    # ========================================================
    # FINE GRID
    # ========================================================
    #
    # Čuvamo potpuno isti physical extent.
    #

    fine_w = (
        (coarse_w - 1)
        *
        boundary_quality
        +
        1
    )

    fine_h = (
        (coarse_h - 1)
        *
        boundary_quality
        +
        1
    )

    #
    # Fine-grid coordinate izražen u COARSE pixel units.
    #

    fine_gx = (
        np.arange(
            fine_w,
            dtype=np.float64,
        )
        /
        boundary_quality
    )

    fine_gy = (
        np.arange(
            fine_h,
            dtype=np.float64,
        )
        /
        boundary_quality
    )

    #
    # ========================================================
    # BILINEAR RESAMPLING
    # ========================================================

    def resample_grid(
        source: np.ndarray,
    ) -> np.ndarray:

        #
        # Prvo X interpolacija.
        #

        x0 = np.floor(
            fine_gx
        ).astype(
            np.int64
        )

        x1 = np.minimum(
            x0 + 1,
            coarse_w - 1,
        )

        tx = (
            fine_gx
            -
            x0
        )

        temp = (
            source[
                :,
                x0
            ]
            *
            (
                1.0
                -
                tx[
                    None,
                    :
                ]
            )
            +
            source[
                :,
                x1
            ]
            *
            tx[
                None,
                :
            ]
        )

        #
        # Onda Y interpolacija.
        #

        y0 = np.floor(
            fine_gy
        ).astype(
            np.int64
        )

        y1 = np.minimum(
            y0 + 1,
            coarse_h - 1,
        )

        ty = (
            fine_gy
            -
            y0
        )

        result = (
            temp[
                y0,
                :
            ]
            *
            (
                1.0
                -
                ty[
                    :,
                    None
                ]
            )
            +
            temp[
                y1,
                :
            ]
            *
            ty[
                :,
                None
            ]
        )

        return np.asarray(
            result,
            dtype=np.float64,
        )

    top = resample_grid(
        top_coarse
    )

    bottom = resample_grid(
        bottom_coarse
    )

    phi = resample_grid(
        pcb_phi_coarse
    )

    #
    # ========================================================
    # EXACT NPTH
    # ========================================================
    #
    # Ovo je bitno:
    #
    # rupa NE dolazi iz rasterizovanog npth_mask.
    #
    # Njena granica dolazi direktno iz:
    #
    #     x, y, diameter
    #
    # iz Excellon fajla.
    #

    if envelope.holes:

        xs = (
            envelope.min_x
            +
            np.arange(
                fine_w,
                dtype=np.float64,
            )
            *
            mesh_resolution
        )

        ys = (
            envelope.min_y
            +
            np.arange(
                fine_h,
                dtype=np.float64,
            )
            *
            mesh_resolution
        )

        xx = xs[
            None,
            :
        ]

        yy = ys[
            :,
            None
        ]

        for hole in envelope.holes:

            radius = (
                float(
                    hole.diameter
                )
                *
                0.5
            )

            #
            # Positive van rupe.
            # Negative unutar rupe.
            #

            hole_phi = (
                np.sqrt(
                    (
                        xx
                        -
                        float(
                            hole.x
                        )
                    )
                    ** 2
                    +
                    (
                        yy
                        -
                        float(
                            hole.y
                        )
                    )
                    ** 2
                )
                -
                radius
            )

            #
            # final solid =
            #
            # PCB ∩ outside-hole
            #

            phi = np.minimum(
                phi,
                hole_phi,
            )

    #
    # ========================================================
    # MARCHING / SUBPIXEL HELPERS
    # ========================================================

    vertices = []
    faces = []

    vertex_cache = {}

    def get_vertex(
        gx: float,
        gy: float,
        side: int,
    ) -> int:

        key = (
            side,
            round(
                gx,
                9,
            ),
            round(
                gy,
                9,
            ),
        )

        if key in vertex_cache:
            return vertex_cache[
                key
            ]

        #
        # gx/gy su FINE grid coordinates.
        #

        x = (
            envelope.min_x
            +
            gx
            *
            mesh_resolution
        )

        y = (
            envelope.min_y
            +
            gy
            *
            mesh_resolution
        )

        #
        # Bilinear Z interpolation unutar FINE height grid-a.
        #

        x0 = int(
            np.floor(
                gx
            )
        )

        y0 = int(
            np.floor(
                gy
            )
        )

        x0 = max(
            0,
            min(
                fine_w - 1,
                x0,
            ),
        )

        y0 = max(
            0,
            min(
                fine_h - 1,
                y0,
            ),
        )

        x1 = min(
            x0 + 1,
            fine_w - 1,
        )

        y1 = min(
            y0 + 1,
            fine_h - 1,
        )

        tx = float(
            np.clip(
                gx - x0,
                0.0,
                1.0,
            )
        )

        ty = float(
            np.clip(
                gy - y0,
                0.0,
                1.0,
            )
        )

        data = (
            top
            if side == 0
            else bottom
        )

        z = float(
            data[y0, x0]
            *
            (1.0 - tx)
            *
            (1.0 - ty)

            +

            data[y0, x1]
            *
            tx
            *
            (1.0 - ty)

            +

            data[y1, x0]
            *
            (1.0 - tx)
            *
            ty

            +

            data[y1, x1]
            *
            tx
            *
            ty
        )

        index = len(
            vertices
        )

        vertices.append(
            [
                x,
                y,
                z,
            ]
        )

        vertex_cache[
            key
        ] = index

        return index

    def zero_cross(
        p0,
        v0,
        p1,
        v1,
    ):

        denominator = (
            v0
            -
            v1
        )

        if abs(
            denominator
        ) < 1e-12:

            t = 0.5

        else:

            t = (
                v0
                /
                denominator
            )

        t = float(
            np.clip(
                t,
                0.0,
                1.0,
            )
        )

        return (
            p0[0]
            +
            (
                p1[0]
                -
                p0[0]
            )
            *
            t,

            p0[1]
            +
            (
                p1[1]
                -
                p0[1]
            )
            *
            t,
        )

    def clip_positive(
        points,
        values,
    ):

        output_points = []
        output_values = []

        count = len(
            points
        )

        for i in range(
            count
        ):

            current_p = points[i]
            current_v = values[i]

            previous_p = points[
                i - 1
            ]

            previous_v = values[
                i - 1
            ]

            current_inside = (
                current_v
                >=
                0.0
            )

            previous_inside = (
                previous_v
                >=
                0.0
            )

            if current_inside:

                if not previous_inside:

                    output_points.append(
                        zero_cross(
                            previous_p,
                            previous_v,
                            current_p,
                            current_v,
                        )
                    )

                    output_values.append(
                        0.0
                    )

                output_points.append(
                    current_p
                )

                output_values.append(
                    current_v
                )

            elif previous_inside:

                output_points.append(
                    zero_cross(
                        previous_p,
                        previous_v,
                        current_p,
                        current_v,
                    )
                )

                output_values.append(
                    0.0
                )

        return (
            output_points,
            output_values,
        )

    #
    # ========================================================
    # MESH CELLS
    # ========================================================

    boundary_edges = {}

    def edge_key(
        a,
        b,
    ):

        pa = (
            round(
                a[0],
                9,
            ),
            round(
                a[1],
                9,
            ),
        )

        pb = (
            round(
                b[0],
                9,
            ),
            round(
                b[1],
                9,
            ),
        )

        return tuple(
            sorted(
                (
                    pa,
                    pb,
                )
            )
        )

    active_cells = 0
    boundary_cells = 0

    for y in range(
        fine_h - 1
    ):

        for x in range(
            fine_w - 1
        ):

            points = [
                (
                    float(x),
                    float(y),
                ),
                (
                    float(x + 1),
                    float(y),
                ),
                (
                    float(x + 1),
                    float(y + 1),
                ),
                (
                    float(x),
                    float(y + 1),
                ),
            ]

            values = [
                float(
                    phi[
                        y,
                        x
                    ]
                ),

                float(
                    phi[
                        y,
                        x + 1
                    ]
                ),

                float(
                    phi[
                        y + 1,
                        x + 1
                    ]
                ),

                float(
                    phi[
                        y + 1,
                        x
                    ]
                ),
            ]

            #
            # Potpuno outside.
            #

            if max(
                values
            ) < 0.0:
                continue

            polygon, polygon_values = (
                clip_positive(
                    points,
                    values,
                )
            )

            if len(
                polygon
            ) < 3:
                continue

            active_cells += 1

            if not all(
                value
                >
                0.0
                for value in values
            ):

                boundary_cells += 1

            top_indices = [
                get_vertex(
                    point[0],
                    point[1],
                    0,
                )
                for point
                in polygon
            ]

            bottom_indices = [
                get_vertex(
                    point[0],
                    point[1],
                    1,
                )
                for point
                in polygon
            ]

            #
            # Convex polygon → fan triangulation.
            #

            for i in range(
                1,
                len(
                    polygon
                )
                -
                1,
            ):

                faces.append(
                    [
                        top_indices[0],
                        top_indices[i],
                        top_indices[i + 1],
                    ]
                )

                faces.append(
                    [
                        bottom_indices[0],
                        bottom_indices[i + 1],
                        bottom_indices[i],
                    ]
                )

            #
            # phi=0 edge je boundary wall.
            #

            count = len(
                polygon
            )

            for i in range(
                count
            ):

                j = (
                    i + 1
                ) % count

                if (
                    abs(
                        polygon_values[i]
                    )
                    <=
                    1e-10
                    and
                    abs(
                        polygon_values[j]
                    )
                    <=
                    1e-10
                ):

                    a = polygon[i]
                    b = polygon[j]

                    boundary_edges[
                        edge_key(
                            a,
                            b,
                        )
                    ] = (
                        a,
                        b,
                    )

    #
    # ========================================================
    # WALLS
    # ========================================================

    for a, b in boundary_edges.values():

        ta = get_vertex(
            a[0],
            a[1],
            0,
        )

        tb = get_vertex(
            b[0],
            b[1],
            0,
        )

        ba = get_vertex(
            a[0],
            a[1],
            1,
        )

        bb = get_vertex(
            b[0],
            b[1],
            1,
        )

        faces.append(
            [
                ta,
                ba,
                bb,
            ]
        )

        faces.append(
            [
                ta,
                bb,
                tb,
            ]
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
        "[mesh] supersampled | "
        f"ray={coarse_resolution:.4f} mm | "
        f"mesh={mesh_resolution:.4f} mm | "
        f"quality={boundary_quality}x | "
        f"cells={active_cells:,} | "
        f"boundary={boundary_cells:,} | "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.faces):,} triangles"
    )

    return mesh

