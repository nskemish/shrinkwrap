from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from shrinkwrap import _native
except ImportError:
    _native = None

import trimesh
from scipy import ndimage

from shrinkwrap.model.drill import DrillHole
from shrinkwrap.model.heightmap import HeightMap


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
    # STEEP-SURFACE REGULARIZATION
    # --------------------------------------------------------
    #
    # Very steep raster-generated cloth ramps can contain
    # tiny staircase/faceting artifacts. They are not real
    # component geometry, so locally approximate planar
    # regions are regularized before meshing.
    #
    # Pixels occupied by the original component relief are
    # locked and never modified.
    #

    if (
        _native is not None
        and
        hasattr(
            _native,
            "regularize_steep_surface",
        )
        and
        slope_degrees >= 70.0
    ):
        result = np.ascontiguousarray(
            result,
            dtype=np.float32,
        )

        original_native = np.ascontiguousarray(
            original,
            dtype=np.float32,
        )

        mask_native = np.ascontiguousarray(
            pcb_mask,
            dtype=np.bool_,
        )

        regularizer_radius = int(
            np.clip(
                round(
                    0.30
                    /
                    resolution
                ),
                2,
                12,
            )
        )

        regularizer_tolerance = max(
            0.010,
            min(
                0.030,
                resolution * 0.25,
            ),
        )

        _native.regularize_steep_surface(
            result,
            original_native,
            mask_native,

            float(resolution),

            70.0,
            float(
                regularizer_tolerance
            ),

            regularizer_radius,
            2,
        )

        #
        # Hard safety constraint:
        # never pass through original geometry.
        #
        np.maximum(
            result,
            original,
            out=result,
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
    """Native C++ relief propagation with Python fallback."""


    return envelope_to_mesh_adaptive(
        envelope=envelope,
        surface_tolerance=surface_tolerance,
        max_span_cells=max_span_cells,
    )

