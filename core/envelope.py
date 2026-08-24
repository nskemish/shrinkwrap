from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from core import _native
except ImportError:
    _native = None

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

    if _native is None:
        return _propagate_relief_python(
            relief=relief,
            pcb_mask=pcb_mask,
            drop_per_pixel=drop_per_pixel,
        )

    result = np.ascontiguousarray(
        relief,
        dtype=np.float32,
    ).copy()

    mask = np.ascontiguousarray(
        pcb_mask,
        dtype=np.bool_,
    )

    _native.propagate_relief(
        result,
        mask,
        float(drop_per_pixel),
    )

    return result


def _propagate_relief_python(
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
    holes,
    keepout_mm: float,
    feather_mm: float,
) -> np.ndarray:
    """
    RAM-efficient analytic hole keepout.

    Each hole is evaluated only inside its local bounding box
    instead of constructing a full-grid distance map.
    """

    h, w = heightmap.raw_top.shape

    result = np.ones(
        (h, w),
        dtype=np.float32,
    )

    if not holes:
        return result

    resolution = float(
        heightmap.resolution
    )

    min_x = float(
        heightmap.min_x
    )

    min_y = float(
        heightmap.min_y
    )

    feather_mm = max(
        1e-6,
        float(feather_mm),
    )

    for hole in holes:

        hx = float(hole.x)
        hy = float(hole.y)

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

        outer_radius = (
            inner_radius
            +
            feather_mm
        )

        extent = (
            outer_radius
            +
            resolution
        )

        x0 = max(
            0,
            int(
                np.floor(
                    (
                        hx
                        -
                        extent
                        -
                        min_x
                    )
                    /
                    resolution
                )
            ),
        )

        x1 = min(
            w,
            int(
                np.ceil(
                    (
                        hx
                        +
                        extent
                        -
                        min_x
                    )
                    /
                    resolution
                )
            )
            +
            1,
        )

        y0 = max(
            0,
            int(
                np.floor(
                    (
                        hy
                        -
                        extent
                        -
                        min_y
                    )
                    /
                    resolution
                )
            ),
        )

        y1 = min(
            h,
            int(
                np.ceil(
                    (
                        hy
                        +
                        extent
                        -
                        min_y
                    )
                    /
                    resolution
                )
            )
            +
            1,
        )

        if (
            x0 >= x1
            or
            y0 >= y1
        ):
            continue

        xs = (
            min_x
            +
            np.arange(
                x0,
                x1,
                dtype=np.float32,
            )
            *
            resolution
        )

        ys = (
            min_y
            +
            np.arange(
                y0,
                y1,
                dtype=np.float32,
            )
            *
            resolution
        )

        dx = (
            xs[None, :]
            -
            np.float32(hx)
        )

        dy = (
            ys[:, None]
            -
            np.float32(hy)
        )

        distance = np.sqrt(
            dx * dx
            +
            dy * dy
        )

        t = (
            distance
            -
            np.float32(
                inner_radius
            )
        )

        t /= np.float32(
            feather_mm
        )

        np.clip(
            t,
            0.0,
            1.0,
            out=t,
        )

        weight = (
            t
            *
            t
            *
            (
                3.0
                -
                2.0 * t
            )
        )

        view = result[
            y0:y1,
            x0:x1
        ]

        np.minimum(
            view,
            weight,
            out=view,
        )

    return result


def _build_mesh_phi(
    pcb_mask: np.ndarray,
    holes,
    min_x: float,
    min_y: float,
    resolution: float,
) -> np.ndarray:
    """
    RAM-efficient signed-distance field for the native mesher.

    The final field is float32. SciPy EDT still internally
    produces float64, but only one full-size EDT array is kept
    alive at a time.

    Analytic drill-hole distance fields are evaluated only in
    a local bounding box around each hole.
    """

    pcb_mask = np.asarray(
        pcb_mask,
        dtype=bool,
    )

    h, w = pcb_mask.shape

    # ========================================================
    # INSIDE DISTANCE
    # ========================================================

    distance64 = ndimage.distance_transform_edt(
        pcb_mask
    )

    distance64 *= resolution

    phi = distance64.astype(
        np.float32,
        copy=True,
    )

    del distance64

    # ========================================================
    # OUTSIDE DISTANCE
    # ========================================================

    distance64 = ndimage.distance_transform_edt(
        ~pcb_mask
    )

    distance64 *= resolution

    np.subtract(
        phi,
        distance64,
        out=phi,
        casting="unsafe",
    )

    del distance64

    # ========================================================
    # ANALYTIC HOLES — LOCAL ONLY
    # ========================================================

    if holes:

        margin = max(
            resolution * 2.5,
            0.05,
        )

        for hole in holes:

            hx = float(hole.x)
            hy = float(hole.y)

            radius = (
                float(hole.diameter)
                *
                0.5
            )

            extent = (
                radius
                +
                margin
            )

            x0 = max(
                0,
                int(
                    np.floor(
                        (
                            hx
                            -
                            extent
                            -
                            min_x
                        )
                        /
                        resolution
                    )
                ),
            )

            x1 = min(
                w,
                int(
                    np.ceil(
                        (
                            hx
                            +
                            extent
                            -
                            min_x
                        )
                        /
                        resolution
                    )
                )
                +
                1,
            )

            y0 = max(
                0,
                int(
                    np.floor(
                        (
                            hy
                            -
                            extent
                            -
                            min_y
                        )
                        /
                        resolution
                    )
                ),
            )

            y1 = min(
                h,
                int(
                    np.ceil(
                        (
                            hy
                            +
                            extent
                            -
                            min_y
                        )
                        /
                        resolution
                    )
                )
                +
                1,
            )

            if (
                x0 >= x1
                or
                y0 >= y1
            ):
                continue

            xs = (
                min_x
                +
                np.arange(
                    x0,
                    x1,
                    dtype=np.float32,
                )
                *
                resolution
            )

            ys = (
                min_y
                +
                np.arange(
                    y0,
                    y1,
                    dtype=np.float32,
                )
                *
                resolution
            )

            dx = (
                xs[None, :]
                -
                np.float32(hx)
            )

            dy = (
                ys[:, None]
                -
                np.float32(hy)
            )

            hole_phi = np.sqrt(
                dx * dx
                +
                dy * dy
            )

            hole_phi -= np.float32(
                radius
            )

            view = phi[
                y0:y1,
                x0:x1
            ]

            np.minimum(
                view,
                hole_phi,
                out=view,
            )

    return np.ascontiguousarray(
        phi,
        dtype=np.float32,
    )



def envelope_to_mesh_adaptive(
    envelope: Envelope,
    surface_tolerance: float = 0.01,
    max_span_cells: int = 128,
) -> trimesh.Trimesh:
    """
    Adaptive native C++ mesher.

    Boundary / NPTH clipping ostaje na originalnom phi=0
    sistemu. Potpuno unutrašnje oblasti se adaptivno
    pojednostavljuju prema dozvoljenoj Z grešci.
    """

    if _native is None:
        raise RuntimeError(
            "Native ShrinkWrap backend is unavailable."
        )

    top = np.ascontiguousarray(
        envelope.top,
        dtype=np.float32,
    )

    bottom = np.ascontiguousarray(
        envelope.bottom,
        dtype=np.float32,
    )

    pcb_mask = np.asarray(
        envelope.pcb_mask,
        dtype=bool,
    )

    resolution = float(
        envelope.resolution
    )

    phi = _build_mesh_phi(
        pcb_mask=pcb_mask,
        holes=envelope.holes,
        min_x=float(envelope.min_x),
        min_y=float(envelope.min_y),
        resolution=resolution,
    )

    (
        vertices,
        faces,
        adaptive_patches,
        boundary_cells,
        base_cells_saved,
    ) = _native.adaptive_envelope_to_mesh(
        top,
        bottom,
        phi,

        float(envelope.min_x),
        float(envelope.min_y),
        resolution,

        float(surface_tolerance),
        int(max_span_cells),
    )

    del phi

    # ========================================================
    # QEM SIMPLIFICATION
    # ========================================================

    (
        qem_vertices,
        qem_faces,
        qem_input_faces,
        qem_output_faces,
        collapsed_edges,
    ) = _native.simplify_qem(
        np.ascontiguousarray(
            vertices,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            faces,
            dtype=np.int64,
        ),
        0.25,
        False,
    )

    vertices = qem_vertices
    faces = qem_faces

    print(
        "[mesh/qem] "
        f"{qem_input_faces:,} -> "
        f"{qem_output_faces:,} triangles | "
        f"reduction="
        f"{(1.0 - qem_output_faces / qem_input_faces) * 100.0:.2f}% | "
        f"collapsed={collapsed_edges:,}"
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    print(
        "[mesh/adaptive] "
        f"simulation={resolution:.4f} mm | "
        f"tolerance={surface_tolerance:.4f} mm | "
        f"patches={adaptive_patches:,} | "
        f"boundary={boundary_cells:,} | "
        f"saved-base-cells={base_cells_saved:,} | "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.faces):,} triangles"
    )

    return mesh


def envelope_to_mesh(
    envelope: Envelope,
    surface_tolerance: float = 0.01,
    max_span_cells: int = 128,
) -> trimesh.Trimesh:
    """
    Default ShrinkWrap mesher.

    Uses the adaptive native backend to preserve fine surface
    detail while avoiding unnecessary triangles in flat and
    slowly-varying regions.
    """

    if _native is None:
        return _envelope_to_mesh_python(
            envelope
        )

    return envelope_to_mesh_adaptive(
        envelope=envelope,
        surface_tolerance=surface_tolerance,
        max_span_cells=max_span_cells,
    )


def envelope_to_mesh_uniform(
    envelope: Envelope,
) -> trimesh.Trimesh:
    """
    Uniform native C++ reference mesher.

    Kept primarily for testing, benchmarking and debugging.
    Normal application use should go through envelope_to_mesh(),
    which uses the adaptive native backend.
    """

    if _native is None:
        return _envelope_to_mesh_python(
            envelope
        )

    top = np.ascontiguousarray(
        envelope.top,
        dtype=np.float32,
    )

    bottom = np.ascontiguousarray(
        envelope.bottom,
        dtype=np.float32,
    )

    pcb_mask = np.asarray(
        envelope.pcb_mask,
        dtype=bool,
    )

    if (
        top.shape != bottom.shape
        or
        top.shape != pcb_mask.shape
    ):
        raise ValueError(
            "Envelope arrays have incompatible shapes."
        )

    resolution = float(
        envelope.resolution
    )

    phi = _build_mesh_phi(
        pcb_mask=pcb_mask,
        holes=envelope.holes,
        min_x=float(envelope.min_x),
        min_y=float(envelope.min_y),
        resolution=resolution,
    )

    vertices, faces, active_cells, boundary_cells = (
        _native.envelope_to_mesh(
            top,
            bottom,
            phi,
            float(envelope.min_x),
            float(envelope.min_y),
            resolution,
        )
    )

    del phi

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    mesh.remove_unreferenced_vertices()

    print(
        "[mesh/native] "
        f"resolution={resolution:.4f} mm | "
        f"cells={active_cells:,} | "
        f"boundary={boundary_cells:,} | "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.faces):,} triangles"
    )

    return mesh


def _envelope_to_mesh_python(
    envelope: Envelope,
) -> trimesh.Trimesh:
    """
    Finalni implicitni mesher.

    Koristi direktno envelope grid rezoluciju.

    PCB edge:
        signed-distance iz pcb_mask-a

    NPTH:
        analitički krugovi iz Excellon podataka

    Boundary ćelije se seku na sub-pixel phi=0 poziciji,
    pa finalna granica nije samo prost kvadratni raster.
    """

    top = np.asarray(
        envelope.top,
        dtype=np.float64,
    )

    bottom = np.asarray(
        envelope.bottom,
        dtype=np.float64,
    )

    pcb_mask = np.asarray(
        envelope.pcb_mask,
        dtype=bool,
    )

    if (
        top.shape
        !=
        bottom.shape
        or
        top.shape
        !=
        pcb_mask.shape
    ):
        raise ValueError(
            "Envelope arrays have incompatible shapes."
        )

    h, w = pcb_mask.shape

    resolution = float(
        envelope.resolution
    )

    #
    # ========================================================
    # PCB SIGNED DISTANCE
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
        resolution
    )

    outside_distance = (
        ndimage.distance_transform_edt(
            ~pcb_mask
        )
        *
        resolution
    )

    phi = (
        inside_distance
        -
        outside_distance
    ).astype(
        np.float64
    )

    #
    # ========================================================
    # EXACT NPTH
    # ========================================================
    #
    # Rupe se ne rasterizuju kao bool mask.
    #
    # Koristimo analitičko distance polje:
    #
    #     distance(center) - radius
    #
    # Positive = van rupe
    # Negative = unutar rupe
    #

    if envelope.holes:

        xs = (
            envelope.min_x
            +
            np.arange(
                w,
                dtype=np.float64,
            )
            *
            resolution
        )

        ys = (
            envelope.min_y
            +
            np.arange(
                h,
                dtype=np.float64,
            )
            *
            resolution
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
            # Final solid:
            #
            #     PCB ∩ outside-hole
            #

            phi = np.minimum(
                phi,
                hole_phi,
            )

    #
    # ========================================================
    # MESH DATA
    # ========================================================

    vertices = []
    faces = []

    vertex_cache = {}

    #
    # ========================================================
    # HEIGHT SAMPLING
    # ========================================================

    def sample_height(
        data: np.ndarray,
        gx: float,
        gy: float,
    ) -> float:

        gx = float(
            np.clip(
                gx,
                0.0,
                w - 1.0,
            )
        )

        gy = float(
            np.clip(
                gy,
                0.0,
                h - 1.0,
            )
        )

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

        x1 = min(
            x0 + 1,
            w - 1,
        )

        y1 = min(
            y0 + 1,
            h - 1,
        )

        tx = (
            gx
            -
            x0
        )

        ty = (
            gy
            -
            y0
        )

        return float(
            data[
                y0,
                x0
            ]
            *
            (1.0 - tx)
            *
            (1.0 - ty)

            +

            data[
                y0,
                x1
            ]
            *
            tx
            *
            (1.0 - ty)

            +

            data[
                y1,
                x0
            ]
            *
            (1.0 - tx)
            *
            ty

            +

            data[
                y1,
                x1
            ]
            *
            tx
            *
            ty
        )

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

        x = (
            envelope.min_x
            +
            gx
            *
            resolution
        )

        y = (
            envelope.min_y
            +
            gy
            *
            resolution
        )

        data = (
            top
            if side == 0
            else bottom
        )

        z = sample_height(
            data,
            gx,
            gy,
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

    #
    # ========================================================
    # ZERO CROSSING
    # ========================================================

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

    #
    # ========================================================
    # CELL CLIPPING
    # ========================================================

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

            current_p = points[
                i
            ]

            current_v = values[
                i
            ]

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
    # BOUNDARY EDGE CACHE
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

    #
    # ========================================================
    # GRID CELLS
    # ========================================================

    for y in range(
        h - 1
    ):

        for x in range(
            w - 1
        ):

            points = [
                (
                    float(
                        x
                    ),
                    float(
                        y
                    ),
                ),
                (
                    float(
                        x + 1
                    ),
                    float(
                        y
                    ),
                ),
                (
                    float(
                        x + 1
                    ),
                    float(
                        y + 1
                    ),
                ),
                (
                    float(
                        x
                    ),
                    float(
                        y + 1
                    ),
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
            # Cela ćelija je van solid-a.
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
                value > 0.0
                for value in values
            ):

                boundary_cells += 1

            top_indices = [
                get_vertex(
                    point[0],
                    point[1],
                    0,
                )
                for point in polygon
            ]

            bottom_indices = [
                get_vertex(
                    point[0],
                    point[1],
                    1,
                )
                for point in polygon
            ]

            #
            # Clipped cell polygon je konveksan.
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
                        top_indices[
                            i + 1
                        ],
                    ]
                )

                faces.append(
                    [
                        bottom_indices[0],
                        bottom_indices[
                            i + 1
                        ],
                        bottom_indices[i],
                    ]
                )

            #
            # Segment sa oba endpoint-a na phi=0
            # pripada finalnoj granici.
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
    # VERTICAL WALLS
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
        "[mesh] "
        f"resolution={resolution:.4f} mm | "
        f"cells={active_cells:,} | "
        f"boundary={boundary_cells:,} | "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.faces):,} triangles"
    )

    return mesh

