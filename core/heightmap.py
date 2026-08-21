from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage


# ============================================================
# DATA
# ============================================================


@dataclass
class HeightMap:
    #
    # First-hit projection.
    #
    raw_top: np.ndarray
    raw_bottom: np.ndarray

    #
    # True tamo gde vertikalni XY zrak pogađa model.
    #
    hit_mask: np.ndarray

    #
    # Detektovan rigidni PCB footprint.
    #
    pcb_mask: np.ndarray

    #
    # Grid.
    #
    min_x: float
    min_y: float
    resolution: float

    #
    # Detektovana PCB geometrija.
    #
    pcb_top_z: float
    pcb_bottom_z: float
    pcb_thickness: float

    #
    # Originalni model bounds, bez analysis padding-a.
    #
    model_min_x: float
    model_min_y: float
    model_max_x: float
    model_max_y: float

    @property
    def width(self) -> int:
        return self.raw_top.shape[1]

    @property
    def height(self) -> int:
        return self.raw_top.shape[0]

    @property
    def max_x(self) -> float:
        return (
            self.min_x
            +
            (self.width - 1)
            *
            self.resolution
        )

    @property
    def max_y(self) -> float:
        return (
            self.min_y
            +
            (self.height - 1)
            *
            self.resolution
        )


# ============================================================
# PUBLIC
# ============================================================


def generate_heightmap(
    mesh: trimesh.Trimesh,
    resolution: float,
    analysis_padding_mm: float = 1.0,
) -> HeightMap:
    """
    Projektuje assembly kao ortografske zrake:

        TOP:
            prvi hit gledano odozgo

        BOTTOM:
            prvi hit gledano odozdo

    Zatim iz distribucije:

        raw_top - raw_bottom

    pronalazi dominantnu PCB debljinu.

    PCB footprint se ne dobija iz bounding box-a.

    PCB kandidat je XY kolona koja obuhvata obe
    detektovane PCB ravni:

        raw_top    >= pcb_top_z
        raw_bottom <= pcb_bottom_z

    Zatvorene rupe u tom footprint-u se popunjavaju.
    Prazan prostor povezan sa spoljnim svetom ostaje prazan.

    NPTH rupe se NAMERNO još ne primenjuju ovde.
    To radi finalni mesh generator.
    """

    if resolution <= 0.0:
        raise ValueError(
            "Resolution must be > 0."
        )

    if analysis_padding_mm < 0.0:
        raise ValueError(
            "analysis_padding_mm must be >= 0."
        )

    mesh = _ensure_mesh(
        mesh
    )

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )

    model_min_x = float(
        bounds[0, 0]
    )

    model_min_y = float(
        bounds[0, 1]
    )

    model_max_x = float(
        bounds[1, 0]
    )

    model_max_y = float(
        bounds[1, 1]
    )

    #
    # Padding postoji samo da bi topološka analiza
    # sigurno imala "outside" region oko PCB-a.
    #
    min_x = (
        model_min_x
        -
        analysis_padding_mm
    )

    min_y = (
        model_min_y
        -
        analysis_padding_mm
    )

    max_x = (
        model_max_x
        +
        analysis_padding_mm
    )

    max_y = (
        model_max_y
        +
        analysis_padding_mm
    )

    width = max(
        2,
        int(
            np.ceil(
                (
                    max_x
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

    height = max(
        2,
        int(
            np.ceil(
                (
                    max_y
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

    #
    # -inf znači da TOP ray još nije pogodio ništa.
    # +inf znači da BOTTOM ray još nije pogodio ništa.
    #
    raw_top = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32,
    )

    raw_bottom = np.full(
        (height, width),
        np.inf,
        dtype=np.float32,
    )

    _project_mesh(
        mesh=mesh,
        top=raw_top,
        bottom=raw_bottom,
        min_x=min_x,
        min_y=min_y,
        resolution=resolution,
    )

    top_hit = np.isfinite(
        raw_top
    )

    bottom_hit = np.isfinite(
        raw_bottom
    )

    hit_mask = (
        top_hit
        &
        bottom_hit
    )

    if not np.any(
        hit_mask
    ):
        raise ValueError(
            "Projection did not hit the model."
        )

    #
    # ========================================================
    # DETECT PCB THICKNESS
    # ========================================================
    #

    thickness_map = np.full(
        raw_top.shape,
        np.nan,
        dtype=np.float32,
    )

    thickness_map[
        hit_mask
    ] = (
        raw_top[
            hit_mask
        ]
        -
        raw_bottom[
            hit_mask
        ]
    )

    (
        pcb_thickness,
        thickness_tolerance,
    ) = _detect_pcb_thickness(
        thickness_map=thickness_map,
        valid_mask=hit_mask,
        resolution=resolution,
    )

    #
    # Bare PCB samples imaju približno dominantnu
    # detektovanu debljinu.
    #
    bare_pcb_samples = (
        hit_mask
        &
        np.isfinite(
            thickness_map
        )
        &
        (
            np.abs(
                thickness_map
                -
                pcb_thickness
            )
            <=
            thickness_tolerance
        )
    )

    if np.count_nonzero(
        bare_pcb_samples
    ) < 16:
        raise ValueError(
            "Could not find enough bare PCB samples."
        )

    #
    # Median je robustan na sitan STL/raster šum.
    #
    pcb_top_z = float(
        np.median(
            raw_top[
                bare_pcb_samples
            ]
        )
    )

    pcb_bottom_z = float(
        np.median(
            raw_bottom[
                bare_pcb_samples
            ]
        )
    )

    pcb_thickness = (
        pcb_top_z
        -
        pcb_bottom_z
    )

    if pcb_thickness <= 0.0:
        raise ValueError(
            "Detected PCB thickness is invalid."
        )

    #
    # ========================================================
    # DETECT PCB FOOTPRINT
    # ========================================================
    #
    # Ovo je ključ:
    #
    # PCB postoji tamo gde XY vertikalni segment modela
    # prelazi kroz OBE PCB ravni.
    #
    # Komponenta koja viri preko edge-a obično NE obuhvata
    # i pcb_top i pcb_bottom ravni, pa ne postaje deo PCB-a.
    #
    plane_tolerance = max(
        0.05,
        resolution * 1.5,
        pcb_thickness * 0.04,
    )

    pcb_column_candidate = (
        hit_mask
        &
        (
            raw_top
            >=
            pcb_top_z
            -
            plane_tolerance
        )
        &
        (
            raw_bottom
            <=
            pcb_bottom_z
            +
            plane_tolerance
        )
    )

    #
    # Sačuvaj najveću povezanu PCB oblast.
    #
    pcb_connected = _largest_component(
        pcb_column_candidate
    )

    if not np.any(
        pcb_connected
    ):
        raise ValueError(
            "Could not detect PCB footprint."
        )

    #
    # ========================================================
    # OUTSIDE-AIR FLOOD FILL
    # ========================================================
    #
    # Ne pokušavamo da razlikujemo rupu od Edge.Cut-a
    # prema veličini.
    #
    # Jedino pitanje je:
    #
    #     "Može li vazduh iz ove praznine da stigne
    #      do ivice simulacionog prostora?"
    #
    # DA  -> outside / PCB edge / notch / otvoreni cut
    #
    # NE  -> zatvorena unutrašnja praznina
    #        koju tretiramo kao solid PCB
    #
    air = ~pcb_connected

    #
    # Seed je samo vazduh na spoljašnjem border-u grida.
    #
    outside_seed = np.zeros_like(
        air,
        dtype=bool,
    )

    outside_seed[
        0,
        :
    ] = air[
        0,
        :
    ]

    outside_seed[
        -1,
        :
    ] = air[
        -1,
        :
    ]

    outside_seed[
        :,
        0
    ] = air[
        :,
        0
    ]

    outside_seed[
        :,
        -1
    ] = air[
        :,
        -1
    ]

    #
    # Flood-fill može da propagira samo kroz AIR.
    #
    # 8-connectivity:
    # i dijagonalni prolaz se računa kao put ka spolja.
    #
    connectivity = ndimage.generate_binary_structure(
        2,
        2,
    )

    outside_air = ndimage.binary_propagation(
        outside_seed,
        structure=connectivity,
        mask=air,
    )

    #
    # Vazduh do kog flood-fill NIJE mogao da stigne
    # potpuno je zatvoren unutar PCB regiona.
    #
    enclosed_voids = (
        air
        &
        ~outside_air
    )

    #
    # Finalna rigidna PCB maska:
    #
    # potvrđen PCB
    # +
    # zatvorene unutrašnje praznine
    #
    # Sve što je povezano sa spoljnim vazduhom ostaje otvoreno.
    #
    pcb_mask = (
        pcb_connected
        |
        enclosed_voids
    )

    #
    # DEBUG MASKS
    #
    _debug_save_mask(
        "hit_mask",
        hit_mask,
    )

    _debug_save_mask(
        "pcb_candidate",
        pcb_column_candidate,
    )

    _debug_save_mask(
        "pcb_connected",
        pcb_connected,
    )

    _debug_save_mask(
        "outside_air",
        outside_air,
    )

    _debug_save_mask(
        "enclosed_voids",
        enclosed_voids,
    )


    print(
        "[heightmap] topology | "
        f"solid={np.count_nonzero(pcb_connected):,} | "
        f"enclosed={np.count_nonzero(enclosed_voids):,} | "
        f"outside={np.count_nonzero(outside_air):,}"
    )

    pcb_mask = np.asarray(
        pcb_mask,
        dtype=bool,
    )

    #
    # NEMA bounding-box clipping-a.
    #
    # Analysis padding je već deo outside_air regiona,
    # tako da flood-fill topologija sama određuje gde
    # rigidni PCB prestaje.
    #
    # Time ne sečemo validan PCB edge prema STL bounds-u.
    #

    pcb_mask = np.asarray(
        pcb_mask,
        dtype=bool,
    )

    _debug_save_mask(
        "pcb_mask",
        pcb_mask,
    )

    print(
        "[heightmap] "
        f"PCB thickness={pcb_thickness:.3f} mm | "
        f"top={pcb_top_z:.3f} | "
        f"bottom={pcb_bottom_z:.3f} | "
        f"PCB pixels={np.count_nonzero(pcb_mask):,}"
    )

    return HeightMap(
        raw_top=raw_top,
        raw_bottom=raw_bottom,

        hit_mask=hit_mask,

        pcb_mask=pcb_mask,

        min_x=float(
            min_x
        ),

        min_y=float(
            min_y
        ),

        resolution=float(
            resolution
        ),

        pcb_top_z=float(
            pcb_top_z
        ),

        pcb_bottom_z=float(
            pcb_bottom_z
        ),

        pcb_thickness=float(
            pcb_thickness
        ),

        model_min_x=model_min_x,
        model_min_y=model_min_y,
        model_max_x=model_max_x,
        model_max_y=model_max_y,
    )


# ============================================================
# PCB DETECTION
# ============================================================


def _detect_pcb_thickness(
    thickness_map: np.ndarray,
    valid_mask: np.ndarray,
    resolution: float,
) -> tuple[
    float,
    float,
]:
    """
    Pronalazi dominantni peak distribucije lokalne
    top-bottom debljine.

    PCB je tipično najveći skup XY tačaka sa veoma
    sličnom malom debljinom.

    Ne koristimo apsolutni minimum jer:
      - pin
      - tanki zid
      - triangulation noise

    mogu biti tanji od samog PCB-a.
    """

    values = np.asarray(
        thickness_map[
            valid_mask
        ],
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    values = values[
        values > 1e-4
    ]

    if len(
        values
    ) < 32:
        raise ValueError(
            "Not enough thickness samples."
        )

    #
    # Ignorišemo ekstremno debele komponente.
    #
    upper = float(
        np.quantile(
            values,
            0.70,
        )
    )

    candidates = values[
        values <= upper
    ]

    if len(
        candidates
    ) < 16:
        candidates = values

    #
    # Bin treba da bude finiji od realne PCB tolerancije,
    # ali ne previše fin u odnosu na raster.
    #
    bin_width = max(
        0.02,
        resolution * 0.20,
    )

    minimum = float(
        np.min(
            candidates
        )
    )

    maximum = float(
        np.max(
            candidates
        )
    )

    if maximum - minimum < bin_width:

        peak = float(
            np.median(
                candidates
            )
        )

    else:

        bins = max(
            8,
            int(
                np.ceil(
                    (
                        maximum
                        -
                        minimum
                    )
                    /
                    bin_width
                )
            ),
        )

        histogram, edges = np.histogram(
            candidates,
            bins=bins,
            range=(
                minimum,
                maximum,
            ),
        )

        peak_index = int(
            np.argmax(
                histogram
            )
        )

        low = edges[
            peak_index
        ]

        high = edges[
            peak_index + 1
        ]

        peak_values = candidates[
            (
                candidates
                >=
                low
            )
            &
            (
                candidates
                <=
                high
            )
        ]

        if len(
            peak_values
        ):

            peak = float(
                np.median(
                    peak_values
                )
            )

        else:

            peak = float(
                (
                    low
                    +
                    high
                )
                *
                0.5
            )

    tolerance = max(
        0.06,
        resolution * 2.0,
        peak * 0.06,
    )

    return (
        peak,
        tolerance,
    )


def _debug_save_mask(
    name: str,
    mask: np.ndarray,
):
    """
    Debug-only PNG dump binarne maske.
    """

    try:
        from PIL import Image

        image = (
            np.asarray(
                mask,
                dtype=np.uint8,
            )
            * 255
        )

        Image.fromarray(
            image
        ).save(
            f"debug_{name}.png"
        )

    except Exception as exc:
        print(
            f"[debug] could not save {name}: {exc}"
        )


def _largest_component(
    mask: np.ndarray,
) -> np.ndarray:

    labels, count = ndimage.label(
        mask
    )

    if count <= 0:

        return np.zeros_like(
            mask,
            dtype=bool,
        )

    sizes = ndimage.sum(
        mask,
        labels,
        index=np.arange(
            1,
            count + 1,
        ),
    )

    best = (
        int(
            np.argmax(
                sizes
            )
        )
        +
        1
    )

    return (
        labels
        ==
        best
    )


# ============================================================
# PROJECTION
# ============================================================


def _project_mesh(
    mesh: trimesh.Trimesh,
    top: np.ndarray,
    bottom: np.ndarray,
    min_x: float,
    min_y: float,
    resolution: float,
):
    """
    Softverski ortografski ray projection.

    Rasterizujemo svaki triangle u XY.

    Za svaki covered XY sample:

        TOP    = najveći Z
        BOTTOM = najmanji Z

    To je ekvivalent prvom hit-u vertikalnog zraka
    gledano odozgo / odozdo.
    """

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    for face in faces:

        tri = vertices[
            face
        ]

        _rasterize_triangle(
            tri=tri,
            top=top,
            bottom=bottom,
            min_x=min_x,
            min_y=min_y,
            resolution=resolution,
        )


def _rasterize_triangle(
    tri: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    min_x: float,
    min_y: float,
    resolution: float,
):
    """
    XY barycentric rasterization.
    """

    p0, p1, p2 = tri

    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    denominator = (
        (y1 - y2)
        *
        (x0 - x2)
        +
        (x2 - x1)
        *
        (y0 - y2)
    )

    #
    # Vertikalan triangle nema XY površinu.
    #
    if abs(
        denominator
    ) < 1e-12:
        return

    ix0 = max(
        0,
        int(
            np.floor(
                (
                    min(
                        x0,
                        x1,
                        x2,
                    )
                    -
                    min_x
                )
                /
                resolution
            )
        ),
    )

    ix1 = min(
        top.shape[1] - 1,
        int(
            np.ceil(
                (
                    max(
                        x0,
                        x1,
                        x2,
                    )
                    -
                    min_x
                )
                /
                resolution
            )
        ),
    )

    iy0 = max(
        0,
        int(
            np.floor(
                (
                    min(
                        y0,
                        y1,
                        y2,
                    )
                    -
                    min_y
                )
                /
                resolution
            )
        ),
    )

    iy1 = min(
        top.shape[0] - 1,
        int(
            np.ceil(
                (
                    max(
                        y0,
                        y1,
                        y2,
                    )
                    -
                    min_y
                )
                /
                resolution
            )
        ),
    )

    if (
        ix0 > ix1
        or
        iy0 > iy1
    ):
        return

    epsilon = -1e-7

    for iy in range(
        iy0,
        iy1 + 1,
    ):

        y = (
            min_y
            +
            iy
            *
            resolution
        )

        for ix in range(
            ix0,
            ix1 + 1,
        ):

            x = (
                min_x
                +
                ix
                *
                resolution
            )

            a = (
                (
                    (y1 - y2)
                    *
                    (x - x2)
                )
                +
                (
                    (x2 - x1)
                    *
                    (y - y2)
                )
            ) / denominator

            b = (
                (
                    (y2 - y0)
                    *
                    (x - x2)
                )
                +
                (
                    (x0 - x2)
                    *
                    (y - y2)
                )
            ) / denominator

            c = (
                1.0
                -
                a
                -
                b
            )

            if not (
                a >= epsilon
                and
                b >= epsilon
                and
                c >= epsilon
            ):
                continue

            z = (
                a * z0
                +
                b * z1
                +
                c * z2
            )

            if z > top[
                iy,
                ix
            ]:

                top[
                    iy,
                    ix
                ] = z

            if z < bottom[
                iy,
                ix
            ]:

                bottom[
                    iy,
                    ix
                ] = z


# ============================================================
# UTIL
# ============================================================


def _ensure_mesh(
    mesh,
) -> trimesh.Trimesh:

    if isinstance(
        mesh,
        trimesh.Trimesh,
    ):

        return mesh

    if isinstance(
        mesh,
        trimesh.Scene,
    ):

        geometries = [
            geometry
            for geometry
            in mesh.geometry.values()
            if isinstance(
                geometry,
                trimesh.Trimesh,
            )
        ]

        if not geometries:
            raise ValueError(
                "Scene contains no triangle meshes."
            )

        return trimesh.util.concatenate(
            geometries
        )

    raise TypeError(
        f"Unsupported mesh type: {type(mesh)!r}"
    )
