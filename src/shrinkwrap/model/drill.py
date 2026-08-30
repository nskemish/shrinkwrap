from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class DrillHole:
    x: float
    y: float
    diameter: float
    plated: bool
    source: str = ""


_TOOL_RE = re.compile(
    r"^T(\d+)C([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)

_SELECT_TOOL_RE = re.compile(
    r"^T(\d+)$"
)

_X_RE = re.compile(
    r"X([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)

_Y_RE = re.compile(
    r"Y([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)


def load_excellon(
    path: str | Path,
    *,
    plated: bool,
) -> list[DrillHole]:
    """
    Parser za KiCad Excellon .drl fajlove.

    Trenutno podržava ono što KiCad generiše u našim
    fajlovima:

        METRIC
        absolute coordinates
        decimal coordinates
        TnCdiameter
        Tn
        X...Y...

    Takođe podržava modalne X/Y koordinate:
    ako linija nema X ili Y, koristi prethodnu vrednost.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Drill file does not exist: {path}"
        )

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    units_scale = 1.0

    tools: dict[int, float] = {}

    current_tool: int | None = None
    current_x: float | None = None
    current_y: float | None = None

    holes: list[DrillHole] = []

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(";"):
            continue

        upper = line.upper()

        if upper == "METRIC":
            units_scale = 1.0
            continue

        if upper in (
            "INCH",
            "M72",
        ):
            units_scale = 25.4
            continue

        tool_definition = _TOOL_RE.match(
            upper
        )

        if tool_definition:

            tool_number = int(
                tool_definition.group(1)
            )

            diameter = float(
                tool_definition.group(2)
            ) * units_scale

            tools[
                tool_number
            ] = diameter

            continue

        tool_selection = (
            _SELECT_TOOL_RE.match(
                upper
            )
        )

        if tool_selection:

            tool_number = int(
                tool_selection.group(1)
            )

            #
            # T0 može da znači cancel.
            #
            if tool_number == 0:
                current_tool = None
            else:
                current_tool = (
                    tool_number
                )

            continue

        x_match = _X_RE.search(
            upper
        )

        y_match = _Y_RE.search(
            upper
        )

        #
        # Nema koordinata.
        #
        if (
            x_match is None
            and
            y_match is None
        ):
            continue

        if x_match is not None:
            current_x = (
                float(
                    x_match.group(1)
                )
                *
                units_scale
            )

        if y_match is not None:
            current_y = (
                float(
                    y_match.group(1)
                )
                *
                units_scale
            )

        if (
            current_x is None
            or
            current_y is None
            or
            current_tool is None
        ):
            continue

        diameter = tools.get(
            current_tool
        )

        if diameter is None:
            continue

        holes.append(
            DrillHole(
                x=float(current_x),
                y=float(current_y),
                diameter=float(diameter),
                plated=plated,
                source=path.name,
            )
        )

    return deduplicate_holes(
        holes
    )


def deduplicate_holes(
    holes: list[DrillHole],
    tolerance_mm: float = 0.01,
) -> list[DrillHole]:
    """
    Ako na praktično istom XY mestu postoji više drill
    operacija, ostavljamo najveći prečnik.

    Ovo je bitno za tvoj NPTH:
    na X78.38 Y-79.07 postoje i 3.0 i 3.5 mm.
    """

    result: list[DrillHole] = []

    for hole in holes:

        found_index = None

        for i, existing in enumerate(
            result
        ):

            dx = (
                hole.x
                -
                existing.x
            )

            dy = (
                hole.y
                -
                existing.y
            )

            if (
                dx * dx
                +
                dy * dy
                <=
                tolerance_mm
                *
                tolerance_mm
            ):
                found_index = i
                break

        if found_index is None:

            result.append(
                hole
            )

            continue

        existing = result[
            found_index
        ]

        if (
            hole.diameter
            >
            existing.diameter
        ):
            result[
                found_index
            ] = hole

    return result


def transform_drills(
    holes: list[DrillHole],
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> list[DrillHole]:
    """
    Transformacija drill koordinata u coordinate system
    3D modela.

    Flip se radi oko center_x / center_y.
    """

    transformed: list[DrillHole] = []

    for hole in holes:

        x = hole.x
        y = hole.y

        if flip_x:
            x = (
                2.0 * center_x
                -
                x
            )

        if flip_y:
            y = (
                2.0 * center_y
                -
                y
            )

        x += offset_x
        y += offset_y

        transformed.append(
            DrillHole(
                x=x,
                y=y,
                diameter=hole.diameter,
                plated=hole.plated,
                source=hole.source,
            )
        )

    return transformed


def drill_bounds(
    holes: list[DrillHole],
) -> tuple[
    float,
    float,
    float,
    float,
] | None:

    if not holes:
        return None

    xs = np.asarray(
        [hole.x for hole in holes],
        dtype=np.float64,
    )

    ys = np.asarray(
        [hole.y for hole in holes],
        dtype=np.float64,
    )

    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    )


def drill_center(
    holes: list[DrillHole],
) -> tuple[float, float] | None:

    bounds = drill_bounds(
        holes
    )

    if bounds is None:
        return None

    min_x, min_y, max_x, max_y = (
        bounds
    )

    return (
        (
            min_x + max_x
        ) * 0.5,
        (
            min_y + max_y
        ) * 0.5,
    )