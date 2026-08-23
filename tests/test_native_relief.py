import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core import _native
from core.envelope import _propagate_relief_python


def run_case(
    height: int,
    width: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    relief = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    mask[
        10:height - 10,
        15:width - 15
    ] = True

    # Nepravilne rupe/prekidi u footprint-u
    for _ in range(10):
        cy = rng.integers(
            15,
            height - 15,
        )

        cx = rng.integers(
            20,
            width - 20,
        )

        r = rng.integers(
            2,
            8,
        )

        yy, xx = np.ogrid[
            :height,
            :width
        ]

        hole = (
            (xx - cx) ** 2
            +
            (yy - cy) ** 2
            <=
            r ** 2
        )

        mask[hole] = False

    # "Komponente"
    for _ in range(50):
        y = rng.integers(
            10,
            height - 10,
        )

        x = rng.integers(
            15,
            width - 15,
        )

        if mask[y, x]:
            relief[y, x] = rng.uniform(
                0.2,
                10.0,
            )

    drop = 0.15

    start = time.perf_counter()

    py_result = _propagate_relief_python(
        relief,
        mask,
        drop,
    )

    py_time = time.perf_counter() - start

    cpp_result = relief.copy()

    start = time.perf_counter()

    _native.propagate_relief(
        cpp_result,
        np.ascontiguousarray(
            mask,
            dtype=np.bool_,
        ),
        drop,
    )

    cpp_time = time.perf_counter() - start

    error = np.max(
        np.abs(
            py_result
            -
            cpp_result
        )
    )

    print(
        f"{width}x{height}: "
        f"error={error:.9f} | "
        f"Python={py_time:.4f}s | "
        f"C++={cpp_time:.4f}s | "
        f"speedup={py_time / cpp_time:.1f}x"
    )

    assert error < 1e-6


def main():
    run_case(
        200,
        300,
        1,
    )

    run_case(
        500,
        700,
        2,
    )

    print()
    print(
        "PASS: C++ relief propagation matches Python."
    )


if __name__ == "__main__":
    main()
