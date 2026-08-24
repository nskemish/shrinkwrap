from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
from scipy import ndimage

from core.loader import load_mesh
import core.heightmap as heightmap_module
import core.envelope as envelope_module


# ============================================================
# MEMORY SAMPLER
# ============================================================


class MemorySampler:
    def __init__(
        self,
        interval: float = 0.005,
    ):
        self.interval = interval

        self.process = psutil.Process(
            os.getpid()
        )

        self.samples = []

        self.current_stage = "startup"

        self.running = False
        self.thread = None

    def rss(self) -> int:
        return self.process.memory_info().rss

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()

    def _worker(self):
        while self.running:

            self.samples.append(
                (
                    time.perf_counter(),
                    self.current_stage,
                    self.rss(),
                )
            )

            time.sleep(
                self.interval
            )

    @contextmanager
    def stage(
        self,
        name: str,
    ):
        old_stage = self.current_stage

        gc.collect()

        before = self.rss()
        start = time.perf_counter()

        self.current_stage = name

        try:
            yield
        finally:
            elapsed = (
                time.perf_counter()
                -
                start
            )

            after = self.rss()

            stage_samples = [
                rss
                for _, stage, rss
                in self.samples
                if stage == name
            ]

            peak = (
                max(stage_samples)
                if stage_samples
                else max(
                    before,
                    after,
                )
            )

            print()
            print(
                f"[MEM] {name}"
            )

            print(
                f"      before: {format_bytes(before)}"
            )

            print(
                f"      after:  {format_bytes(after)}"
            )

            print(
                f"      delta:  {format_signed(after - before)}"
            )

            print(
                f"      peak:   {format_bytes(peak)}"
            )

            print(
                f"      spike:  {format_signed(peak - before)}"
            )

            print(
                f"      time:   {elapsed:.4f} s"
            )

            self.current_stage = old_stage

    def report(self):
        if not self.samples:
            return

        print()
        print("=" * 78)
        print("PEAK RSS BY STAGE")
        print("=" * 78)

        stages = {}

        for _, stage, rss in self.samples:
            stages.setdefault(
                stage,
                []
            ).append(rss)

        rows = []

        for stage, values in stages.items():
            rows.append(
                (
                    max(values),
                    min(values),
                    stage,
                )
            )

        rows.sort(
            reverse=True
        )

        for peak, minimum, stage in rows:
            print(
                f"{stage:<36} "
                f"peak={format_bytes(peak):>10} "
                f"range={format_bytes(peak - minimum):>10}"
            )

        global_peak = max(
            rss
            for _, _, rss
            in self.samples
        )

        print()
        print(
            "GLOBAL PEAK:",
            format_bytes(global_peak),
        )


def format_bytes(
    value: int | float,
) -> str:
    value = float(value)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    for unit in units:
        if abs(value) < 1024.0:
            return f"{value:8.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


def format_signed(
    value: int | float,
) -> str:
    sign = (
        "+"
        if value >= 0
        else "-"
    )

    return (
        sign
        +
        format_bytes(
            abs(value)
        ).strip()
    )


# ============================================================
# NDIMAGE WRAPPERS
# ============================================================


def install_ndimage_profiling(
    sampler: MemorySampler,
):
    original_grey_closing = (
        ndimage.grey_closing
    )

    original_gaussian_filter = (
        ndimage.gaussian_filter
    )

    original_distance_transform = (
        ndimage.distance_transform_edt
    )

    original_binary_erosion = (
        ndimage.binary_erosion
    )

    original_binary_propagation = (
        ndimage.binary_propagation
    )

    original_label = (
        ndimage.label
    )

    def grey_closing(*args, **kwargs):
        with sampler.stage(
            "scipy.grey_closing"
        ):
            return original_grey_closing(
                *args,
                **kwargs,
            )

    def gaussian_filter(*args, **kwargs):
        with sampler.stage(
            "scipy.gaussian_filter"
        ):
            return original_gaussian_filter(
                *args,
                **kwargs,
            )

    def distance_transform_edt(
        *args,
        **kwargs,
    ):
        with sampler.stage(
            "scipy.distance_transform_edt"
        ):
            return original_distance_transform(
                *args,
                **kwargs,
            )

    def binary_erosion(*args, **kwargs):
        with sampler.stage(
            "scipy.binary_erosion"
        ):
            return original_binary_erosion(
                *args,
                **kwargs,
            )

    def binary_propagation(
        *args,
        **kwargs,
    ):
        with sampler.stage(
            "scipy.binary_propagation"
        ):
            return original_binary_propagation(
                *args,
                **kwargs,
            )

    def label(*args, **kwargs):
        with sampler.stage(
            "scipy.label"
        ):
            return original_label(
                *args,
                **kwargs,
            )

    ndimage.grey_closing = (
        grey_closing
    )

    ndimage.gaussian_filter = (
        gaussian_filter
    )

    ndimage.distance_transform_edt = (
        distance_transform_edt
    )

    ndimage.binary_erosion = (
        binary_erosion
    )

    ndimage.binary_propagation = (
        binary_propagation
    )

    ndimage.label = (
        label
    )


# ============================================================
# NATIVE WRAPPERS
# ============================================================


def install_native_profiling(
    sampler: MemorySampler,
):
    native = getattr(
        heightmap_module,
        "_native",
        None,
    )

    if native is None:
        print(
            "[WARN] Native module nije aktivan."
        )
        return

    # ========================================================
    # PROJECT MESH
    # ========================================================

    if hasattr(
        native,
        "project_mesh",
    ):
        native_project_mesh = (
            native.project_mesh
        )

        def project_mesh(
            *args,
            **kwargs,
        ):
            with sampler.stage(
                "native.project_mesh"
            ):
                return native_project_mesh(
                    *args,
                    **kwargs,
                )

        native.project_mesh = (
            project_mesh
        )

    # ========================================================
    # PROPAGATE RELIEF
    # ========================================================

    if hasattr(
        native,
        "propagate_relief",
    ):
        native_propagate_relief = (
            native.propagate_relief
        )

        def propagate_relief(
            *args,
            **kwargs,
        ):
            with sampler.stage(
                "native.propagate_relief"
            ):
                return native_propagate_relief(
                    *args,
                    **kwargs,
                )

        native.propagate_relief = (
            propagate_relief
        )

    # ========================================================
    # ENVELOPE MESHER
    # ========================================================

    if hasattr(
        native,
        "envelope_to_mesh",
    ):
        native_envelope_to_mesh = (
            native.envelope_to_mesh
        )

        def envelope_to_mesh(
            *args,
            **kwargs,
        ):
            with sampler.stage(
                "native.envelope_to_mesh"
            ):
                return native_envelope_to_mesh(
                    *args,
                    **kwargs,
                )

        native.envelope_to_mesh = (
            envelope_to_mesh
        )


# ============================================================
# ARRAY REPORT
# ============================================================


def report_array(
    name: str,
    array,
):
    if array is None:
        return

    if not isinstance(
        array,
        np.ndarray,
    ):
        return

    print(
        f"[ARRAY] {name:<24} "
        f"{str(array.shape):<18} "
        f"{str(array.dtype):<10} "
        f"{format_bytes(array.nbytes)}"
    )


def report_heightmap_arrays(
    hm,
):
    print()
    print("=" * 78)
    print("HEIGHTMAP RETAINED ARRAYS")
    print("=" * 78)

    report_array(
        "raw_top",
        hm.raw_top,
    )

    report_array(
        "raw_bottom",
        hm.raw_bottom,
    )

    report_array(
        "pcb_mask",
        hm.pcb_mask,
    )

    total = (
        hm.raw_top.nbytes
        +
        hm.raw_bottom.nbytes
        +
        hm.pcb_mask.nbytes
    )

    print()
    print(
        "HeightMap retained total:",
        format_bytes(total),
    )


def report_envelope_arrays(
    env,
):
    print()
    print("=" * 78)
    print("ENVELOPE RETAINED ARRAYS")
    print("=" * 78)

    report_array(
        "top",
        env.top,
    )

    report_array(
        "bottom",
        env.bottom,
    )

    report_array(
        "pcb_mask",
        env.pcb_mask,
    )

    total = (
        env.top.nbytes
        +
        env.bottom.nbytes
        +
        env.pcb_mask.nbytes
    )

    print()
    print(
        "Envelope retained total:",
        format_bytes(total),
    )


# ============================================================
# PIPELINE
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mesh",
        type=Path,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--bridge",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--slope",
        type=float,
        default=55.0,
    )

    parser.add_argument(
        "--clearance",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--keepout",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--no-mesh",
        action="store_true",
    )

    args = parser.parse_args()

    sampler = MemorySampler(
        interval=0.003
    )

    sampler.start()

    install_ndimage_profiling(
        sampler
    )

    install_native_profiling(
        sampler
    )

    try:

        print()
        print("=" * 78)
        print("SHRINKWRAP MEMORY PROFILE")
        print("=" * 78)

        print(
            "Input:",
            args.mesh,
        )

        print(
            "Resolution:",
            args.resolution,
            "mm",
        )

        print(
            "Initial RSS:",
            format_bytes(
                sampler.rss()
            ),
        )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        with sampler.stage(
            "01 load_mesh"
        ):
            model = load_mesh(
                args.mesh
            )

        print()
        print(
            f"Model: "
            f"{len(model.vertices):,} vertices / "
            f"{len(model.faces):,} triangles"
        )

        print(
            "Bounds:",
            np.asarray(
                model.bounds
            ),
        )

        # ----------------------------------------------------
        # HEIGHTMAP
        # ----------------------------------------------------

        with sampler.stage(
            "02 generate_heightmap TOTAL"
        ):
            hm = (
                heightmap_module
                .generate_heightmap(
                    mesh=model,
                    resolution=args.resolution,
                )
            )

        report_heightmap_arrays(
            hm
        )

        gc.collect()

        print(
            "\nRSS after heightmap + GC:",
            format_bytes(
                sampler.rss()
            ),
        )

        # Model više nije potreban za envelope.
        del model
        gc.collect()

        print(
            "RSS after deleting source mesh:",
            format_bytes(
                sampler.rss()
            ),
        )

        # ----------------------------------------------------
        # ENVELOPE
        # ----------------------------------------------------

        with sampler.stage(
            "03 build_envelope TOTAL"
        ):
            env = (
                envelope_module
                .build_envelope(
                    heightmap=hm,
                    smoothing_mm=args.smoothing,
                    clearance_mm=args.clearance,
                    bridge_mm=args.bridge,
                    slope_degrees=args.slope,
                    holes=[],
                    hole_keepout_mm=args.keepout,
                )
            )

        report_envelope_arrays(
            env
        )

        gc.collect()

        print(
            "\nRSS after envelope + GC:",
            format_bytes(
                sampler.rss()
            ),
        )

        # ----------------------------------------------------
        # TEST RETENTION
        # ----------------------------------------------------

        before_delete_hm = (
            sampler.rss()
        )

        del hm
        gc.collect()

        after_delete_hm = (
            sampler.rss()
        )

        print()
        print(
            "[RETENTION] delete HeightMap:"
        )

        print(
            "            before:",
            format_bytes(
                before_delete_hm
            ),
        )

        print(
            "            after: ",
            format_bytes(
                after_delete_hm
            ),
        )

        print(
            "            freed: ",
            format_bytes(
                max(
                    0,
                    before_delete_hm
                    -
                    after_delete_hm
                )
            ),
        )

        # ----------------------------------------------------
        # MESH
        # ----------------------------------------------------

        if not args.no_mesh:

            with sampler.stage(
                "04 envelope_to_mesh TOTAL"
            ):
                result_mesh = (
                    envelope_module
                    .envelope_to_mesh(
                        env
                    )
                )

            print()
            print(
                f"Output mesh: "
                f"{len(result_mesh.vertices):,} vertices / "
                f"{len(result_mesh.faces):,} triangles"
            )

            print(
                "Watertight check: SKIPPED "
                "(avoiding expensive Trimesh topology cache)"
            )

            gc.collect()

            print(
                "RSS after mesher + GC:",
                format_bytes(
                    sampler.rss()
                ),
            )

            before_delete_mesh = (
                sampler.rss()
            )

            del result_mesh
            gc.collect()

            after_delete_mesh = (
                sampler.rss()
            )

            print()
            print(
                "[RETENTION] delete result mesh:"
            )

            print(
                "            freed:",
                format_bytes(
                    max(
                        0,
                        before_delete_mesh
                        -
                        after_delete_mesh
                    )
                ),
            )

        # ----------------------------------------------------
        # DELETE ENVELOPE
        # ----------------------------------------------------

        before = sampler.rss()

        del env
        gc.collect()

        after = sampler.rss()

        print()
        print(
            "[RETENTION] delete Envelope:"
        )

        print(
            "            freed:",
            format_bytes(
                max(
                    0,
                    before
                    -
                    after
                )
            ),
        )

    finally:
        sampler.stop()

    sampler.report()


if __name__ == "__main__":
    main()
