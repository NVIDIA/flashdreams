# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic rendering regression test for the Ludus OpenGL raster path.

Drives the raster backend directly via `iterate_frame_chunks` (no presenter,
no Xvfb, no wall-clock) with a fixed DriverCommand, grabs a specific frame,
and compares a low-resolution downscale of the rendered image against a
committed reference PNG using the ꟻLIP perceptual metric
(https://github.com/NVlabs/flip) to catch visual regressions.

Native-resolution output is written to `tests/output/` (gitignored) for human
inspection and for CI artifact upload. On comparison failure the FLIP error
map is also written there, colour-mapped with magma.

Two parameterizations:

* ``real`` - the production USDZ (opportunistic; silently skipped when the
  asset hasn't been fetched by ``prepare.py``).
* ``synthetic`` - a USDZ built in-process by ``build_synthetic_scene_usdz``;
  relied upon for CI coverage and never silently skipped beyond the GPU gate.

Refreshing a reference PNG (after an intentional rendering change, or to
bootstrap a new parameterization):

    INTERACTIVE_DRIVE_UPDATE_REFERENCES=1 pytest tests/test_raster_reference_image.py

In that mode the test renders as usual, skips the comparison, overwrites the
committed reference under ``tests/artifacts/`` with the fresh downscale, and
passes. Inspect the diff via ``git diff`` / image viewer before committing.
"""

import os
from collections.abc import Callable
from pathlib import Path

import flip_evaluator
import numpy as np
import numpy.typing as npt
import pytest
from omnidreams.interactive_drive._sample_assets import SAMPLE_SCENE
from omnidreams.interactive_drive.backends.raster import RasterRenderBackend
from omnidreams.interactive_drive.config import AppConfig
from omnidreams.interactive_drive.control import iterate_frame_chunks
from omnidreams.interactive_drive.scene_fixture import build_synthetic_scene_usdz
from omnidreams.interactive_drive.scene_loader import load_scene_bundle
from omnidreams.interactive_drive.types import DriverCommand
from PIL import Image

_TESTS_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _TESTS_DIR / "artifacts"
_OUTPUT_DIR = _TESTS_DIR / "output"

_TARGET_CHUNK_IDX = 4  # 5th chunk (0-indexed)
_TARGET_FRAME_IDX = 3
_THROTTLE = 0.5
_REFERENCE_HEIGHT = 300
# Mean FLIP error bound. FLIP is in [0, 1]; 0.0 is identical, higher is worse.
# Calibrated by instrumenting the test: real-scene and synthetic-scene both
# score 0.0000 against the committed references on the development GPU, so
# the actual render-to-render jitter floor is at or below the FP resolution
# of the metric. Injected-regression calibration (see commit message):
#   - R<->B channel swap:  mean FLIP ~= 0.145
#   - blanked output:      mean FLIP ~= 0.210
#   - ~1px shift (synth):  mean FLIP ~= 0.039
#   - 3/255 Gaussian noise: mean FLIP ~= 0.028
# A gate of 0.03 catches all real rendering regressions we've considered
# while leaving headroom above any jitter we've actually measured.
_MAX_MEAN_FLIP = 0.03

_UPDATE_REFERENCES_ENV = "INTERACTIVE_DRIVE_UPDATE_REFERENCES"


def _downscale_to_height(
    image: npt.NDArray[np.uint8], target_height: int
) -> npt.NDArray[np.uint8]:
    height, width = image.shape[:2]
    target_width = max(1, round(width * target_height / height))
    pil = Image.fromarray(image).resize(
        (target_width, target_height), Image.Resampling.BILINEAR
    )
    return np.asarray(pil, dtype=np.uint8)


def _to_ldr_float(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    return image.astype(np.float32) / 255.0


def _compute_flip(
    reference: npt.NDArray[np.uint8], test: npt.NDArray[np.uint8]
) -> tuple[float, npt.NDArray[np.float32]]:
    """Return (mean_flip, magma-coloured error-map as float32 [0, 1] RGB)."""
    ref_f = _to_ldr_float(reference)
    test_f = _to_ldr_float(test)
    error_map, mean_flip, _ = flip_evaluator.evaluate(
        ref_f,
        test_f,
        "LDR",
        inputsRGB=True,
        applyMagma=True,
        computeMeanError=True,
    )
    return float(mean_flip), error_map


def _resolve_real_scene(tmp_path: Path) -> Path:
    del tmp_path
    return SAMPLE_SCENE


def _resolve_synthetic_scene(tmp_path: Path) -> Path:
    return build_synthetic_scene_usdz(tmp_path / "synthetic_scene.usdz")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("scene_factory", "native_png_name", "reference_png_name"),
    [
        pytest.param(
            _resolve_real_scene,
            "raster_native_real_chunk5_frame3.png",
            "raster_reference_chunk5_frame3_300p.png",
            id="real",
            # Opportunistic: the production USDZ is only present on workstations
            # that have run ``prepare.py``. The ``synthetic`` param below runs
            # the same pipeline on an in-process scene every time.
            marks=pytest.mark.skipif(
                not SAMPLE_SCENE.exists(), reason="sample scene is not available"
            ),
        ),
        pytest.param(
            _resolve_synthetic_scene,
            "raster_native_synthetic_chunk5_frame3.png",
            "raster_reference_synthetic_chunk5_frame3_300p.png",
            id="synthetic",
        ),
    ],
)
def test_raster_reference_image(
    tmp_path: Path,
    scene_factory: Callable[[Path], Path],
    native_png_name: str,
    reference_png_name: str,
) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = scene_factory(tmp_path)
    reference_png = _ARTIFACTS_DIR / reference_png_name

    config = AppConfig(scene_path=scene_path, backend="raster")
    scene = load_scene_bundle(
        scene_path=config.scene_path,
        camera_name=config.camera_name,
        variant=config.variant,
        prompt_override=config.prompt_override,
        raster=config.raster,
    )
    backend = RasterRenderBackend(chunk=config.chunk, raster=config.raster)
    try:
        backend.warmup(scene)
        command = DriverCommand(throttle=_THROTTLE)
        chunks = []
        for chunk in iterate_frame_chunks(
            scene=scene,
            backend=backend,
            chunk_config=config.chunk,
            vehicle_config=config.vehicle,
            command_source=lambda: command,
        ):
            chunks.append(chunk)
            if len(chunks) > _TARGET_CHUNK_IDX:
                break

        frame = chunks[_TARGET_CHUNK_IDX].frames[_TARGET_FRAME_IDX]
        rendered = np.asarray(frame.rgb_host_uint8, dtype=np.uint8)

        native_png = _OUTPUT_DIR / native_png_name
        Image.fromarray(rendered).save(native_png)
        print(f"[raster-reference] native-resolution frame written to {native_png}")

        downscaled = _downscale_to_height(rendered, _REFERENCE_HEIGHT)

        if os.environ.get(_UPDATE_REFERENCES_ENV):
            reference_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(downscaled).save(reference_png)
            print(
                f"[raster-reference] {_UPDATE_REFERENCES_ENV}=1: wrote {reference_png}"
            )
            return

        if not reference_png.exists():
            raise AssertionError(
                f"Missing committed reference {reference_png}. Inspect "
                f"{native_png} and rerun with {_UPDATE_REFERENCES_ENV}=1 to "
                f"bootstrap the reference."
            )
        reference_pil = Image.open(reference_png)
        reference = np.asarray(reference_pil, dtype=np.uint8)

        if reference.shape != downscaled.shape:
            raise AssertionError(
                f"Reference shape {reference.shape} does not match downscaled "
                f"shape {downscaled.shape}. Did the render resolution change? "
                f"Inspect {native_png} and rerun with {_UPDATE_REFERENCES_ENV}=1 "
                f"to regenerate the committed reference."
            )

        mean_flip, error_map = _compute_flip(reference=reference, test=downscaled)
        print(
            f"[raster-reference] mean FLIP = {mean_flip:.4f} (gate <= {_MAX_MEAN_FLIP:.4f})"
        )

        if mean_flip > _MAX_MEAN_FLIP:
            error_map_png = _OUTPUT_DIR / f"flip_error_{reference_png_name}"
            Image.fromarray((error_map * 255.0 + 0.5).astype(np.uint8)).save(
                error_map_png
            )
            downscaled_png = _OUTPUT_DIR / reference_png_name
            Image.fromarray(downscaled).save(downscaled_png)
            raise AssertionError(
                f"Rendered frame diverged from reference: mean FLIP "
                f"{mean_flip:.4f} > {_MAX_MEAN_FLIP:.4f}. "
                f"Inspect {native_png}, {downscaled_png}, and error map "
                f"{error_map_png}. If the change is intentional, rerun with "
                f"{_UPDATE_REFERENCES_ENV}=1 to overwrite the committed reference."
            )
    finally:
        backend.close()
