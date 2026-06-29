# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime helpers for booting interactive-drive without HD-map content.

Produces a fully procedural USDZ with the same on-disk layout the scene
loader expects (synthetic trajectory + lane lines + intersection geometry +
a caller-supplied initial RGB), so the world-model runtime runs unchanged and
unaware the scene is procedural. Thin wrapper around
``scene_fixture.build_synthetic_scene_usdz`` that adds an ``--initial-rgb``
path so demos can seed a real driving photo instead of the test gradient.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import numpy as np
from omnidreams.interactive_drive.scene_fixture import build_synthetic_scene_usdz
from PIL import Image

# Internal frame-rate of the underlying scene_fixture trajectory and the
# nominal driving speed it bakes in. ``length_km`` is converted to
# ``length_frames`` via these so the runtime API stays in km (which is
# what users actually want to reason about) instead of seconds-of-clock-
# time-at-a-fixed-speed.
_SCENE_FIXTURE_FPS = 30
_SCENE_FIXTURE_SPEED_MPS = 10.0

# "Golden track" length handed to the scene loader: 20 km at 10 m/s = 60 000
# frames (~12 MB temp USDZ, ~600 ms at startup), well past any demo session.
# Exposed as a kwarg only (not on the CLI) so tests can build smaller scenes.
_DEFAULT_LENGTH_KM = 20.0


def _length_km_to_frames(length_km: float) -> int:
    """Convert a caller-friendly km value to scene_fixture frame count.

    The scene_fixture trajectory advances at a fixed
    ``_SCENE_FIXTURE_SPEED_MPS`` along the centerline at
    ``_SCENE_FIXTURE_FPS`` Hz, so the frame count is just the product of
    distance / speed * fps. Floor at 2 frames so the loader's initial-
    speed estimator (which subtracts pose[1] from pose[0]) has something
    to work with.
    """
    metres = length_km * 1000.0
    seconds = metres / _SCENE_FIXTURE_SPEED_MPS
    return max(2, int(round(seconds * _SCENE_FIXTURE_FPS)))


def build_default_synthetic_scene(
    output_path: Path,
    *,
    initial_rgb_path: Path | None = None,
    prompt: str | None = None,
    length_km: float = _DEFAULT_LENGTH_KM,
) -> Path:
    """Materialise a procedural USDZ at ``output_path``.

    Args:
        output_path: Destination ``.usdz`` file. Parent directory is
            created if missing.
        initial_rgb_path: Optional JPG / PNG to embed as
            ``first_image.png`` inside the USDZ. The scene loader resizes
            it to ``RasterConfig.resolution_wh`` at load time so any
            input shape is fine. When ``None``, the scene_fixture's
            debug colour gradient is used (correct for smoke tests, ugly
            for real demos).
        prompt: Optional text prompt embedded as ``prompt.txt``. Defaults
            to the scene_fixture's generic synthetic-test caption.
        length_km: How much road to materialise. Lane lines, road
            boundaries, periodic poles, and parked-car obstacles are
            all spec'd along the trajectory, so longer values produce
            more drivable road. Default 20 km, generous for any demo.
    """
    initial_rgb: np.ndarray | None = None
    if initial_rgb_path is not None:
        with Image.open(initial_rgb_path) as image:
            initial_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return build_synthetic_scene_usdz(
        output_path,
        initial_rgb=initial_rgb,
        prompt=prompt,
        length_frames=_length_km_to_frames(length_km),
    )


def build_synthetic_scene_to_temp(
    *,
    initial_rgb_path: Path | None = None,
    prompt: str | None = None,
    length_km: float = _DEFAULT_LENGTH_KM,
) -> Path:
    """Build a synthetic USDZ to a process-lifetime temp directory.

    Returns the absolute path to the new USDZ. The temp directory is
    registered with :mod:`atexit` so it disappears when the process
    exits, which avoids leaving stale files around when the same
    runtime spawns multiple interactive-drive sessions.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="interactive_drive_synthetic_"))
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    return build_default_synthetic_scene(
        tmp_dir / "synthetic.usdz",
        initial_rgb_path=initial_rgb_path,
        prompt=prompt,
        length_km=length_km,
    )
