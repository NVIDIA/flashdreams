# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke tests for the vendored ludus-renderer backends.

Tests cover:
- EGL/OpenGL backend (``LudusTimestampedContext``) -- requires Turing+ GPU with EGL
- CUDA software rasterizer (``LudusCudaTimestampedContext``) -- any CUDA GPU
- High-level ``LudusRenderer`` wrapper (uses CUDA backend by default)

All tests are excluded from the default test run.  Run explicitly with::

    uv run pytest integrations/alpadreams/tests/test_ludus_renderer.py --runxfail -v
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
CLIPGT_ZIP = REPO_ROOT / "assets" / "example_data" / "alpadreams" / "clipgt.zip"


@pytest.fixture(scope="module")
def clipgt_scene_dir() -> Path:
    """Extract clipgt.zip to a temporary directory (shared across the module)."""
    assert CLIPGT_ZIP.exists(), f"clipgt.zip not found at {CLIPGT_ZIP}"
    tmpdir = tempfile.mkdtemp(prefix="ludus_test_")
    with zipfile.ZipFile(CLIPGT_ZIP, "r") as zf:
        zf.extractall(tmpdir)
    return Path(tmpdir)


# ---------------------------------------------------------------------------
# Low-level: LudusTimestampedContext renders without crashing
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_ludus_timestamped_context_renders_frame(clipgt_scene_dir: Path) -> None:
    """JIT-compile the EGL plugin, load a clipgt scene, render one frame."""
    from ludus_renderer import load_clipgt_scene
    from ludus_renderer.render_utils import compute_camera_poses, create_camera
    from ludus_renderer.torch import LudusTimestampedContext
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR
    from ludus_renderer.util import resample_timestamps

    device = torch.device("cuda")
    width, height = 640, 360  # small resolution for speed

    # Load scene
    scene_raw = load_clipgt_scene(str(clipgt_scene_dir), device=device)
    from ludus_renderer.render_utils import SceneAdapter

    scene = SceneAdapter(scene_raw)

    # Resample timestamps to 10 Hz, take just the first frame
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100_000, 20_000_000)
    assert len(timestamps) > 0, "No timestamps after resampling"

    # Create context + camera
    ctx = LudusTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    camera = create_camera(width, height, device, scene=scene)
    ctx.upload_cameras([camera])
    scene_id = ctx.upload_scene(scene.timestamped_scene)

    # Compute pose for first frame
    poses, _ = compute_camera_poses(scene, timestamps[:1], device)

    # Render
    images = ctx.render(
        torch.tensor([scene_id], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        timestamps[:1].to(torch.int64),
        torch.full((1,), CAMERA_TYPE_REGULAR, dtype=torch.int32, device=device),
        poses,
        resolution=(height, width),
    )

    # Validate
    assert images.shape == (1, height, width, 4), f"Unexpected shape {images.shape}"
    assert images.dtype == torch.uint8
    # Ensure the frame is not entirely black (renderer actually produced output)
    rgb = images[0, :, :, :3]
    assert rgb.any(), "Rendered frame is entirely black -- renderer may have failed"


# ---------------------------------------------------------------------------
# Low-level: LudusCudaTimestampedContext (CUDA software rasterizer)
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_ludus_cuda_context_renders_frame(clipgt_scene_dir: Path) -> None:
    """JIT-compile the CUDA-only plugin, load a clipgt scene, render one frame."""
    from ludus_renderer import load_clipgt_scene
    from ludus_renderer.render_utils import (
        SceneAdapter,
        compute_camera_poses,
        create_camera,
    )
    from ludus_renderer.torch import LudusCudaTimestampedContext
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR
    from ludus_renderer.util import resample_timestamps

    device = torch.device("cuda")
    width, height = 640, 360

    # Load scene
    scene_raw = load_clipgt_scene(str(clipgt_scene_dir), device=device)
    scene = SceneAdapter(scene_raw)

    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100_000, 20_000_000)
    assert len(timestamps) > 0

    # Create CUDA context (no EGL/GL dependency)
    ctx = LudusCudaTimestampedContext(device=device)
    assert not ctx.needs_vflip, (
        "CUDA backend renders top-down, needs_vflip should be False"
    )
    camera = create_camera(width, height, device, scene=scene)
    ctx.upload_cameras([camera])
    scene_id = ctx.upload_scene(scene.timestamped_scene)

    poses, _ = compute_camera_poses(scene, timestamps[:1], device)

    images = ctx.render(
        torch.tensor([scene_id], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        timestamps[:1].to(torch.int64),
        torch.full((1,), CAMERA_TYPE_REGULAR, dtype=torch.int32, device=device),
        poses,
        resolution=(height, width),
    )

    assert images.shape == (1, height, width, 4), f"Unexpected shape {images.shape}"
    assert images.dtype == torch.uint8
    rgb = images[0, :, :, :3]
    assert rgb.any(), (
        "Rendered frame is entirely black -- CUDA rasterizer may have failed"
    )


# ---------------------------------------------------------------------------
# High-level: LudusRenderer wrapper used by the alpadreams pipeline
# ---------------------------------------------------------------------------


@pytest.mark.manual
@pytest.mark.parametrize(
    "n_frames", [1, 2, 3], ids=["single-frame", "two-frame", "multi-frame"]
)
def test_ludus_renderer_wrapper_renders_frames(
    clipgt_scene_dir: Path, n_frames: int
) -> None:
    """Exercise the ``LudusRenderer`` wrapper that the gRPC server uses.

    Parametrized over batch sizes to cover the single-image edge case where
    the batch dimension is 1.
    """
    from alpadreams.conditioning.renderer import (
        LudusRenderer,
        load_and_attach_ludus_scene,
    )
    from alpadreams.conditioning.world_scenario.data_loaders import load_scene
    from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
    from alpadreams.conditioning.world_scenario.settings import SETTINGS

    device = torch.device("cuda")
    camera_name = "camera_front_wide_120fov"
    target_h, target_w = 360, 640  # small for speed

    # Load scene data via the ClipGT loader
    scene_data = load_scene(
        str(clipgt_scene_dir),
        camera_names=[camera_name],
        max_frames=-1,
        input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
        resize_resolution_hw=[target_h, target_w],
    )

    # Attach the ludus GPU scene
    scene_data = load_and_attach_ludus_scene(
        str(clipgt_scene_dir),
        scene_data,
        device=device,
    )

    # Get camera model
    assert camera_name in scene_data.camera_models
    camera_model = scene_data.camera_models[camera_name]
    assert isinstance(camera_model, FThetaCamera)

    # Build renderer
    renderer = LudusRenderer(
        scene_data=scene_data,
        camera_models={camera_name: camera_model},
        device=device,
    )

    # LudusRenderer expects camera-to-world transforms; it calls
    # torch.linalg.inv internally. For a smoke test identity poses are
    # fine -- the scene renders from the world origin and we just check
    # shapes/dtypes.
    camera_poses = torch.eye(4, device=device, dtype=torch.float32)
    camera_poses = camera_poses.unsqueeze(0).expand(n_frames, -1, -1).contiguous()
    timestamps_us = [int(scene_data.ego_poses[i].timestamp) for i in range(n_frames)]

    output = renderer.render_all_frames_and_cameras(
        camera_names=[camera_name],
        camera_poses_per_camera={camera_name: camera_poses},
        frame_timestamps_us=timestamps_us,
    )

    # [n_cameras=1, n_frames={1, 2}, 3, H, W]
    assert output.shape == (1, n_frames, 3, target_h, target_w), f"Got {output.shape}"
    assert output.device.type == "cuda"

    renderer.cleanup()
