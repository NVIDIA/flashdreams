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

"""Smoke tests for the vendored ludus-renderer.

Tests cover:
- CUDA software rasterizer (``LudusCudaTimestampedContext``)
- High-level ``LudusRenderer`` wrapper

All tests are excluded from the default test run.  Run explicitly with::

    uv run pytest integrations/omnidreams/tests/test_ludus_renderer.py --runxfail -v
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
CLIPGT_ZIP = REPO_ROOT / "assets" / "example_data" / "omnidreams" / "clipgt.zip"


@pytest.fixture(scope="module")
def clipgt_scene_dir() -> Path:
    """Extract clipgt.zip to a temporary directory (shared across the module)."""
    assert CLIPGT_ZIP.exists(), f"clipgt.zip not found at {CLIPGT_ZIP}"
    tmpdir = tempfile.mkdtemp(prefix="ludus_test_")
    with zipfile.ZipFile(CLIPGT_ZIP, "r") as zf:
        zf.extractall(tmpdir)
    return Path(tmpdir)


# ---------------------------------------------------------------------------
# Low-level: LudusCudaTimestampedContext (CUDA software rasterizer)
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_load_clipgt_scene_can_skip_dynamic_obstacles(
    clipgt_scene_dir: Path,
) -> None:
    from ludus_renderer import PRIM_OBSTACLE, load_clipgt_scene

    with_dynamic = load_clipgt_scene(
        str(clipgt_scene_dir), device="cpu", include_dynamic_obstacles=True
    )
    without_dynamic = load_clipgt_scene(
        str(clipgt_scene_dir), device="cpu", include_dynamic_obstacles=False
    )

    with_dynamic_count = sum(
        pool.prim_type_id == PRIM_OBSTACLE
        for pool in (with_dynamic.timestamped_scene.cube_pools or [])
    )
    without_dynamic_count = sum(
        pool.prim_type_id == PRIM_OBSTACLE
        for pool in (without_dynamic.timestamped_scene.cube_pools or [])
    )
    assert with_dynamic_count > 0
    assert without_dynamic_count == 0


@pytest.mark.manual
def test_ludus_cuda_context_renders_frame(clipgt_scene_dir: Path) -> None:
    """JIT-compile the CUDA plugin, load a clipgt scene, render one frame."""
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
    width, height = 640, 360  # small resolution for speed

    # Load scene
    scene_raw = load_clipgt_scene(str(clipgt_scene_dir), device=device)
    scene = SceneAdapter(scene_raw)

    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100_000, 20_000_000)
    assert len(timestamps) > 0

    # Create CUDA context
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
# High-level: LudusRenderer wrapper used by the omnidreams pipeline
# ---------------------------------------------------------------------------


@pytest.mark.ci_cpu
@pytest.mark.parametrize(("env_value", "expected"), [("0", 0), ("4", 4)])
def test_ludus_renderer_msaa_samples_can_be_configured(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: int
) -> None:
    from omnidreams.conditioning import renderer as renderer_module

    class _FakeCtx:
        instances = []

        def __init__(self, device: torch.device) -> None:
            self.device = device
            self.msaa_samples = []
            _FakeCtx.instances.append(self)

        def set_depth_scaling(self, enabled: bool) -> None:
            self.depth_scaling = enabled

        def set_msaa_samples(self, samples: int) -> None:
            self.msaa_samples.append(samples)

        def set_max_tessellation_levels(self, **kwargs: int) -> None:
            self.max_tessellation_levels = kwargs

        def upload_cameras(self, cameras: list[object]) -> None:
            self.cameras = cameras

        def upload_scene(self, scene: object) -> int:
            self.scene = scene
            return 7

    monkeypatch.setenv("OMNIDREAMS_LUDUS_MSAA_SAMPLES", env_value)
    monkeypatch.setattr(renderer_module, "LudusCudaTimestampedContext", _FakeCtx)
    monkeypatch.setattr(
        renderer_module.LudusRenderer,
        "to_ludus_camera",
        staticmethod(lambda camera: camera),
    )

    scene_data = SimpleNamespace(
        metadata={"ludus_scene": SimpleNamespace(timestamped_scene=object())}
    )

    renderer_module.LudusRenderer(
        scene_data=scene_data,
        camera_models={"front": SimpleNamespace()},
        device=torch.device("cpu"),
    )

    assert _FakeCtx.instances[-1].msaa_samples == [expected]


@pytest.mark.ci_cpu
def test_ludus_renderer_records_render_substage_stats_with_cpu_fake_ctx() -> None:
    from omnidreams.conditioning.renderer import LudusRenderer

    class _FakeCtx:
        needs_vflip = False
        last_render_profile = {
            "ctx_render_scalar_item_host_ms_sum": 2.0,
            "ctx_render_plugin_host_ms_sum": 5.0,
            "ctx_render_cat_host_ms": 1.0,
        }

        def render(
            self,
            scene_id_batch: torch.Tensor,
            camera_id_batch: torch.Tensor,
            timestamps_batch: torch.Tensor,
            camera_type_id_batch: torch.Tensor,
            camera_poses_batch: torch.Tensor,
            *,
            resolution: tuple[int, int],
        ) -> torch.Tensor:
            del camera_id_batch, timestamps_batch, camera_type_id_batch
            assert tuple(scene_id_batch.shape) == (2,)
            assert tuple(camera_poses_batch.shape) == (2, 4, 4)
            height, width = resolution
            return torch.zeros((2, height, width, 4), dtype=torch.uint8)

    renderer = LudusRenderer.__new__(LudusRenderer)
    renderer.device = torch.device("cpu")
    renderer.scene_id = 7
    renderer.msaa_samples = 0
    renderer.ctx = _FakeCtx()
    renderer.all_camera_map = {"front": 0}
    renderer.all_cameras = [
        SimpleNamespace(image_size=torch.tensor([4, 3], dtype=torch.int64))
    ]

    output = renderer.render_all_frames_and_cameras(
        camera_names=["front"],
        camera_poses_per_camera={"front": torch.eye(4).repeat(2, 1, 1)},
        frame_timestamps_us=[1_000_000, 1_033_333],
    )

    assert output.shape == (1, 2, 3, 3, 4)
    assert {
        "renderer_scene_id_ms",
        "renderer_batch_setup_ms",
        "renderer_camera_pose_ms",
        "renderer_ctx_render_ms",
        "renderer_output_layout_ms",
        "renderer_total_ms",
        "renderer_num_frames",
        "renderer_num_cameras",
        "renderer_msaa_samples",
        "renderer_height",
        "renderer_width",
        "ctx_render_scalar_item_host_ms_sum",
        "ctx_render_plugin_host_ms_sum",
        "ctx_render_cat_host_ms",
    } <= renderer.last_render_stats.keys()


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
    from omnidreams.conditioning.renderer import (
        LudusRenderer,
        load_and_attach_ludus_scene,
    )
    from omnidreams.conditioning.world_scenario.data_loaders import load_scene
    from omnidreams.conditioning.world_scenario.ftheta import FThetaCamera
    from omnidreams.conditioning.world_scenario.settings import SETTINGS

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

    # [n_cameras=1, n_frames={1, 2, 3}, 3, H, W]
    assert output.shape == (1, n_frames, 3, target_h, target_w), f"Got {output.shape}"
    assert output.device.type == "cuda"

    renderer.cleanup()
