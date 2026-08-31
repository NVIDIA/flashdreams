# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the HY-WorldPlay Cam2V binding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomli as tomllib
import torch
from cam2v import Cam2VApplication, CameraControlInput
from hy_worldplay.apps.cam2v import adapter
from hy_worldplay.apps.cam2v.adapter import (
    HyWorldPlayCam2VApplication,
    create_app,
)
from hy_worldplay.config import PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
from hy_worldplay.impl import conditioning
from hy_worldplay.impl._action import HyWorldPlayWanCtrlEncoder
from hy_worldplay.impl.pipeline import HyWorldPlayPipelineConfig

pytestmark = pytest.mark.ci_cpu

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_package_registers_cam2v_with_the_model() -> None:
    """Keep model and application entry points in one v2 integration package."""
    manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text())

    assert "flashdreams-cam2v" in manifest["project"]["dependencies"]
    assert (
        manifest["project"]["entry-points"]["flashdreams.applications_v2"][
            "cam2v-hy-worldplay"
        ]
        == "hy_worldplay.apps.cam2v.adapter:create_app"
    )
    assert "flashdreams.runner_configs" not in manifest["project"].get(
        "entry-points", {}
    )
    assert (_PACKAGE_ROOT / "config.py").is_file()
    assert not (_PACKAGE_ROOT / "apps" / "cam2v" / "config.py").exists()
    assert (_PACKAGE_ROOT / "impl").is_dir()
    assert not (_PACKAGE_ROOT / "apps" / "cam2v" / "impl").exists()
    model_impl = _PACKAGE_ROOT / "impl"
    assert not (model_impl / "runner.py").exists()
    assert not (model_impl / "launch.py").exists()
    assert not (model_impl / "demo").exists()
    assert not (model_impl / "webrtc").exists()
    assert not (model_impl / "trajectory_viz.py").exists()


def test_application_uses_hy_worldplay_pipeline_config() -> None:
    """Bind the Cam2V application directly to the model pipeline config."""
    application = HyWorldPlayCam2VApplication()

    assert isinstance(application, Cam2VApplication)
    assert isinstance(application.pipeline_config, HyWorldPlayPipelineConfig)
    assert application.pipeline_config is PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
    assert application.defaults.total_blocks == 4
    assert application.session_desc().video_width == 1280
    assert application.session_desc().video_height == 704
    assert application.session_desc().frames_per_second_for_step == 16
    assert application.defaults.generate_step is adapter.generate_hy_worldplay_step
    assert isinstance(create_app(), HyWorldPlayCam2VApplication)


def test_resolver_uses_hy_defaults_without_a_replay_pose(tmp_path: Path) -> None:
    """Resolve live input with the trained calibration and motion scale defaults."""
    image_path = tmp_path / "image.png"
    image_path.touch()

    result = conditioning.resolve_hy_worldplay_conditioning(
        {
            "prompt": "  explore   the room ",
            "image_path": image_path,
            "intrinsic_path": None,
            "world_scale": None,
            "example_data": False,
            "pixel_height": 704,
            "pixel_width": 1280,
        }
    )

    assert result.prompt == "explore the room"
    assert result.first_frame_path == image_path
    assert result.base_intrinsics.shape == (1, 4)
    assert result.world_scale == pytest.approx(2.5)


class _FakePipeline:
    def __init__(self) -> None:
        encoder = HyWorldPlayWanCtrlEncoder.__new__(HyWorldPlayWanCtrlEncoder)
        encoder.encoder = SimpleNamespace(temporal_compression_ratio=4)
        encoder._action_labels = None
        encoder._viewmats = None
        encoder._intrinsics = None
        encoder._memory_config = None
        self.encoder = encoder
        self.config = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B
        self.generate_calls: list[tuple[int, Any]] = []
        self._parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))

    def parameters(self):
        yield self._parameter

    def generate(self, autoregressive_index: int, cache: Any) -> torch.Tensor:
        self.generate_calls.append((autoregressive_index, cache))
        return torch.full((1,), autoregressive_index)


def _camera_input(frame_count: int, *, start_x: float) -> CameraControlInput:
    poses = torch.eye(4).repeat(frame_count, 1, 1)
    poses[:, 0, 3] = torch.arange(frame_count) * 0.05 + start_x
    return CameraControlInput(
        intrinsics=torch.tensor([640.0, 640.0, 640.0, 352.0]).repeat(frame_count, 1),
        poses=poses,
        world_scale=2.5,
    )


def test_generate_step_accumulates_latent_camera_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind current and historical PRoPE inputs before each model step."""
    monkeypatch.setattr(
        adapter,
        "generate_points_in_sphere",
        lambda *args, **kwargs: torch.zeros(8, 3),
    )
    pipeline = _FakePipeline()
    cache = SimpleNamespace()

    first = adapter.generate_hy_worldplay_step(
        pipeline,
        0,
        cache,
        _camera_input(13, start_x=0.0),
    )
    second = adapter.generate_hy_worldplay_step(
        pipeline,
        1,
        cache,
        _camera_input(16, start_x=0.65),
    )

    assert first.item() == 0
    assert second.item() == 1
    assert [index for index, _ in pipeline.generate_calls] == [0, 1]
    assert pipeline.encoder._action_labels.shape == (8,)
    assert pipeline.encoder._viewmats.shape == (1, 8, 4, 4)
    assert pipeline.encoder._intrinsics.shape == (1, 8, 3, 3)
    assert pipeline.encoder._memory_config is not None
    assert pipeline.encoder._viewmats[0, -1, 0, 3].item() == pytest.approx(
        -0.56,
        abs=2e-3,
    )
