# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Lingbot's thin shared Cam2V specialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tomli as tomllib
import torch
from cam2v import Cam2VApplication, Cam2VConditioning
from lingbot.apps.cam2v import adapter
from lingbot.apps.cam2v.adapter import LingbotCam2VApplication, create_app
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.impl import conditioning

pytestmark = pytest.mark.ci_cpu

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_package_registers_the_shared_cam2v_application() -> None:
    """Keep v2 entry-point ownership in the v2 integration package."""
    manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text())

    dependencies = manifest["project"]["dependencies"]
    assert "flashdreams-cam2v" in dependencies
    entry_points = manifest["project"]["entry-points"]["flashdreams.applications_v2"]
    target = "lingbot.apps.cam2v.adapter:"
    assert entry_points == {
        "cam2v-lingbot": f"{target}create_app",
        "cam2v-lingbot-world-fast": f"{target}create_app_fast",
        "cam2v-lingbot-world-fast-taehv-window15-sink3": (
            f"{target}create_app_fast_taehv_window15_sink3"
        ),
        "cam2v-lingbot-world-v2-14b-causal-fast": (
            f"{target}create_app_v2_14b_causal_fast"
        ),
        "cam2v-lingbot-world-v2-14b-causal-fast-taehv-window15-sink3": (
            f"{target}create_app_v2_14b_causal_fast_taehv_window15_sink3"
        ),
    }
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


def test_application_uses_lingbot_pipeline_config() -> None:
    """Bind the Cam2V application directly to the model pipeline config."""
    application = LingbotCam2VApplication()
    pipeline_config = PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3

    assert isinstance(application, Cam2VApplication)
    assert application.pipeline_config is not pipeline_config
    assert application.pipeline_config.enable_sync_and_profile is False
    assert application.defaults.log_model_timing is True
    assert (
        application.pipeline_config.diffusion_model == pipeline_config.diffusion_model
    )
    assert application.defaults.total_blocks == 20
    assert application.session_desc().video_width == 832
    assert application.session_desc().video_height == 464
    assert application.session_desc().frames_per_second_for_step == 16
    assert application.defaults.first_frame_dtype is torch.bfloat16
    assert application.defaults.first_frame_interpolation == "cubic"
    assert isinstance(create_app(), LingbotCam2VApplication)


@pytest.mark.parametrize(
    ("factory", "pipeline_config"),
    [
        (adapter.create_app_fast, PIPELINE_LINGBOT_WORLD_FAST),
        (
            adapter.create_app_fast_taehv_window15_sink3,
            PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
        ),
        (
            adapter.create_app_v2_14b_causal_fast,
            PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
        ),
        (
            adapter.create_app_v2_14b_causal_fast_taehv_window15_sink3,
            PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
        ),
    ],
)
def test_variant_factories_select_their_pipeline_config(
    factory: Any,
    pipeline_config: Any,
) -> None:
    """Bind each named factory to its advertised pipeline config."""
    application = factory()

    assert isinstance(application, LingbotCam2VApplication)
    assert application.pipeline_config is not pipeline_config
    assert application.pipeline_config.name == pipeline_config.name
    assert application.pipeline_config.enable_sync_and_profile is False
    assert (
        application.pipeline_config.diffusion_model == pipeline_config.diffusion_model
    )


def test_resolver_builds_conditioning_without_legacy_runtime(tmp_path: Path) -> None:
    """Resolve prompt, calibration, and scale entirely in the v2 package."""
    image_path = tmp_path / "image.jpg"
    image_path.touch()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("  move   through the room  \nignored\n")
    intrinsic_path = tmp_path / "intrinsics.npy"
    np.save(intrinsic_path, np.array([[832.0, 480.0, 416.0, 240.0]]))
    pose_path = tmp_path / "poses.npy"
    poses = np.repeat(np.eye(4)[None], 13, axis=0)
    poses[:, 0, 3] = np.arange(13)
    np.save(pose_path, poses)

    result = conditioning.resolve_lingbot_conditioning(
        {
            "prompt": "",
            "prompt_path": prompt_path,
            "image_path": image_path,
            "pose_path": pose_path,
            "intrinsic_path": intrinsic_path,
            "world_scale": None,
            "example_data": False,
            "example_idx": 0,
            "pixel_height": 240,
            "pixel_width": 416,
        }
    )

    assert isinstance(result, Cam2VConditioning)
    assert result.prompt == "move through the room"
    assert result.first_frame_path == image_path
    assert torch.equal(
        result.base_intrinsics,
        torch.tensor([[416.0, 240.0, 208.0, 120.0]]),
    )
    assert result.world_scale == pytest.approx(6.0)


def test_explicit_world_scale_does_not_require_replay_poses(tmp_path: Path) -> None:
    """Live Cam2V control only needs poses when deriving the normalizer."""
    image_path = tmp_path / "image.jpg"
    image_path.touch()
    intrinsic_path = tmp_path / "intrinsics.npy"
    np.save(intrinsic_path, np.array([832.0, 480.0, 416.0, 240.0]))

    result = conditioning.resolve_lingbot_conditioning(
        {
            "prompt": "forward",
            "image_path": image_path,
            "pose_path": None,
            "intrinsic_path": intrinsic_path,
            "world_scale": 2.5,
            "example_data": False,
            "example_idx": 0,
            "pixel_height": 480,
            "pixel_width": 832,
        }
    )

    assert result.world_scale == 2.5


def test_example_data_fills_missing_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the documented example-data command independent of legacy helpers."""
    (tmp_path / "image.jpg").touch()
    (tmp_path / "prompt.txt").write_text("example prompt\n")
    np.save(tmp_path / "intrinsics.npy", np.array([[832.0, 480.0, 416.0, 240.0]]))
    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    np.save(tmp_path / "poses.npy", poses)
    monkeypatch.setattr(conditioning, "_ensure_example_data", lambda index: tmp_path)

    result = conditioning.resolve_lingbot_conditioning(
        {
            "prompt": "",
            "prompt_path": None,
            "image_path": None,
            "pose_path": None,
            "intrinsic_path": None,
            "world_scale": None,
            "example_data": True,
            "example_idx": 0,
            "pixel_height": 480,
            "pixel_width": 832,
        }
    )

    assert result.prompt == "example prompt"
    assert result.first_frame_path == tmp_path / "image.jpg"
    assert result.world_scale == 0.0
