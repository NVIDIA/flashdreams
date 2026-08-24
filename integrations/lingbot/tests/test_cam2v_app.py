# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Lingbot's thin shared Cam2V specialization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cam2v import Cam2VApplication, Cam2VConditioning
from lingbot.cam2v import LingbotCam2VApplication, create_app
from lingbot.cam2v import app as application_module
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3

pytestmark = pytest.mark.ci_cpu


def test_lingbot_reuses_its_runner_config_for_cam2v_defaults() -> None:
    """Avoid restating the model's pipeline, geometry, rate, or rollout length."""
    pipeline_config = object()
    application = LingbotCam2VApplication(pipeline_config=pipeline_config)
    runner = RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3

    assert isinstance(application, Cam2VApplication)
    assert application.pipeline_config is pipeline_config
    assert application.defaults.total_blocks == runner.total_blocks
    assert application.session_desc().video_width == runner.pixel_width
    assert application.session_desc().video_height == runner.pixel_height
    assert application.session_desc().frames_per_second_for_step == runner.fps
    assert isinstance(create_app(), LingbotCam2VApplication)


def test_lingbot_resolver_only_adapts_assets_to_shared_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Lingbot-specific trace preprocessing at the integration boundary."""
    replay = SimpleNamespace(
        prompt="move through the room",
        first_frame_path=Path("image.jpg"),
        camera_poses_path=Path("poses.npy"),
        camera_intrinsics_path=Path("intrinsics.npy"),
        pixel_height=464,
        pixel_width=832,
        world_scale=None,
    )
    trace = SimpleNamespace(
        intrinsics=torch.tensor([[500.0, 500.0, 416.0, 232.0]]),
        world_scale=2.0,
    )
    monkeypatch.setattr(
        application_module,
        "replay_inputs_from_mapping",
        lambda values: replay,
    )
    monkeypatch.setattr(
        application_module,
        "load_camera_trace",
        lambda **kwargs: trace,
    )

    conditioning = application_module._resolve_lingbot_conditioning({})

    assert isinstance(conditioning, Cam2VConditioning)
    assert conditioning.prompt == replay.prompt
    assert conditioning.first_frame_path == replay.first_frame_path
    assert torch.equal(conditioning.base_intrinsics, trace.intrinsics)
    assert conditioning.world_scale == trace.world_scale
