# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Self-Forcing application, against a stand-in model.

Only what is particular to this integration: which model it runs, and turning
compilation off. It also covers a run to a real MP4 on behalf of all five, each
being the same factory over the same shared layer. The checkpoint itself is
``test_real_model.py``.
"""

import shutil
from pathlib import Path

import pytest
from self_forcing.config import RUNNER_WAN21_T2V_1PT3B
from t2v_self_forcing import SelfForcingT2VApplication

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.testing import (
    ExpectedFrameStats,
    FakeT2VPipeline,
    FakeT2VPipelineConfig,
    check_t2v_model_impl,
)

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The numbers are the checkpoint's, read off the runner config this
    integration already ships rather than written down again."""
    app = SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        RUNNER_WAN21_T2V_1PT3B.pixel_width,
        RUNNER_WAN21_T2V_1PT3B.pixel_height,
    )
    assert desc.frames_per_second_for_step == RUNNER_WAN21_T2V_1PT3B.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == RUNNER_WAN21_T2V_1PT3B.total_blocks


def test_compilation_can_be_turned_off_for_a_run() -> None:
    """Run against the real config rather than a stand-in, since what this
    covers is the override landing where this model keeps the setting. No model
    is loaded to answer it."""
    app = SelfForcingT2VApplication()

    app.init(["--prompt", _PROMPT, "--no-compile"])

    transformer = app.pipeline_config.diffusion_model.transformer
    assert transformer.compile_network is False


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="writing an MP4 needs ffmpeg on PATH"
)
def test_a_run_writes_every_generated_frame_to_an_mp4(tmp_path: Path) -> None:
    """An integration to a file, over the stand-in, through the shared check."""
    pipeline = FakeT2VPipeline()
    steps = 3
    path = tmp_path / "clip.mp4"

    result = check_t2v_model_impl(
        SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(pipeline)),
        # The stand-in generates its own size rather than the checkpoint's, so
        # it says so here rather than asking the application.
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_step=RUNNER_WAN21_T2V_1PT3B.fps,
            video_width=pipeline.width,
            video_height=pipeline.height,
        ),
        steps=steps,
        commandline_args=["--prompt", _PROMPT, "--device", "cpu"],
        expected=ExpectedFrameStats(
            frame_count=pipeline.first_block_frames
            + (steps - 1) * pipeline.block_frames,
            mean_luminance=(64.0, 192.0),
            min_frame_difference=1.0,
        ),
        mp4_path=path,
    )

    assert result.passed, result.failures
    assert result.frames_per_step == (
        pipeline.first_block_frames,
        pipeline.block_frames,
        pipeline.block_frames,
    )
    assert result.metrics == ({"total_ms": 1.5},) * steps
    assert path.stat().st_size > 0
