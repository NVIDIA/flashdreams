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
from self_forcing.config import T2V_APPLICATION_DEFAULTS
from t2v.testing import (
    ExpectedFrameStats,
    FakeT2VPipeline,
    FakeT2VPipelineConfig,
    TransparentUIRenderer,
    check_t2v_model_impl,
)
from self_forcing.apps.t2v import create_app

from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The numbers are the checkpoint's, read off the runner config this
    integration already ships rather than written down again."""
    app = create_app(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        T2V_APPLICATION_DEFAULTS.pixel_width,
        T2V_APPLICATION_DEFAULTS.pixel_height,
    )
    assert desc.frames_per_second_for_step == T2V_APPLICATION_DEFAULTS.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == T2V_APPLICATION_DEFAULTS.total_blocks


def test_compilation_can_be_turned_off_for_a_run() -> None:
    """Run against the real config rather than a stand-in, since what this
    covers is the override landing where this model keeps the setting. No model
    is loaded to answer it."""
    app = create_app()

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
    expected_frame_count = (
        pipeline.first_block_frames + (steps - 1) * pipeline.block_frames
    )

    result = check_t2v_model_impl(
        create_app(
            pipeline_config=FakeT2VPipelineConfig(pipeline),
            ui_renderer_factory=lambda width, height: TransparentUIRenderer(
                width=width,
                height=height,
            ),
        ),
        # The stand-in generates its own size rather than the checkpoint's, so
        # it says so here rather than asking the application.
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
            frames_per_second_for_step=T2V_APPLICATION_DEFAULTS.fps,
            video_width=pipeline.width,
            video_height=pipeline.height,
        ),
        steps=steps,
        commandline_args=["--prompt", _PROMPT, "--device", "cpu"],
        expected=ExpectedFrameStats(
            frame_count=expected_frame_count,
            mean_luminance=(64.0, 192.0),
            min_frame_difference=1.0,
        ),
        mp4_path=path,
    )

    assert result.passed, result.failures
    assert result.frames_per_step == (1,) * expected_frame_count
    assert path.stat().st_size > 0
