# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the check an integration's tests run against their model.

Every integration exercises the passing path of ``check_t2v_model_impl``
against its own stand-in, so what is covered here is the failing one: a check
that says nothing useful about a run that fell short is worse than no check.
"""

import pytest

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.application import T2VApplication
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults
from flashdreams.t2v_v2.testing import (
    ExpectedFrameStats,
    FakeT2VPipeline,
    FakeT2VPipelineConfig,
    check_t2v_model_impl,
)

pytestmark = pytest.mark.ci_cpu

_WIDTH = 128
"""Frame width the stand-in generates."""

_HEIGHT = 64
"""Frame height it generates."""

_FIRST_BLOCK_FRAMES = 9
"""Frames the one step under test decodes."""

_FPS = 16
"""Rate the generated frames are meant to play at."""

_PROMPT = "A cat surfing"
"""Prompt the check generates from."""


def test_the_check_reports_what_a_run_failed_to_generate() -> None:
    pipeline = FakeT2VPipeline(
        width=_WIDTH, height=_HEIGHT, first_block_frames=_FIRST_BLOCK_FRAMES
    )
    app = T2VApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=FakeT2VPipelineConfig(pipeline),
            total_blocks=1,
            pixel_width=_WIDTH,
            pixel_height=_HEIGHT,
            device="cpu",
            fps=_FPS,
            output_layout=VideoTensorLayout.tchw,
        )
    )

    result = check_t2v_model_impl(
        app,
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_step=_FPS,
            video_width=_WIDTH,
            video_height=_HEIGHT,
        ),
        steps=1,
        commandline_args=["--prompt", _PROMPT],
        # Nothing the stand-in generates meets any of these, so each one is a
        # failure the check has to name.
        expected=ExpectedFrameStats(
            frame_count=_FIRST_BLOCK_FRAMES + 1,
            mean_luminance=(250.0, 255.0),
            min_frame_difference=1000.0,
        ),
        # No path, so no MP4 is written and the check needs no ffmpeg.
    )

    assert not result.passed
    assert len(result.failures) == 3
    assert f"generated {_FIRST_BLOCK_FRAMES}" in result.failures[0]
