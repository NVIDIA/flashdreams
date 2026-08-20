# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Causal-Forcing application, against a stand-in model.

What is specific to this integration is which model it runs, so that is what
these cover: that the defaults come off the runner config this package ships,
and that a run reaches a file. How a text-to-video application behaves in
general belongs to the shared layer and is covered in ``flashdreams/test_v2``,
which is why there is so little here.

The one thing a stand-in cannot show is what the checkpoint generates, which is
what ``test_real_model.py`` alongside this is for. Nothing here needs a GPU.
"""

import shutil
from pathlib import Path

import pytest
from causal_forcing.config import RUNNER_WAN21_T2V_1PT3B_CHUNKWISE
from t2v_causal_forcing import CausalForcingT2VApplication

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

_STEPS = 2
"""Blocks to generate. Two, so the steady-state block size is covered as well
as the first one."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """These numbers are the checkpoint's, and are only written down once.

    They come from the runner config this integration ships, which is the point
    of deriving the defaults from it: a caller wanting the clip the model was
    trained to generate passes no size flags at all.
    """
    app = CausalForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig())
    app.init(["--prompt", _PROMPT, "--device", "cpu"])

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.pixel_width,
        RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.pixel_height,
    )
    assert desc.frames_per_second_for_step == RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.total_blocks == RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.total_blocks


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="writing an MP4 needs ffmpeg on PATH"
)
def test_a_run_writes_every_generated_frame_to_an_mp4(tmp_path: Path) -> None:
    """The whole batch path, over the stand-in, through the shared check."""
    pipeline = FakeT2VPipeline()
    path = tmp_path / "clip.mp4"

    result = check_t2v_model_impl(
        CausalForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(pipeline)),
        # The stand-in generates its own size rather than the checkpoint's, so
        # it says so here rather than asking the application.
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_step=RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.fps,
            video_width=pipeline.width,
            video_height=pipeline.height,
        ),
        steps=_STEPS,
        commandline_args=["--prompt", _PROMPT, "--device", "cpu"],
        expected=ExpectedFrameStats(
            frame_count=pipeline.first_block_frames + pipeline.block_frames,
            mean_luminance=(16.0, 240.0),
            min_frame_difference=0.5,
        ),
        mp4_path=path,
    )

    assert result.passed, result.failures
    assert path.exists()
    # Each step continued the last rather than starting again.
    assert pipeline.generated == [0, 1]
    assert len(pipeline.caches) == 1
