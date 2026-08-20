# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Wan 2.1 application, against a stand-in model.

What is specific to this integration is which model it runs and that the model
generates its clip in one block, so that is what these cover. How such an
application behaves, and how a run of one reaches a file, are covered in
``flashdreams/test_v2`` and in the Self-Forcing integration on behalf of all of
them.

The one thing a stand-in cannot show is what the checkpoint generates, which is
what ``test_real_model.py`` alongside this is for. Nothing here needs a GPU.
"""

import pytest
from t2v_wan21 import Wan21T2VApplication
from wan21.config import RUNNER_WAN21_T2V_1PT3B_480P

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.testing import FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """These numbers are the checkpoint's, and are only written down once.

    They come from the runner config this integration ships, which is the point
    of deriving the defaults from it: a caller wanting the clip the model was
    trained to generate passes no size flags at all. The rollout length is the
    exception, since a config for a model that does not roll out does not carry
    one.
    """
    app = Wan21T2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        RUNNER_WAN21_T2V_1PT3B_480P.pixel_width,
        RUNNER_WAN21_T2V_1PT3B_480P.pixel_height,
    )
    assert desc.frames_per_second_for_step == RUNNER_WAN21_T2V_1PT3B_480P.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == 1


@pytest.mark.parametrize("total_blocks", [2, 60])
def test_a_rollout_is_refused_because_this_model_does_not_roll_out(
    total_blocks: int,
) -> None:
    """A second block would not continue the first, so asking is a mistake."""
    app = Wan21T2VApplication(pipeline_config=FakeT2VPipelineConfig())

    with pytest.raises(ValueError, match="must be 1"):
        app.init(
            [
                "--prompt",
                _PROMPT,
                "--device",
                "cpu",
                "--total-blocks",
                str(total_blocks),
            ]
        )
