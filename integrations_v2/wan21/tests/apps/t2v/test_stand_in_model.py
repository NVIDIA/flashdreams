# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Wan 2.1 application, against a stand-in model.

Only what is particular to this integration: which model it runs, and that the
model generates its clip in one block. The checkpoint itself is
``test_real_model.py``.
"""

import pytest
from t2v.testing import FakeT2VPipelineConfig
from wan21.apps.t2v import create_app
from wan21.config import T2V_APPLICATION_DEFAULTS

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The numbers are the checkpoint's, read off the runner config this
    integration already ships. The rollout length is the exception: a config
    for a model that does not roll out does not carry one."""
    app = create_app(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        T2V_APPLICATION_DEFAULTS.pixel_width,
        T2V_APPLICATION_DEFAULTS.pixel_height,
    )
    assert desc.frames_per_second_for_step == T2V_APPLICATION_DEFAULTS.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == 1


@pytest.mark.parametrize("total_blocks", [2, 60])
def test_a_rollout_is_refused_because_this_model_does_not_roll_out(
    total_blocks: int,
) -> None:
    """A second block would not continue the first, so asking is a mistake."""
    app = create_app(pipeline_config=FakeT2VPipelineConfig())

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
