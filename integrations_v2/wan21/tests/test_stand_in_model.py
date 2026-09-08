# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Wan 2.1 application, against a stand-in model.

Only what is particular to this integration: which model it runs, and that the
model generates its clip in one block. The checkpoint itself is
``test_real_model.py``.
"""

import pytest
from t2v.testing import FakeT2VPipelineConfig
from wan21.apps.t2v.adapter import WAN21_T2V_DEFAULTS, Wan21T2VApplication

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The adapter exposes the model's native output shape and cadence."""
    app = Wan21T2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        WAN21_T2V_DEFAULTS.pixel_width,
        WAN21_T2V_DEFAULTS.pixel_height,
    )
    assert desc.frames_per_second_for_step == WAN21_T2V_DEFAULTS.fps
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
