# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Causal-Forcing application, against a stand-in model.

All this integration does is point the shared layer at its own runner config,
so pointing at the right one is all there is to cover here. The checkpoint
itself is ``test_real_model.py``.
"""

import pytest
from causal_forcing.config import RUNNER_WAN21_T2V_1PT3B_CHUNKWISE
from t2v_causal_forcing import CausalForcingT2VApplication

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.testing import FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the test generates from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The numbers are the checkpoint's, read off the runner config this
    integration already ships rather than written down again."""
    app = CausalForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.pixel_width,
        RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.pixel_height,
    )
    assert desc.frames_per_second_for_step == RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == RUNNER_WAN21_T2V_1PT3B_CHUNKWISE.total_blocks
