# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Causal-Forcing application, against a stand-in model.

This covers the model-owned pipeline and presentation defaults. The checkpoint
itself is covered by ``test_real_model.py``.
"""

import pytest
from causal_forcing.apps.t2v.adapter import (
    CAUSAL_FORCING_T2V_DEFAULTS,
    CausalForcingT2VApplication,
)

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from t2v.testing import FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the test generates from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The adapter exposes the model's native output shape and cadence."""
    app = CausalForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        CAUSAL_FORCING_T2V_DEFAULTS.pixel_width,
        CAUSAL_FORCING_T2V_DEFAULTS.pixel_height,
    )
    assert desc.frames_per_second_for_step == CAUSAL_FORCING_T2V_DEFAULTS.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == CAUSAL_FORCING_T2V_DEFAULTS.total_blocks
