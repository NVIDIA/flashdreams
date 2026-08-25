# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Causal-Forcing application, against a stand-in model.

All this integration does is point the shared layer at its own runner config,
so pointing at the right one is all there is to cover here. The checkpoint
itself is ``test_real_model.py``.
"""

import pytest
from causal_forcing.config import T2V_APPLICATION_DEFAULTS
from t2v.testing import FakeT2VPipelineConfig
from causal_forcing.apps.t2v import create_app

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the test generates from."""


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
