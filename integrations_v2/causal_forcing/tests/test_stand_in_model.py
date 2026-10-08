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
from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

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


def test_compilation_can_be_turned_off_for_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run against the real config rather than a stand-in, since what this
    covers is the override landing where this model keeps the setting: the
    transformer and, because this preset compiles it by default, the decoder.
    Only the setup call is replaced, so answering does not load the checkpoint."""
    app = CausalForcingT2VApplication()
    monkeypatch.setattr(type(app.pipeline_config), "setup", lambda _: FakeT2VPipeline())

    app.init(["--prompt", _PROMPT, "--no-compile"])

    assert app.pipeline_config.diffusion_model.transformer.compile_network is False
    assert app.pipeline_config.decoder.use_compile is False
