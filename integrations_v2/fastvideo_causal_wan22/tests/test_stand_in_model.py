# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the CausalWan 2.2 application, against a stand-in model.

Only what is particular to this integration: which model it runs, and that the
model denoises with two transformers. The checkpoint itself is
``test_real_model.py``.
"""

import pytest
from fastvideo_causal_wan22.apps.t2v.adapter import (
    FASTVIDEO_CAUSAL_WAN22_T2V_DEFAULTS,
    FastvideoCausalWan22T2VApplication,
)
from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The adapter exposes the model's native output shape and cadence."""
    app = FastvideoCausalWan22T2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        FASTVIDEO_CAUSAL_WAN22_T2V_DEFAULTS.pixel_width,
        FASTVIDEO_CAUSAL_WAN22_T2V_DEFAULTS.pixel_height,
    )
    assert desc.frames_per_second_for_step == FASTVIDEO_CAUSAL_WAN22_T2V_DEFAULTS.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == FASTVIDEO_CAUSAL_WAN22_T2V_DEFAULTS.total_blocks


def test_compilation_is_turned_off_for_both_noise_level_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half a compiled model is the failure this integration has to avoid.

    The real config compiles by default, which costs minutes on first use. This
    model splits denoising across two transformers, and the shared override
    reaches only one of them, so it is overridden here. Only setup is replaced,
    keeping the checkpoint out of this CPU test.
    """
    app = FastvideoCausalWan22T2VApplication()
    monkeypatch.setattr(type(app.pipeline_config), "setup", lambda _: FakeT2VPipeline())

    app.init(["--prompt", _PROMPT, "--no-compile"])

    transformer = app.pipeline_config.diffusion_model.transformer
    assert transformer.transformer_high_noise.compile_network is False
    assert transformer.transformer_low_noise.compile_network is False
