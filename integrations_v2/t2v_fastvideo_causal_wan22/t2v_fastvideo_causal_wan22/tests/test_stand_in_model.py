# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the CausalWan 2.2 application, against a stand-in model.

Only what is particular to this integration: which model it runs, and that the
model denoises with two transformers. The checkpoint itself is
``test_real_model.py``.
"""

import pytest
from fastvideo_causal_wan22.config import RUNNER_WAN22_T2V_14B
from t2v_fastvideo_causal_wan22 import FastvideoCausalWan22T2VApplication

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from flashdreams.t2v_v2.testing import FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu

_PROMPT = "A cat surfing"
"""Prompt the tests generate from."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The numbers are the checkpoint's, read off the runner config this
    integration already ships rather than written down again."""
    app = FastvideoCausalWan22T2VApplication(pipeline_config=FakeT2VPipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        RUNNER_WAN22_T2V_14B.pixel_width,
        RUNNER_WAN22_T2V_14B.pixel_height,
    )
    assert desc.frames_per_second_for_step == RUNNER_WAN22_T2V_14B.fps
    assert desc.output_layout is VideoTensorLayout.tchw
    assert app.defaults.total_blocks == RUNNER_WAN22_T2V_14B.total_blocks


def test_compilation_is_turned_off_for_both_noise_level_transformers() -> None:
    """Half a compiled model is the failure this integration has to avoid.

    The real config compiles by default, which costs minutes on first use. This
    model splits denoising across two transformers, and the shared override
    reaches only one of them, so it is overridden here.
    """
    app = FastvideoCausalWan22T2VApplication()

    app.init(["--prompt", _PROMPT, "--no-compile"])

    transformer = app.pipeline_config.diffusion_model.transformer
    assert transformer.transformer_high_noise.compile_network is False
    assert transformer.transformer_low_noise.compile_network is False
