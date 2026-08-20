# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real model, generating a short clip somebody can watch.

Too heavy for an automated run, so it skips unless asked for, and heavier here
than anywhere else: two 14B checkpoints. Run it with a base temporary directory
you can reach, then play the file::

    T2V_FASTVIDEO_CAUSAL_WAN22_REAL_MODEL_RUN=1 uv run --no-sync pytest \
        integrations_v2/t2v_fastvideo_causal_wan22 -m ci_gpu -s \
        --basetemp="$HOME/t2v-out"
    vlc "$HOME"/t2v-out/*current/clip.mp4

What the run does, and why it skips, is
:func:`flashdreams.t2v_v2.testing.check_real_model_generates_a_clip`.
"""

from pathlib import Path

import pytest
from fastvideo_causal_wan22.runner import DEFAULT_T2V_PROMPT
from t2v_fastvideo_causal_wan22 import FastvideoCausalWan22T2VApplication

from flashdreams.t2v_v2.testing import (
    check_real_model_generates_a_clip,
    real_model_run_skip_reason,
)

pytestmark = pytest.mark.ci_gpu

_SKIP = real_model_run_skip_reason("T2V_FASTVIDEO_CAUSAL_WAN22_REAL_MODEL_RUN")
"""Why this cannot run here, if it cannot."""

_STEPS = 3
"""Blocks to generate. Two seconds of video, and enough that the steady-state
block size is covered as well as the first one."""

_FIRST_BLOCK_FRAMES = 9
"""Frames the first block decodes. The config generates three latent frames a
block, and the Wan VAE decodes the first to one frame and each after it to four,
so the first block is 1 + 2 * 4."""

_BLOCK_FRAMES = 12
"""Frames every block after it decodes, being 3 * 4."""


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_the_model_generates_a_clip_worth_watching(tmp_path: Path) -> None:
    result = check_real_model_generates_a_clip(
        FastvideoCausalWan22T2VApplication(),
        prompt=DEFAULT_T2V_PROMPT,
        steps=_STEPS,
        frame_count=_FIRST_BLOCK_FRAMES + (_STEPS - 1) * _BLOCK_FRAMES,
        mp4_path=tmp_path / "clip.mp4",
    )

    assert result.passed, result.failures
