# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real model, generating a short clip somebody can watch.

Skips unless asked for, being too heavy to run automatically. Give it a base
temporary directory you can reach, then play the file::

    T2V_SELF_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
        integrations_v2/t2v_self_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
    vlc "$HOME"/t2v-out/*current/clip.mp4
"""

from pathlib import Path

import pytest
from self_forcing.runner import DEFAULT_T2V_PROMPT
from t2v_self_forcing import SelfForcingT2VApplication

from flashdreams.t2v_v2.testing import (
    check_real_model_generates_a_clip,
    real_model_run_skip_reason,
)

pytestmark = pytest.mark.ci_gpu

_SKIP = real_model_run_skip_reason("T2V_SELF_FORCING_REAL_MODEL_RUN")
"""Why this cannot run here, if it cannot."""

_STEPS = 3
"""Blocks to generate: enough to cover the steady-state size and the first."""

_FIRST_BLOCK_FRAMES = 9
"""Frames the first block decodes."""

_BLOCK_FRAMES = 12
"""Frames every block after it decodes."""


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_the_model_generates_a_clip_worth_watching(tmp_path: Path) -> None:
    result = check_real_model_generates_a_clip(
        SelfForcingT2VApplication(),
        prompt=DEFAULT_T2V_PROMPT,
        steps=_STEPS,
        frame_count=_FIRST_BLOCK_FRAMES + (_STEPS - 1) * _BLOCK_FRAMES,
        mp4_path=tmp_path / "clip.mp4",
    )

    assert result.passed, result.failures
