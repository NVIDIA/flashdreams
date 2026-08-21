# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real model, generating a short clip somebody can watch.

Skips unless asked for, being too heavy to run automatically. Give it a base
temporary directory you can reach, then play the file::

    T2V_WAN21_REAL_MODEL_RUN=1 uv run --no-sync pytest \
        integrations_v2/t2v_wan21 -m ci_gpu -s --basetemp="$HOME/t2v-out"
    vlc "$HOME"/t2v-out/*current/clip.mp4
"""

from pathlib import Path

import pytest
from t2v_wan21 import Wan21T2VApplication
from wan21.runner import DEFAULT_PROMPT

from flashdreams.t2v_v2.testing import (
    check_real_model_generates_a_clip,
    real_model_run_skip_reason,
)

pytestmark = pytest.mark.ci_gpu

_SKIP = real_model_run_skip_reason("T2V_WAN21_REAL_MODEL_RUN")
"""Why this cannot run here, if it cannot."""

_CLIP_FRAMES = 81
"""Frames the one block decodes: 21 latent frames, as 1 + 20 * 4. About five
seconds at 16 frames per second."""


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_the_model_generates_a_clip_worth_watching(tmp_path: Path) -> None:
    result = check_real_model_generates_a_clip(
        Wan21T2VApplication(),
        prompt=DEFAULT_PROMPT,
        # One block is the whole clip, for a model that attends over all of it.
        steps=1,
        frame_count=_CLIP_FRAMES,
        mp4_path=tmp_path / "clip.mp4",
    )

    assert result.passed, result.failures
