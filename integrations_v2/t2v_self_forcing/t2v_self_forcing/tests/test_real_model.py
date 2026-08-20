# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real model, generating a short clip somebody can watch.

Too heavy for any automated run: it needs a GPU, and on a machine that has not
run this model before it downloads tens of gigabytes of checkpoint. So it
carries the ``ci_gpu`` tier marker but skips unless
``T2V_SELF_FORCING_REAL_MODEL_RUN`` is set, which costs the GPU job milliseconds
and still keeps the module imported and collected there. Run it with a base
temporary directory you can reach, then play the file::

    T2V_SELF_FORCING_REAL_MODEL_RUN=1 uv run --no-sync pytest \
        integrations_v2/t2v_self_forcing -m ci_gpu -s --basetemp="$HOME/t2v-out"
    vlc "$HOME"/t2v-out/*current/clip.mp4

The ``manual`` marker describes this test better and cannot be used: the
``pytest-manual-marker`` plugin xfails every ``manual`` test at setup, so a test
marked that way never runs, here or on anybody's machine.
"""

import os
import shutil
from pathlib import Path

import pytest
import torch
from self_forcing.runner import DEFAULT_T2V_PROMPT
from t2v_self_forcing import SelfForcingT2VApplication, default_session_desc

from flashdreams.testing_v2.t2v_conformance import (
    ExpectedFrameStats,
    check_t2v_model_impl,
)

pytestmark = pytest.mark.ci_gpu

_RUN_ENV = "T2V_SELF_FORCING_REAL_MODEL_RUN"
"""Set this to run the model rather than skip it."""

_STEPS = 3
"""Blocks to generate. Two seconds of video, and enough that the steady-state
block size is covered as well as the first one."""

_FIRST_BLOCK_FRAMES = 9
"""Frames the first block decodes."""

_BLOCK_FRAMES = 12
"""Frames every block after it decodes."""


@pytest.mark.skipif(
    not os.environ.get(_RUN_ENV),
    reason=f"set {_RUN_ENV}=1 to download the checkpoints and generate a clip",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the model needs a GPU")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="writing an MP4 needs ffmpeg")
def test_the_model_generates_a_clip_worth_watching(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"

    result = check_t2v_model_impl(
        SelfForcingT2VApplication(),
        default_session_desc(),
        steps=_STEPS,
        # Compilation costs minutes and buys back milliseconds a block, which
        # is the wrong trade for three blocks.
        commandline_args=["--prompt", DEFAULT_T2V_PROMPT, "--no-compile"],
        expected=ExpectedFrameStats(
            frame_count=_FIRST_BLOCK_FRAMES + (_STEPS - 1) * _BLOCK_FRAMES,
            # A picture rather than a blank frame. Loose, because what a model
            # samples is its own business.
            mean_luminance=(16.0, 240.0),
            min_frame_difference=0.5,
        ),
        mp4_path=path,
    )

    print(f"\nwrote {path}\n{result}")
    assert result.passed, result.failures
