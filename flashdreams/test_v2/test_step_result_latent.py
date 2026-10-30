# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the latent ``StepResult`` kind and its RGB-consumer guards.

A latent result carries a pre-decode latent chunk as ``output`` for the
token-streaming path. It must read back like any output (same tensor, CUDA
readiness) yet be rejected loudly by every RGB consumer so a latent can never
be composited or encoded as pixels.
"""

from __future__ import annotations

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import _frame_at
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def test_rgb_step_result_is_not_latent() -> None:
    frames = torch.zeros(3, 3, 4, 4)  # tchw: 3 frames, RGB
    result = StepResult(
        step_index=0,
        output=frames,
        frame_count=3,
        output_layout=VideoTensorLayout.tchw,
    )

    assert result.is_latent is False
    assert result.read_output() is frames


def test_latent_step_result_carries_the_latent() -> None:
    latent = torch.zeros(1, 1, 2, 48, 4, 4)  # bvtchw with 48 latent channels
    result = StepResult.latent(step_index=1, output=latent, frame_count=2)

    assert result.is_latent is True
    assert result.output_layout is VideoTensorLayout.bvtchw
    assert result.frame_count == 2
    assert result.read_output() is latent


def test_frame_at_rejects_a_latent_result() -> None:
    latent = torch.zeros(1, 1, 2, 48, 4, 4)
    result = StepResult.latent(step_index=0, output=latent, frame_count=2)

    with pytest.raises(ValueError, match="latent"):
        _frame_at(result, 0)
