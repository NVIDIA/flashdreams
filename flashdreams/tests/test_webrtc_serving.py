# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from flashdreams.serving.webrtc.controls import (
    WSAD_SUPPORTED_KEYS,
    KeyboardResampler,
    KeyboardState,
)
from flashdreams.serving.webrtc.media import tensor_chunk_to_rgb_frames

pytestmark = pytest.mark.ci_cpu


def test_wsad_keyboard_state_rejects_non_driving_keys() -> None:
    state = KeyboardState(supported_keys=WSAD_SUPPORTED_KEYS)

    assert state.apply_event(event="keydown", key="ArrowUp")
    assert state.resolved_effective_keys() == frozenset({"w"})
    assert not state.apply_event(event="keydown", key="q")
    assert state.resolved_effective_keys() == frozenset({"w"})


def test_wsad_resampler_preserves_held_key() -> None:
    resampler = KeyboardResampler(
        fps=30,
        start_v=1.0,
        supported_keys=WSAD_SUPPORTED_KEYS,
    )
    resampler.on_edge(arrival_t=0.5, event="keydown", key="w")

    segments, frame_times = resampler.sample_chunk(num_frames=2)

    assert segments == [(1.0, 1.0 + 2 / 30, frozenset({"w"}))]
    assert frame_times == pytest.approx([1.0 + 1 / 30, 1.0 + 2 / 30])


def test_tensor_chunk_to_rgb_frames_supports_alpadreams_layout() -> None:
    chunk = torch.zeros((1, 1, 2, 3, 4, 5), dtype=torch.uint8)
    chunk[0, 0, 1, 0] = 255

    frames = tensor_chunk_to_rgb_frames(chunk)

    assert len(frames) == 2
    assert frames[0].shape == (4, 5, 3)
    assert frames[0].dtype == np.uint8
    assert frames[1][0, 0, 0] == 255
