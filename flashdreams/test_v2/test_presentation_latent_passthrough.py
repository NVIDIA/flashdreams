# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for PresentationManager latent passthrough (token streaming).

A token-stream session must deliver every latent chunk exactly once, in order,
with no drops and no fps pacing. This drives the manager directly, without a UI
loop or WebSocket.
"""

from __future__ import annotations

import threading

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import BackpressureMode
from flashdreams.runtime_v2.step_result import StepResult

pytestmark = pytest.mark.ci_cpu


def _latent_chunk(step_index: int) -> list[StepResult]:
    latent = torch.zeros(1, 1, 2, 48, 4, 4)  # bvtchw, Cl=48
    return [StepResult.latent(step_index=step_index, output=latent, frame_count=2)]


def _manager() -> PresentationManager:
    pm = PresentationManager(device=torch.device("cpu"))
    pm.configure(
        backpressure_mode=BackpressureMode.DROP_OLDEST,  # forced to BLOCK below
        stop=threading.Event(),
        put_timeout=0.01,
        latent_passthrough=True,
    )
    return pm


def test_latent_passthrough_forces_block_backpressure() -> None:
    # A token session must never drop a latent chunk, whatever mode was asked.
    assert _manager()._backpressure_mode is BackpressureMode.BLOCK


def test_latent_passthrough_delivers_each_chunk_once_in_order() -> None:
    pm = _manager()
    received: list[int] = []
    for i in range(5):
        pm.publish(generation=0, chunk=_latent_chunk(i))
        changed, taken = pm.advance(generation=0)
        assert changed is True
        assert taken is not None
        result = pm.presented_latent_result()
        assert result is not None
        assert result.is_latent is True
        received.append(result.step_index)
        # Nothing new is buffered: advance must not re-select, so the UI loop
        # will not re-emit the same latent chunk.
        changed_again, _ = pm.advance(generation=0)
        assert changed_again is False
    assert received == [0, 1, 2, 3, 4]


def test_latent_passthrough_takes_whole_chunks_without_fps_pacing() -> None:
    # Two chunks published back to back are each taken immediately (no clock
    # gating, no per-frame stepping between them).
    pm = _manager()
    pm.publish(generation=0, chunk=_latent_chunk(0))
    assert pm.advance(generation=0)[0] is True
    first = pm.presented_latent_result()
    assert first is not None and first.step_index == 0

    pm.publish(generation=0, chunk=_latent_chunk(1))
    assert pm.advance(generation=0)[0] is True
    second = pm.presented_latent_result()
    assert second is not None and second.step_index == 1


def test_rgb_result_is_not_returned_as_latent() -> None:
    pm = _manager()
    frames = torch.zeros(2, 3, 4, 4)  # tchw RGB
    from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

    pm.publish(
        generation=0,
        chunk=[
            StepResult(
                step_index=0,
                output=frames,
                frame_count=2,
                output_layout=VideoTensorLayout.tchw,
            )
        ],
    )
    assert pm.advance(generation=0)[0] is True
    # An RGB chunk that somehow reached a token session is not surfaced as latent.
    assert pm.presented_latent_result() is None
