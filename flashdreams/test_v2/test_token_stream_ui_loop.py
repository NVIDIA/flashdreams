# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for TokenStreamUILoop forwarding presented latent chunks.

The token UI loop reads the presented latent chunk (selected by the manager's
latent passthrough) and returns it uncomposited, exactly once per chunk.
"""

from __future__ import annotations

import queue
import threading

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import BackpressureMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.token_stream_ui_loop import TokenStreamUILoop
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _latent_chunk(step_index: int) -> list[StepResult]:
    latent = torch.zeros(1, 1, 2, 48, 4, 4)
    return [StepResult.latent(step_index=step_index, output=latent, frame_count=2)]


def _loop_with_manager() -> tuple[TokenStreamUILoop[None], PresentationManager]:
    pm = PresentationManager(device=torch.device("cpu"))
    pm.configure(
        backpressure_mode=BackpressureMode.BLOCK,
        stop=threading.Event(),
        put_timeout=0.01,
        latent_passthrough=True,
    )
    loop: TokenStreamUILoop[None] = TokenStreamUILoop()
    loop.register_session_loop_objects(
        state=None,
        frequency=30,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    loop.register_session_ui_loop_objects(
        session_desc=SessionDesc(token_stream=True),
        presentation_manager=pm,
    )
    return loop, pm


def test_token_ui_loop_forwards_each_latent_once_in_order() -> None:
    loop, pm = _loop_with_manager()
    forwarded: list[int] = []
    for i in range(3):
        pm.publish(generation=0, chunk=_latent_chunk(i))
        pm.advance(generation=0)

        out = loop.step(0, UserInputEvents([]))
        assert len(out) == 1
        assert out[0].is_latent is True
        assert out[0].step_index == i
        forwarded.append(out[0].step_index)

        # Same presented chunk, no new advance: deduped to nothing.
        assert loop.step(0, UserInputEvents([])) == []
    assert forwarded == [0, 1, 2]


def test_token_ui_loop_emits_nothing_before_a_chunk_is_presented() -> None:
    loop, _ = _loop_with_manager()
    assert loop.step(0, UserInputEvents([])) == []
