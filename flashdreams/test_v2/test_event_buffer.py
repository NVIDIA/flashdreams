# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for lossless cross-loop input retention."""

import pytest
from numpy import uint64

from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def _move(timestamp: int, x: float) -> MouseUserInputEvent:
    return MouseUserInputEvent(timestamp=uint64(timestamp), x=x, y=0.5)


def test_consecutive_pointer_moves_are_retained_for_fast_and_slow_readers() -> None:
    buffer = EventBuffer()
    buffer.register(0)
    buffer.register(1)

    buffer.append(UserInputEvents([_move(1, 0.1)]))
    first_events = buffer.read(0)[0].get_events()
    assert len(first_events) == 1
    assert isinstance(first_events[0], MouseUserInputEvent)
    assert (first_events[0].timestamp, first_events[0].x) == (uint64(1), 0.1)

    buffer.append(UserInputEvents([_move(2, 0.2), _move(3, 0.3)]))
    fast_events, _ = buffer.read(0)
    assert [
        (event.timestamp, event.x)
        for event in fast_events.get_events()
        if isinstance(event, MouseUserInputEvent)
    ] == [(uint64(2), 0.2), (uint64(3), 0.3)]

    key = KeyboardUserInputEvent(
        timestamp=uint64(4),
        key="w",
        state=KeyboardInputState.PRESSED,
    )
    buffer.append(UserInputEvents([key, _move(5, 0.5), _move(6, 0.6)]))

    slow_events, _ = buffer.read(1)
    assert [type(event) for event in slow_events.get_events()] == [
        MouseUserInputEvent,
        MouseUserInputEvent,
        MouseUserInputEvent,
        KeyboardUserInputEvent,
        MouseUserInputEvent,
        MouseUserInputEvent,
    ]
    assert [
        event.x
        for event in slow_events.get_events()
        if isinstance(event, MouseUserInputEvent)
    ] == [0.1, 0.2, 0.3, 0.5, 0.6]


def test_each_read_reports_when_its_events_reached_the_runtime() -> None:
    """Latency starts here, since event timestamps count from each window's start."""
    buffer = EventBuffer()
    buffer.register(0)
    buffer.append(UserInputEvents([_move(1, 0.1), _move(2, 0.2)]))
    buffer.read(0)
    first = buffer.last_read_received_ns(0)
    buffer.append(UserInputEvents([_move(3, 0.3)]))
    buffer.collect_garbage()
    buffer.read(0)
    second = buffer.last_read_received_ns(0)

    assert len(first) == 2 and first[0] == first[1], "one stamp per batch"
    assert len(second) == 1 and second[0] >= first[0]
