# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue

import numpy as np
import pytest

from flashdreams.serving.realtime.presenter import (
    PresentationQueue,
    materialize_rgb_host_uint8,
    wait_until_present_time,
)

pytestmark = pytest.mark.ci_cpu


def test_presentation_queue_drops_oldest_frame_at_capacity() -> None:
    frames = PresentationQueue[str](capacity=2)

    assert frames.append("first") is None
    assert frames.append("second") is None
    assert frames.append("third") == "first"

    assert list(frames) == ["second", "third"]


def test_presentation_queue_drains_available_frames_with_filter_and_prepare() -> None:
    source: queue.Queue[int] = queue.Queue()
    for value in [1, 2, 3, 4]:
        source.put(value)
    prepared: list[int] = []
    ready = PresentationQueue[int](capacity=2)

    result = ready.drain_nowait(
        source,
        include=lambda value: value % 2 == 0,
        prepare=prepared.append,
    )

    assert result.pulled == 4
    assert result.accepted == 2
    assert result.skipped == 2
    assert result.dropped == ()
    assert prepared == [2, 4]
    assert list(ready) == [2, 4]


def test_presentation_queue_reports_drops_during_drain() -> None:
    source: queue.Queue[str] = queue.Queue()
    for value in ["first", "second", "third"]:
        source.put(value)
    ready = PresentationQueue[str](capacity=1)

    result = ready.drain_nowait(source)

    assert result.accepted == 3
    assert result.dropped == ("first", "second")
    assert ready.pop_ready() == "third"
    assert ready.pop_ready() is None


def test_presentation_queue_clear_flushes_ready_frames() -> None:
    ready = PresentationQueue[int]()
    ready.append(1)
    ready.append(2)

    assert ready.clear() == 2
    assert len(ready) == 0


def test_wait_until_present_time_sleeps_until_poll_timeout() -> None:
    times = iter([1.0, 1.25])
    sleeps: list[float] = []

    wait = wait_until_present_time(
        2.0,
        poll_timeout_s=0.25,
        clock=lambda: next(times),
        sleep=sleeps.append,
    )

    assert wait is not None
    assert wait.begin_time == 1.0
    assert wait.end_time == 1.25
    assert wait.duration_s == pytest.approx(0.25)
    assert sleeps == [0.25]


def test_wait_until_present_time_returns_none_when_due() -> None:
    wait = wait_until_present_time(
        1.0,
        poll_timeout_s=0.25,
        clock=lambda: 1.1,
        sleep=lambda _seconds: None,
    )

    assert wait is None


def test_materialize_rgb_host_uint8_strips_alpha_from_lazy_frame() -> None:
    class LazyFrame:
        def to_numpy(self) -> np.ndarray:
            return np.array(
                [[[1, 2, 3, 255], [4, 5, 6, 255]]],
                dtype=np.uint8,
            )

    frame = materialize_rgb_host_uint8(LazyFrame())

    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(
        frame,
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
    )
