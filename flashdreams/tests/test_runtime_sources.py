# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for replay and live user-input sources."""

from __future__ import annotations

import threading

import pytest

from flashdreams.runtime.inputs import TimeWindow, UserInputEvent, UserInputs
from flashdreams.runtime.sources import QueuedUserInputSource, UserInputSource

pytestmark = pytest.mark.ci_cpu


def _event(timestamp_s: float, key: str = "w") -> UserInputEvent:
    return UserInputEvent(
        timestamp_s=timestamp_s,
        event_type="keydown",
        payload={"key": key},
    )


def test_user_inputs_batch_satisfies_source_protocol() -> None:
    """A fully-known replay batch is usable as a source with no adaptation."""
    batch = UserInputs(events=(_event(0.0), _event(1.0)))

    assert isinstance(batch, UserInputSource)


def test_queued_source_satisfies_source_protocol() -> None:
    assert isinstance(QueuedUserInputSource(), UserInputSource)


def test_queued_source_windows_are_half_open() -> None:
    source = QueuedUserInputSource()
    source.extend([_event(0.0), _event(1.0), _event(2.0)])

    windowed = source.window(TimeWindow(start_s=0.0, end_s=2.0))

    assert [event.timestamp_s for event in windowed.events] == [0.0, 1.0]


def test_queued_source_discards_events_before_the_requested_window() -> None:
    """Retention is what bounds the queue across a long live session."""
    source = QueuedUserInputSource()
    source.extend([_event(0.0), _event(1.0), _event(2.0)])

    source.window(TimeWindow(start_s=2.0, end_s=3.0))

    assert source.pending_count == 1


def test_queued_source_keeps_events_beyond_the_requested_window() -> None:
    source = QueuedUserInputSource()
    source.extend([_event(0.0), _event(5.0)])

    windowed = source.window(TimeWindow(start_s=0.0, end_s=1.0))

    assert [event.timestamp_s for event in windowed.events] == [0.0]
    assert source.pending_count == 2


def test_queued_source_rejects_out_of_order_arrivals() -> None:
    source = QueuedUserInputSource()
    source.append(_event(1.0))

    with pytest.raises(ValueError, match="non-decreasing"):
        source.append(_event(0.5))


def test_queued_source_reset_clears_events_and_ordering_watermark() -> None:
    source = QueuedUserInputSource()
    source.append(_event(5.0))

    source.reset()
    source.append(_event(0.0))

    assert source.pending_count == 1


def test_queued_source_carries_snapshot_and_metadata_into_windows() -> None:
    source = QueuedUserInputSource(
        snapshot={"pressed": ("w",)}, metadata={"transport": "local-window"}
    )

    windowed = source.window(TimeWindow(start_s=0.0, end_s=1.0))

    assert windowed.snapshot["pressed"] == ("w",)
    assert windowed.metadata["transport"] == "local-window"


def test_queued_source_set_snapshot_applies_to_later_windows() -> None:
    source = QueuedUserInputSource(snapshot={"pressed": ()})

    source.set_snapshot({"pressed": ("a",)})

    assert source.window(TimeWindow(start_s=0.0, end_s=1.0)).snapshot["pressed"] == (
        "a",
    )


def test_queued_source_appends_from_a_producer_thread_are_all_observed() -> None:
    """A transport thread feeds the queue while the loop thread reads it."""
    source = QueuedUserInputSource()
    total = 200

    def produce() -> None:
        for index in range(total):
            source.append(_event(float(index)))

    producer = threading.Thread(target=produce)
    producer.start()
    producer.join()

    windowed = source.window(TimeWindow(start_s=0.0, end_s=float(total)))
    assert len(windowed.events) == total
