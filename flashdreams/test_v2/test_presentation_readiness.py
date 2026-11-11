# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pending GPU output stays bounded and never enters the UI consumer path."""

import threading

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import BackpressureMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class Event:
    def __init__(self, ready=False):
        self.ready = ready
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.ready

    def synchronize(self):
        raise AssertionError("No host wait for presentation readiness")


def result(index=0, *, frames=1, event=None):
    value = StepResult(
        index,
        torch.full((frames, 3, 1, 1), float(index)),
        frames,
        VideoTensorLayout.tchw,
    )
    if event is not None:
        object.__setattr__(value, "_output_ready_event", event)
    return value


def manager(mode=BackpressureMode.BLOCK):
    value = PresentationManager(device=torch.device("cpu"))
    value.configure(
        backpressure_mode=mode,
        stop=threading.Event(),
        put_timeout=0.005,
        frames_per_second=10,
        maximum_frames_per_second=60,
    )
    return value


def test_pending_channels_keep_queue_slot_and_never_read_output(monkeypatch):
    pending = Event()
    value = result(event=pending)
    m = manager()

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Pending output must not schedule a UI stream dependency")

    monkeypatch.setattr(StepResult, "read_output", forbidden_read)
    m.publish(0, [value])
    for now in (0.0, 0.1, 0.3):
        assert m.advance(0, now=now) == (False, None)
        assert m.presented_frames() == ()
        assert m.has_pending_frames()  # model-done must continue draining
        assert m.buffered_chunk_count == m.buffer_capacity == 1
    pending.ready = True
    assert m.advance(0, now=0.4) == (True, [value])
    assert m.buffered_chunk_count == 0
    assert not m.has_pending_frames()


def test_all_channels_must_complete_before_selection():
    a, b = Event(True), Event(False)
    chunk = [result(0, event=a), result(1, event=b)]
    m = manager()
    m.publish(0, chunk)
    assert not m.advance(0, now=0)[0]
    b.ready = True
    assert m.advance(0, now=0.1) == (True, chunk)


def test_pending_backlog_does_not_rush_active_frames():
    m = manager()
    m.publish(0, [result(0, frames=2)])
    assert m.advance(0, now=0)[0]
    m.publish(0, [result(1, event=Event())])
    assert m.is_backlogged  # still occupies the bounded queue
    assert not m.advance(0, now=0.01)[0]
    assert m.advance(0, now=0.1)[0]
    assert float(m.presented_frame(0)[0, 0, 0]) == 0
    assert not m.advance(0, now=0.2)[0]
    assert float(m.presented_frame(0)[0, 0, 0]) == 0


def test_block_waits_for_ready_selection_not_just_readiness_poll():
    m = manager()
    ready = Event()
    m.publish(0, [result(0, event=ready)])
    entered, done = threading.Event(), threading.Event()

    def publish():
        entered.set()
        m.publish(0, [result(1)])
        done.set()

    thread = threading.Thread(target=publish)
    thread.start()
    assert entered.wait(1)
    try:
        assert not done.wait(0.02)
        assert not m.advance(0, now=0)[0]
        assert not done.is_set()
        ready.ready = True
        assert m.advance(0, now=0.1)[0]
        assert done.wait(1)
        assert m.buffered_chunk_count == 1
    finally:
        m._stop.set()
        thread.join(1)
    assert not thread.is_alive()


def test_drop_oldest_waits_before_enqueue_and_counts_completion_time(monkeypatch):
    m = manager(BackpressureMode.DROP_OLDEST)
    now = 0.0
    waited = []

    class CompletingEvent(Event):
        def synchronize(self):
            nonlocal now
            assert m.buffered_chunk_count == 0
            waited.append(now)
            now += 0.3
            self.ready = True

    monkeypatch.setattr(
        "flashdreams.runtime_v2.presentation_manager.time.monotonic", lambda: now
    )
    for i in range(2):
        event = CompletingEvent()
        m.publish(0, [result(i, event=event)], step_elapsed_s=0.4)
        assert event.ready
        assert m.buffered_chunk_count == 1
        m.clear()
        now += 0.4
    assert len(waited) == 2
    assert m._presentation_clock.frames_per_second == pytest.approx(1 / 0.7 / 0.98)


def test_drop_oldest_replaces_completed_output_in_same_single_slot():
    m = manager(BackpressureMode.DROP_OLDEST)
    m.publish(0, [result(0)])
    m.publish(0, [result(1)])
    assert m.buffered_chunk_count == 1
    assert m.dropped_for_space == 1
    changed, chunk = m.advance(0, now=0.1)
    assert changed and chunk[0].step_index == 1


def test_drop_replacement_between_poll_and_selection_rechecks_current_head(monkeypatch):
    m = manager(BackpressureMode.DROP_OLDEST)
    old, new = result(0), result(1)
    m.publish(0, [old])

    def replace_before_selection(*args, **kwargs):
        producer = threading.Thread(target=lambda: m.publish(0, [new]))
        producer.start()
        producer.join(1)
        assert not producer.is_alive()
        return True

    monkeypatch.setattr(m._presentation_clock, "is_due", replace_before_selection)
    assert m.advance(0, now=0) == (True, [new])
    assert m.buffered_chunk_count == 0
    assert m.dropped_for_space == 1


def test_reset_discards_unready_and_late_old_generation_without_polling():
    m = manager()
    old = Event()
    m.publish(0, [result(event=old)])
    assert m.advance(1, now=0) == (False, None)
    assert old.queries == 0
    assert not m.has_pending_frames()
    m.publish(0, [result(event=old)])
    assert m.advance(1, now=0.1) == (False, None)
    assert old.queries == 0
    assert m.discarded_at_reset == 2
    m.publish(1, [result(2)])
    assert m.advance(1, now=0.2)[0]


def test_gpu_pacing_uses_polled_completion_intervals_once_per_chunk():
    m = manager()
    a = Event()
    m.publish(0, [result(0, event=a)], step_elapsed_s=0.001)
    assert not m.advance(0, now=0)[0]
    a.ready = True
    assert m.advance(0, now=0.5)[0]
    b = Event()
    m.publish(0, [result(1, event=b)], step_elapsed_s=0.001)
    assert not m.advance(0, now=0.6)[0]
    b.ready = True
    assert m.advance(0, now=1.5)[0]
    assert m._presentation_clock.frames_per_second == pytest.approx(1 / 0.98)
    assert not m.advance(0, now=2)[0]
    assert m._presentation_clock.frames_per_second == pytest.approx(1 / 0.98)


def test_zero_completion_interval_clamps_and_reset_restarts_pacing():
    m = manager()
    for i in range(2):
        m.publish(0, [result(i, event=Event(True))], step_elapsed_s=0)
        assert m.advance(0, now=0)[0] == (i == 0)
    assert m._presentation_clock.frames_per_second == 60
    assert m.advance(1, now=1) == (False, None)
    m.publish(1, [result(2, event=Event(True))], step_elapsed_s=0)
    assert m.advance(1, now=1)[0]
    assert m._presentation_clock.frames_per_second == 10
