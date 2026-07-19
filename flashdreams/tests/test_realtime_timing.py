# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from flashdreams.serving.realtime.timing import (
    ChunkHistory,
    ChunkTimes,
    InputToPresentProfileWindow,
    TraceComponentValue,
    TraceContext,
    VideoModelTimings,
    chunk_stage_durations_ms,
    emit_video_model_timing_ranges,
    summarize_chunk_history,
)

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _TraceEvent:
    name: str
    depends_on: list[int]
    components: dict[str, TraceComponentValue]


class _RecordingTraceSink:
    def __init__(self) -> None:
        self.threads: list[str] = []
        self.events: list[_TraceEvent] = []

    def add_thread(self, name: str) -> int:
        self.threads.append(name)
        return len(self.threads) - 1

    def add_instant(
        self,
        name: str,
        *,
        thread: int,
        time_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int:
        del thread, time_ns
        return self._append_event(name, depends_on, components)

    def add_range(
        self,
        name: str,
        *,
        thread: int,
        begin_ns: int,
        end_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int:
        del thread
        event_components = dict(components)
        event_components["begin_ns"] = begin_ns
        event_components["end_ns"] = end_ns
        return self._append_event(name, depends_on, event_components)

    def _append_event(
        self,
        name: str,
        depends_on: list[int] | None,
        components: dict[str, TraceComponentValue],
    ) -> int:
        self.events.append(
            _TraceEvent(
                name=name,
                depends_on=[] if depends_on is None else depends_on,
                components=components,
            )
        )
        return len(self.events) - 1


class _NamedIdTraceSink(_RecordingTraceSink):
    def __init__(self, event_ids: dict[str, int]) -> None:
        super().__init__()
        self._event_ids = event_ids

    def _append_event(
        self,
        name: str,
        depends_on: list[int] | None,
        components: dict[str, TraceComponentValue],
    ) -> int:
        super()._append_event(name, depends_on, components)
        return self._event_ids[name]


def _chunk_times() -> ChunkTimes:
    chunk = ChunkTimes.create(
        chunk_index=7,
        input_sample_time=1.0,
        request_time=1.1,
        request_poses_ready_time=1.2,
        intended_present_times=[1.5, 1.6],
    )
    chunk.chunk_render_start_time = 1.25
    chunk.chunk_ready_time = 1.45
    chunk.frames[0].image_ready_time = 1.47
    chunk.frames[0].sample_display_pose_time = 1.49
    chunk.frames[0].present_time = 1.52
    return chunk


def test_chunk_times_create_allocates_frame_times() -> None:
    chunk = ChunkTimes.create(
        chunk_index=0,
        input_sample_time=1.0,
        request_time=1.0,
        request_poses_ready_time=1.001,
        intended_present_times=[1.5, 1.6, 1.7],
    )

    assert [frame.frame_index for frame in chunk.frames] == [0, 1, 2]
    assert [frame.intended_present_time for frame in chunk.frames] == [1.5, 1.6, 1.7]


def test_chunk_history_iter_returns_iterator() -> None:
    first = _chunk_times()
    second = _chunk_times()
    history = ChunkHistory(capacity=2)
    history.append(first)
    history.append(second)

    iterator = iter(history)

    assert isinstance(iterator, Iterator)
    assert next(iterator) is first
    assert next(iterator) is second


def test_chunk_stage_durations_are_derived_from_milestones() -> None:
    durations = chunk_stage_durations_ms(_chunk_times())

    assert durations["input_to_request"] == pytest.approx(100.0)
    assert durations["request_to_poses_ready"] == pytest.approx(100.0)
    assert durations["queue_wait"] == pytest.approx(50.0)
    assert durations["chunk_render"] == pytest.approx(200.0)
    assert durations["chunk_ready_to_first_image"] == pytest.approx(20.0)
    assert durations["first_image_to_present"] == pytest.approx(50.0)
    assert durations["input_to_first_present"] == pytest.approx(520.0)


def test_summarize_chunk_history_builds_stage_statistics() -> None:
    first = _chunk_times()
    second = _chunk_times()
    second.chunk_render_start_time = 2.0
    second.chunk_ready_time = 2.4

    summary = summarize_chunk_history([first, second])

    assert summary.chunk_count == 2
    assert summary.stages["chunk_render"].count == 2
    assert summary.stages["chunk_render"].avg_ms == pytest.approx(300.0)
    assert summary.stages["chunk_render"].p90_ms == pytest.approx(380.0)


def test_video_model_timings_include_optional_decode_and_cache_durations() -> None:
    timings = VideoModelTimings(
        condition_start_time=1.0,
        condition_ready_time=1.1,
        model_start_time=1.1,
        model_ready_time=1.6,
        merge_start_time=1.65,
        merge_ready_time=1.7,
        cache_update_start_time=1.2,
        cache_update_ready_time=1.3,
        decode_start_time=1.4,
        decode_ready_time=1.55,
    )

    durations = timings.stage_durations_ms()

    assert durations["condition"] == pytest.approx(100.0)
    assert durations["model"] == pytest.approx(500.0)
    assert durations["cache_update"] == pytest.approx(100.0)
    assert durations["decode"] == pytest.approx(150.0)
    assert durations["merge"] == pytest.approx(50.0)
    assert durations["total"] == pytest.approx(700.0)


def test_emit_video_model_timing_ranges_adds_optional_subranges() -> None:
    sink = _RecordingTraceSink()
    trace_context = TraceContext.create(sink)
    timings = VideoModelTimings(
        condition_start_time=1.0,
        condition_ready_time=1.1,
        model_start_time=1.1,
        model_ready_time=1.6,
        merge_start_time=1.65,
        merge_ready_time=1.7,
        cache_update_start_time=1.2,
        cache_update_ready_time=1.3,
        decode_start_time=1.4,
        decode_ready_time=1.55,
    )

    events = emit_video_model_timing_ranges(
        trace_context,
        timings=timings,
        thread=trace_context.worker_thread,
        depends_on=[123],
        chunk_index=9,
    )

    names = [event.name for event in sink.events]
    assert names == [
        "condition_raster",
        "model_generate",
        "cache_update",
        "decode",
        "frame_merge",
    ]
    assert sink.events[0].depends_on == [123]
    assert sink.events[2].depends_on == [events.model_event_id]
    assert sink.events[3].depends_on == [events.cache_update_event_id]
    assert sink.events[4].depends_on == [events.decode_event_id]
    assert events.final_event_id == events.merge_event_id


def test_emit_video_model_timing_ranges_keeps_zero_valued_event_ids() -> None:
    sink = _NamedIdTraceSink(
        {
            "condition_raster": 10,
            "model_generate": 11,
            "cache_update": 0,
            "decode": 12,
            "frame_merge": 13,
        }
    )
    trace_context = TraceContext.create(sink)
    timings = VideoModelTimings(
        condition_start_time=1.0,
        condition_ready_time=1.1,
        model_start_time=1.1,
        model_ready_time=1.6,
        merge_start_time=1.65,
        merge_ready_time=1.7,
        cache_update_start_time=1.2,
        cache_update_ready_time=1.3,
        decode_start_time=1.4,
        decode_ready_time=1.55,
    )

    events = emit_video_model_timing_ranges(
        trace_context,
        timings=timings,
        thread=trace_context.worker_thread,
        chunk_index=9,
    )

    assert events.cache_update_event_id == 0
    assert sink.events[3].depends_on == [0]


def test_emit_video_model_timing_ranges_uses_zero_decode_event_for_merge() -> None:
    sink = _NamedIdTraceSink(
        {
            "condition_raster": 10,
            "model_generate": 11,
            "cache_update": 12,
            "decode": 0,
            "frame_merge": 13,
        }
    )
    trace_context = TraceContext.create(sink)
    timings = VideoModelTimings(
        condition_start_time=1.0,
        condition_ready_time=1.1,
        model_start_time=1.1,
        model_ready_time=1.6,
        merge_start_time=1.65,
        merge_ready_time=1.7,
        cache_update_start_time=1.2,
        cache_update_ready_time=1.3,
        decode_start_time=1.4,
        decode_ready_time=1.55,
    )

    events = emit_video_model_timing_ranges(
        trace_context,
        timings=timings,
        thread=trace_context.worker_thread,
        chunk_index=9,
    )

    assert events.decode_event_id == 0
    assert sink.events[4].depends_on == [0]


def test_input_to_present_profile_window_returns_summary_on_interval() -> None:
    window = InputToPresentProfileWindow(interval_s=0.25)

    assert (
        window.record(
            present_time=1.0,
            input_sample_time=0.9,
            frame_index=0,
            frame_interval_s=0.1,
        )
        is None
    )
    summary = window.record(
        present_time=1.3,
        input_sample_time=0.9,
        frame_index=1,
        frame_interval_s=0.1,
    )

    assert summary is not None
    assert summary.samples == 2
    assert summary.avg_raw_control_to_present_ms == pytest.approx(250.0)
    assert summary.avg_adj_control_to_present_ms == pytest.approx(200.0)
    assert "avg_raw_control_to_present_ms=250.00" in summary.log_message()
