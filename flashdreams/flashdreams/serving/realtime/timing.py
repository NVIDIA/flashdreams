# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral realtime timing records and trace helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

TraceComponentValue = str | int | float | bool | None


def trace_time_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def event_dependencies(*events: int | None) -> list[int]:
    return [event for event in events if event is not None]


def _duration_ms(start_time: float | None, end_time: float | None) -> float | None:
    if start_time is None or end_time is None:
        return None
    return (end_time - start_time) * 1000.0


@dataclass
class FrameTimes:
    frame_index: int
    intended_present_time: float
    image_ready_time: float | None = None
    sample_display_pose_time: float | None = None
    present_time: float | None = None

    def input_to_present_ms(self, *, input_sample_time: float) -> float | None:
        return _duration_ms(input_sample_time, self.present_time)

    def present_jitter_ms(self) -> float | None:
        return _duration_ms(self.intended_present_time, self.present_time)


@dataclass(frozen=True)
class ChunkPrediction:
    """Predicted timestamps for a chunk's pipeline stages."""

    first_present: float

    @classmethod
    def create(
        cls, *, request_time: float, frame_interval_s: float
    ) -> "ChunkPrediction":
        return cls(first_present=request_time + frame_interval_s)


@dataclass
class ChunkTimes:
    """Mutable timing record that travels with one realtime chunk."""

    chunk_index: int
    input_sample_time: float
    request_time: float
    request_poses_ready_time: float
    frames: list[FrameTimes]
    prediction: ChunkPrediction | None = None
    chunk_render_start_time: float | None = None
    chunk_ready_time: float | None = None

    @classmethod
    def create(
        cls,
        chunk_index: int,
        input_sample_time: float,
        request_time: float,
        request_poses_ready_time: float,
        intended_present_times: list[float],
        prediction: ChunkPrediction | None = None,
    ) -> "ChunkTimes":
        frames = [
            FrameTimes(frame_index=index, intended_present_time=time_value)
            for index, time_value in enumerate(intended_present_times)
        ]
        return cls(
            chunk_index=chunk_index,
            input_sample_time=input_sample_time,
            request_time=request_time,
            request_poses_ready_time=request_poses_ready_time,
            frames=frames,
            prediction=prediction,
        )

    def stage_durations_ms(self) -> dict[str, float]:
        """Return available milestone-derived durations for this chunk."""
        return chunk_stage_durations_ms(self)


@dataclass(frozen=True)
class VideoModelTimings:
    """Observable backend timing milestones for one video-model chunk."""

    condition_start_time: float
    condition_ready_time: float
    model_start_time: float
    model_ready_time: float
    merge_start_time: float
    merge_ready_time: float
    decode_start_time: float | None = None
    decode_ready_time: float | None = None
    cache_update_start_time: float | None = None
    cache_update_ready_time: float | None = None

    def stage_durations_ms(self) -> dict[str, float]:
        durations: dict[str, float] = {}
        _add_duration(
            durations,
            "condition",
            self.condition_start_time,
            self.condition_ready_time,
        )
        _add_duration(durations, "model", self.model_start_time, self.model_ready_time)
        _add_duration(
            durations,
            "cache_update",
            self.cache_update_start_time,
            self.cache_update_ready_time,
        )
        _add_duration(
            durations,
            "decode",
            self.decode_start_time,
            self.decode_ready_time,
        )
        _add_duration(durations, "merge", self.merge_start_time, self.merge_ready_time)
        _add_duration(
            durations, "total", self.condition_start_time, self.merge_ready_time
        )
        return durations


class ChunkHistory:
    def __init__(self, capacity: int) -> None:
        self._deque: deque[ChunkTimes] = deque(maxlen=capacity)

    def append(self, chunk: ChunkTimes) -> None:
        self._deque.append(chunk)

    def __iter__(self) -> Iterator[ChunkTimes]:
        return iter(self._deque)

    def __len__(self) -> int:
        return len(self._deque)

    def summary(self) -> "RecentTimingSummary":
        return summarize_chunk_history(self._deque)


class TraceSink(Protocol):
    def add_thread(self, name: str) -> int: ...

    def add_instant(
        self,
        name: str,
        *,
        thread: int,
        time_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int: ...

    def add_range(
        self,
        name: str,
        *,
        thread: int,
        begin_ns: int,
        end_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int: ...


@dataclass(frozen=True)
class TraceContext:
    sink: TraceSink
    main_thread: int
    worker_thread: int
    lock: Lock

    @classmethod
    def create(cls, sink: TraceSink) -> "TraceContext":
        return cls(
            sink=sink,
            main_thread=sink.add_thread("main"),
            worker_thread=sink.add_thread("pipeline-worker"),
            lock=Lock(),
        )

    def add_instant(
        self,
        name: str,
        *,
        thread: int,
        time_ns: int,
        depends_on: list[int] | None = None,
        **components: TraceComponentValue,
    ) -> int:
        with self.lock:
            return self.sink.add_instant(
                name,
                thread=thread,
                time_ns=time_ns,
                depends_on=depends_on,
                **components,
            )

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
        with self.lock:
            return self.sink.add_range(
                name,
                thread=thread,
                begin_ns=begin_ns,
                end_ns=end_ns,
                depends_on=depends_on,
                **components,
            )


@dataclass(frozen=True)
class VideoModelTraceEvents:
    condition_event_id: int
    model_event_id: int
    merge_event_id: int
    cache_update_event_id: int | None = None
    decode_event_id: int | None = None

    @property
    def final_event_id(self) -> int:
        return self.merge_event_id


def emit_video_model_timing_ranges(
    trace_context: TraceContext,
    *,
    timings: VideoModelTimings,
    thread: int,
    depends_on: list[int] | None = None,
    chunk_index: int,
) -> VideoModelTraceEvents:
    """Emit standard trace ranges for backend-visible video-model stages."""
    condition_event = trace_context.add_range(
        "condition_raster",
        thread=thread,
        begin_ns=trace_time_ns(timings.condition_start_time),
        end_ns=trace_time_ns(timings.condition_ready_time),
        depends_on=depends_on,
        chunk_index=chunk_index,
    )
    model_event = trace_context.add_range(
        "model_generate",
        thread=thread,
        begin_ns=trace_time_ns(timings.model_start_time),
        end_ns=trace_time_ns(timings.model_ready_time),
        depends_on=event_dependencies(condition_event),
        chunk_index=chunk_index,
    )
    cache_update_event = _add_optional_trace_range(
        trace_context,
        "cache_update",
        thread=thread,
        begin_time=timings.cache_update_start_time,
        end_time=timings.cache_update_ready_time,
        depends_on=event_dependencies(model_event),
        chunk_index=chunk_index,
    )
    decode_event = _add_optional_trace_range(
        trace_context,
        "decode",
        thread=thread,
        begin_time=timings.decode_start_time,
        end_time=timings.decode_ready_time,
        depends_on=event_dependencies(
            cache_update_event if cache_update_event is not None else model_event
        ),
        chunk_index=chunk_index,
    )
    last_event = decode_event
    if last_event is None:
        last_event = cache_update_event
    if last_event is None:
        last_event = model_event
    merge_event = trace_context.add_range(
        "frame_merge",
        thread=thread,
        begin_ns=trace_time_ns(timings.merge_start_time),
        end_ns=trace_time_ns(timings.merge_ready_time),
        depends_on=event_dependencies(last_event),
        chunk_index=chunk_index,
    )
    return VideoModelTraceEvents(
        condition_event_id=condition_event,
        model_event_id=model_event,
        cache_update_event_id=cache_update_event,
        decode_event_id=decode_event,
        merge_event_id=merge_event,
    )


def chunk_stage_durations_ms(chunk: ChunkTimes) -> dict[str, float]:
    durations: dict[str, float] = {}
    _add_duration(
        durations, "input_to_request", chunk.input_sample_time, chunk.request_time
    )
    _add_duration(
        durations,
        "request_to_poses_ready",
        chunk.request_time,
        chunk.request_poses_ready_time,
    )
    _add_duration(
        durations,
        "queue_wait",
        chunk.request_poses_ready_time,
        chunk.chunk_render_start_time,
    )
    _add_duration(
        durations,
        "chunk_render",
        chunk.chunk_render_start_time,
        chunk.chunk_ready_time,
    )
    if chunk.frames:
        first_frame = chunk.frames[0]
        _add_duration(
            durations,
            "chunk_ready_to_first_image",
            chunk.chunk_ready_time,
            first_frame.image_ready_time,
        )
        _add_duration(
            durations,
            "first_image_to_present",
            first_frame.image_ready_time,
            first_frame.present_time,
        )
        _add_duration(
            durations,
            "input_to_first_present",
            chunk.input_sample_time,
            first_frame.present_time,
        )
    return durations


@dataclass(frozen=True)
class StageDurationSummary:
    count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p90_ms: float


@dataclass(frozen=True)
class RecentTimingSummary:
    chunk_count: int
    stages: dict[str, StageDurationSummary]


def summarize_stage_durations(
    samples: Iterable[Mapping[str, float]],
) -> dict[str, StageDurationSummary]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for stage_name, duration_ms in sample.items():
            grouped[stage_name].append(float(duration_ms))
    return {
        stage_name: _summarize_values(values)
        for stage_name, values in sorted(grouped.items())
    }


def summarize_chunk_history(chunks: Iterable[ChunkTimes]) -> RecentTimingSummary:
    chunk_list = list(chunks)
    return RecentTimingSummary(
        chunk_count=len(chunk_list),
        stages=summarize_stage_durations(
            chunk.stage_durations_ms() for chunk in chunk_list
        ),
    )


class RollingChunkTimingSummary:
    def __init__(self, capacity: int) -> None:
        self._chunks: deque[ChunkTimes] = deque(maxlen=capacity)

    def append(self, chunk: ChunkTimes) -> None:
        self._chunks.append(chunk)

    def reset(self) -> None:
        self._chunks.clear()

    def summary(self) -> RecentTimingSummary:
        return summarize_chunk_history(self._chunks)


@dataclass(frozen=True)
class InputToPresentSummary:
    window_s: float
    samples: int
    wall_present_fps: float
    avg_raw_control_to_present_ms: float
    avg_adj_control_to_present_ms: float

    def log_message(self) -> str:
        return (
            "[profile] e2e "
            f"wall_present_fps={self.wall_present_fps:.1f} "
            f"avg_adj_control_to_present_ms={self.avg_adj_control_to_present_ms:.2f} "
            f"avg_raw_control_to_present_ms={self.avg_raw_control_to_present_ms:.2f} "
            f"samples={self.samples}"
        )


class InputToPresentProfileWindow:
    """Rolling wall-clock input-to-present summary window."""

    def __init__(self, *, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.reset()

    def reset(self, *, interval_s: float | None = None) -> None:
        if interval_s is not None:
            self.interval_s = interval_s
        self._sum_raw_ms = 0.0
        self._sum_adj_ms = 0.0
        self._count = 0
        self._window_start: float | None = None

    def record(
        self,
        *,
        present_time: float,
        input_sample_time: float,
        frame_index: int,
        frame_interval_s: float,
    ) -> InputToPresentSummary | None:
        raw_ms = (present_time - input_sample_time) * 1000.0
        scheduled_ms = frame_index * (frame_interval_s * 1000.0)
        adj_ms = raw_ms - scheduled_ms
        self._sum_raw_ms += raw_ms
        self._sum_adj_ms += adj_ms
        self._count += 1
        if self._window_start is None:
            self._window_start = present_time

        window_s = present_time - self._window_start
        if window_s < self.interval_s:
            return None

        samples = self._count
        summary = InputToPresentSummary(
            window_s=window_s,
            samples=samples,
            wall_present_fps=float(samples) / window_s if window_s > 1e-9 else 0.0,
            avg_raw_control_to_present_ms=self._sum_raw_ms / float(samples),
            avg_adj_control_to_present_ms=self._sum_adj_ms / float(samples),
        )
        self._sum_raw_ms = 0.0
        self._sum_adj_ms = 0.0
        self._count = 0
        self._window_start = present_time
        return summary


def _add_duration(
    durations: dict[str, float],
    name: str,
    start_time: float | None,
    end_time: float | None,
) -> None:
    duration = _duration_ms(start_time, end_time)
    if duration is not None:
        durations[name] = duration


def _add_optional_trace_range(
    trace_context: TraceContext,
    name: str,
    *,
    thread: int,
    begin_time: float | None,
    end_time: float | None,
    depends_on: list[int] | None,
    chunk_index: int,
) -> int | None:
    if begin_time is None or end_time is None:
        return None
    return trace_context.add_range(
        name,
        thread=thread,
        begin_ns=trace_time_ns(begin_time),
        end_ns=trace_time_ns(end_time),
        depends_on=depends_on,
        chunk_index=chunk_index,
    )


def _summarize_values(values: list[float]) -> StageDurationSummary:
    ordered = sorted(values)
    count = len(ordered)
    if count <= 0:
        raise ValueError("Cannot summarize an empty stage duration list.")
    return StageDurationSummary(
        count=count,
        avg_ms=sum(ordered) / float(count),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        median_ms=_percentile_sorted(ordered, 0.5),
        p90_ms=_percentile_sorted(ordered, 0.9),
    )


def _percentile_sorted(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")
    if len(values) == 1:
        return values[0]
    clamped = min(1.0, max(0.0, percentile))
    index = clamped * (len(values) - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = index - lower_index
    return values[lower_index] * (1.0 - fraction) + values[upper_index] * fraction
