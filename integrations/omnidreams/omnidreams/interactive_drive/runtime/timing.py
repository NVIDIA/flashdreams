# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Timing records for latency measurement.

``FrameTimes`` / ``ChunkTimes`` are mutable on purpose: a single instance
travels through the pipeline accumulating per-stage timestamps, so the record
created at request time is the same one that records present time (direct
correlation, no copying or re-association).
"""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

TraceComponentValue = str | int | float | bool | None


def trace_time_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def event_dependencies(*events: int | None) -> list[int]:
    return [event for event in events if event is not None]


@dataclass
class FrameTimes:
    frame_index: int
    intended_present_time: float
    image_ready_time: float | None = None
    sample_display_pose_time: float | None = None
    present_time: float | None = None


@dataclass(frozen=True)
class ChunkPrediction:
    """Predicted timestamps for a chunk's pipeline stages.

    Stage 1 only predicts ``first_present`` (when the chunk's first frame
    will reach the screen). The Stage 2 design adds intermediate
    EMA-summed milestones (``request -> render_start -> chunk_ready ->
    decode_first -> first_present``) per ``alpasim.frame_timing``; new
    fields land here as named attributes when that work arrives.

    Use :meth:`create` rather than constructing directly: the prediction
    formula lives there so it stays beside the data it produces.
    """

    first_present: float

    @classmethod
    def create(
        cls, *, request_time: float, frame_interval_s: float
    ) -> "ChunkPrediction":
        """Stage 1 prediction: ``first_present = request_time + frame_interval_s``.

        Placeholder for the Stage 2 EMA-summed prediction chain; the
        signature stays the same so callers don't change when Stage 2
        lands and the body grows to consume EMA latency stats.
        """
        return cls(first_present=request_time + frame_interval_s)


@dataclass
class ChunkTimes:
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


class ChunkHistory:
    def __init__(self, capacity: int) -> None:
        self._deque: deque[ChunkTimes] = deque(maxlen=capacity)

    def append(self, chunk: ChunkTimes) -> None:
        self._deque.append(chunk)


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
