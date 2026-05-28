# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Timing records for latency measurement.

FrameTimes and ChunkTimes are intentionally mutable (not frozen). They travel
through the pipeline as the same object instance, accumulating timestamps at
each stage. This preserves object identity as required by the timestamped
pipeline design: the same ChunkTimes that was created at request time is the
same object that records the present time, enabling direct correlation without
copying or re-association.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class FrameTimes:
    frame_index: int
    intended_present_time: float
    image_ready_time: float | None = None
    sample_display_pose_time: float | None = None
    present_time: float | None = None

    def is_complete(self) -> bool:
        return self.present_time is not None


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
    prediction: ChunkPrediction
    frames: list[FrameTimes]
    chunk_render_start_time: float | None = None
    chunk_ready_time: float | None = None

    @classmethod
    def create(
        cls,
        chunk_index: int,
        input_sample_time: float,
        request_time: float,
        request_poses_ready_time: float,
        prediction: ChunkPrediction,
        intended_present_times: list[float],
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
            prediction=prediction,
            frames=frames,
        )


class ChunkHistory:
    def __init__(self, capacity: int) -> None:
        self._deque: deque[ChunkTimes] = deque(maxlen=capacity)

    @classmethod
    def create(cls, capacity: int) -> "ChunkHistory":
        return cls(capacity)

    def append(self, chunk: ChunkTimes) -> None:
        self._deque.append(chunk)

    def latest(self) -> ChunkTimes:
        if not self._deque:
            raise RuntimeError("ChunkHistory is empty")
        return self._deque[-1]

    def recent(self, count: int) -> list[ChunkTimes]:
        items = list(self._deque)
        if count < len(items):
            return items[-count:]
        return items
