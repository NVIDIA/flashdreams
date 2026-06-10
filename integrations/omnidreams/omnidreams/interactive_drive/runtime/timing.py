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


@dataclass
class FrameTimes:
    frame_index: int
    intended_present_time: float
    image_ready_time: float | None = None
    sample_display_pose_time: float | None = None
    present_time: float | None = None


@dataclass
class ChunkTimes:
    chunk_index: int
    input_sample_time: float
    request_time: float
    request_poses_ready_time: float
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
            frames=frames,
        )


class ChunkHistory:
    def __init__(self, capacity: int) -> None:
        self._deque: deque[ChunkTimes] = deque(maxlen=capacity)

    def append(self, chunk: ChunkTimes) -> None:
        self._deque.append(chunk)
