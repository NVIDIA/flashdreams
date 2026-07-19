# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral realtime presentation queue and pacing helpers."""

from __future__ import annotations

import queue
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np

from flashdreams.serving.realtime.media import rgb_frame_to_uint8

FrameT = TypeVar("FrameT")


class SupportsGetNowait(Protocol[FrameT]):
    def get_nowait(self) -> FrameT: ...


@dataclass(frozen=True, slots=True)
class QueueDrainResult(Generic[FrameT]):
    """Summary of frames drained from a producer queue into a presentation queue."""

    pulled: int
    accepted: int
    skipped: int
    dropped: tuple[FrameT, ...]


@dataclass(frozen=True, slots=True)
class PresentationWait:
    """Sleep interval consumed while waiting for the next presentation slot."""

    begin_time: float
    end_time: float

    @property
    def duration_s(self) -> float:
        return self.end_time - self.begin_time


class PresentationQueue(Generic[FrameT]):
    """Small FIFO queue for frames ready to present.

    ``capacity`` bounds queued frames by dropping the oldest ready frame when a
    newer frame arrives. That policy keeps realtime outputs moving forward
    under backpressure instead of building unbounded latency.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self._capacity = capacity
        self._frames: deque[FrameT] = deque()

    @property
    def capacity(self) -> int | None:
        return self._capacity

    def __len__(self) -> int:
        return len(self._frames)

    def __bool__(self) -> bool:
        return bool(self._frames)

    def __iter__(self) -> Iterator[FrameT]:
        return iter(self._frames)

    def append(self, frame: FrameT) -> FrameT | None:
        """Append ``frame`` and return any frame dropped by capacity pressure."""
        dropped = None
        if self._capacity is not None and len(self._frames) >= self._capacity:
            dropped = self._frames.popleft()
        self._frames.append(frame)
        return dropped

    def pop_ready(self) -> FrameT | None:
        """Return the oldest ready frame, or ``None`` if no frame is ready."""
        if not self._frames:
            return None
        return self._frames.popleft()

    def popleft(self) -> FrameT:
        """Return the oldest ready frame, raising ``IndexError`` if empty."""
        return self._frames.popleft()

    def clear(self) -> int:
        """Drop all queued frames and return the number removed."""
        count = len(self._frames)
        self._frames.clear()
        return count

    def drain_nowait(
        self,
        source: SupportsGetNowait[FrameT],
        *,
        include: Callable[[FrameT], bool] | None = None,
        prepare: Callable[[FrameT], None] | None = None,
    ) -> QueueDrainResult[FrameT]:
        """Drain currently available frames from ``source`` without blocking."""
        pulled = 0
        accepted = 0
        skipped = 0
        accepted_frames: list[FrameT] = []
        while True:
            try:
                frame = source.get_nowait()
            except queue.Empty:
                break
            pulled += 1
            if include is not None and not include(frame):
                skipped += 1
                continue
            accepted_frames.append(frame)
            accepted += 1

        if self._capacity is None:
            retained_entries = [(False, frame) for frame in self._frames] + [
                (True, frame) for frame in accepted_frames
            ]
            dropped: list[FrameT] = []
        else:
            entries = [(False, frame) for frame in self._frames] + [
                (True, frame) for frame in accepted_frames
            ]
            overflow = max(0, len(entries) - self._capacity)
            dropped = [frame for _, frame in entries[:overflow]]
            retained_entries = entries[overflow:]

        if prepare is not None:
            for is_new, frame in retained_entries:
                if is_new:
                    prepare(frame)
        self._frames.clear()
        self._frames.extend(frame for _, frame in retained_entries)

        return QueueDrainResult(
            pulled=pulled,
            accepted=accepted,
            skipped=skipped,
            dropped=tuple(dropped),
        )


def wait_until_present_time(
    present_time: float,
    *,
    poll_timeout_s: float,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> PresentationWait | None:
    """Sleep until ``present_time`` is due, capped by ``poll_timeout_s``.

    Returns the consumed sleep interval for tracing, or ``None`` when the
    target presentation time is already due.
    """
    if poll_timeout_s < 0:
        raise ValueError("poll_timeout_s must be >= 0")
    now = clock()
    if now >= present_time:
        return None
    begin_time = now
    sleep(min(poll_timeout_s, max(0.0, present_time - now)))
    return PresentationWait(begin_time=begin_time, end_time=clock())


def materialize_rgb_host_uint8(frame: object) -> np.ndarray:
    """Materialize a frame-like object to contiguous ``(H, W, 3)`` uint8 RGB."""
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    return rgb_frame_to_uint8(array, value_range="uint8")
