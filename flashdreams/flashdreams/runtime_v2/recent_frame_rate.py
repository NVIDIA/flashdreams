# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recent frame-throughput measurement shared by runtime applications."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _FrameRateObservation:
    """One completed frame-producing operation."""

    completed_at: float
    """Completion timestamp on the tracker's monotonic clock."""

    frame_count: int
    """Number of frames produced by the operation."""

    elapsed_s: float
    """Operation wall time in seconds."""


@dataclass(frozen=True, slots=True)
class RecentFrameRateSnapshot:
    """Immutable recent frame-rate observations safe to share across threads."""

    _window_seconds: float
    """Trailing completion-time window width."""

    _observations: tuple[_FrameRateObservation, ...]
    """Observations retained when the snapshot was created."""

    @property
    def window_seconds(self) -> float:
        """Return the trailing window width in seconds."""
        return self._window_seconds

    def frames_per_second(self, now: float | None = None) -> float:
        """Return throughput for operations completed in the trailing window.

        Args:
            now: Sampling time on the observation clock. When omitted, use the
                latest completion time. It cannot precede the latest observed
                completion.

        Returns:
            Frames divided by elapsed operation time, or zero when the window
            contains no observations.
        """
        sampled_at = None if now is None else float(now)
        if sampled_at is not None and not math.isfinite(sampled_at):
            raise ValueError("now must be finite.")
        if not self._observations:
            return 0.0
        latest_completed_at = self._observations[-1].completed_at
        if sampled_at is None:
            sampled_at = latest_completed_at
        elif sampled_at < latest_completed_at:
            raise ValueError("now must not precede the latest observation.")
        cutoff = sampled_at - self._window_seconds
        recent = tuple(
            observation
            for observation in self._observations
            if observation.completed_at > cutoff
        )
        if not recent:
            return 0.0
        elapsed_s = sum(item.elapsed_s for item in recent)
        if elapsed_s == 0.0:
            return math.inf
        return sum(item.frame_count for item in recent) / elapsed_s


class RecentFrameRateTracker:
    """Track frame throughput over a trailing completion-time window."""

    def __init__(self, *, window_seconds: float) -> None:
        """Create an empty tracker.

        Args:
            window_seconds: Width of the trailing completion-time window.

        Raises:
            ValueError: ``window_seconds`` is not finite and positive.
        """
        window_seconds = float(window_seconds)
        if not math.isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError("window_seconds must be finite and > 0.")
        self._window_seconds = window_seconds
        self._observations: deque[_FrameRateObservation] = deque()

    def observe(
        self,
        *,
        completed_at: float,
        frame_count: int,
        elapsed_s: float,
    ) -> float:
        """Record one completed operation and return current throughput.

        Args:
            completed_at: Completion time on a monotonic clock.
            frame_count: Number of frames produced by the operation.
            elapsed_s: Operation wall time in seconds.

        Returns:
            Current trailing-window throughput in frames per second.

        Raises:
            TypeError: ``frame_count`` is not an integer.
            ValueError: A value is invalid or completion times move backward.
        """
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer.")
        if frame_count <= 0:
            raise ValueError("frame_count must be > 0.")
        completed_at = float(completed_at)
        elapsed_s = float(elapsed_s)
        if not math.isfinite(completed_at):
            raise ValueError("completed_at must be finite.")
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("elapsed_s must be finite and >= 0.")
        if self._observations and completed_at < self._observations[-1].completed_at:
            raise ValueError("completed_at must not precede the latest observation.")

        self._observations.append(
            _FrameRateObservation(
                completed_at=completed_at,
                frame_count=frame_count,
                elapsed_s=elapsed_s,
            )
        )
        cutoff = completed_at - self._window_seconds
        while self._observations and self._observations[0].completed_at <= cutoff:
            self._observations.popleft()
        return self.snapshot().frames_per_second()

    def reset(self) -> None:
        """Discard all observations."""
        self._observations.clear()

    def snapshot(self) -> RecentFrameRateSnapshot:
        """Return an immutable copy of the current observations."""
        return RecentFrameRateSnapshot(
            _window_seconds=self._window_seconds,
            _observations=tuple(self._observations),
        )


__all__ = ["RecentFrameRateSnapshot", "RecentFrameRateTracker"]
