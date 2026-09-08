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

"""Realtime sampling windows shared by stateful user-input tracks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InputWindow:
    """One virtual-time interval and its requested sample locations."""

    start_s: float
    """Inclusive state boundary in session-relative seconds."""

    end_s: float
    """End boundary; transitions there carry into the following window."""

    sample_times_s: tuple[float, ...]
    """Strictly increasing sample times in ``(start_s, end_s]``.

    A sample at ``end_s`` observes state over the interval ending there. An
    input transition timestamped exactly at that boundary affects the next
    positive-duration interval, so state tracks consume it only as carried
    state for the following window.
    """

    def __post_init__(self) -> None:
        """Validate and normalize the public window contract."""
        start_s = _finite_time(self.start_s, name="start_s")
        end_s = _finite_time(self.end_s, name="end_s")
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s.")
        sample_times_s = tuple(
            _finite_time(value, name="sample_times_s") for value in self.sample_times_s
        )
        if not sample_times_s:
            raise ValueError("sample_times_s must not be empty.")

        previous_s = start_s
        for sample_s in sample_times_s:
            if sample_s <= previous_s:
                raise ValueError(
                    "sample_times_s must be strictly increasing after start_s."
                )
            if sample_s > end_s:
                raise ValueError("sample_times_s must not extend beyond end_s.")
            previous_s = sample_s

        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "end_s", end_s)
        object.__setattr__(self, "sample_times_s", sample_times_s)


class RealtimeInputTimeline:
    """Advance fixed-rate input windows on a session-relative clock.

    The owning model loop supplies timestamps only for newly accepted input.
    When those timestamps are newer than the nominal next window, the clock
    skips stale time while retaining up to one window of the new input batch.
    Input modalities remain responsible for deciding which events they accept.

    Instances are mutable and intended to be owned by one session loop. They do
    not provide internal locking and must not be shared with the UI loop.
    """

    def __init__(
        self,
        *,
        samples_per_second: float,
        start_s: float = 0.0,
    ) -> None:
        samples_per_second = float(samples_per_second)
        if not math.isfinite(samples_per_second) or samples_per_second <= 0.0:
            raise ValueError("samples_per_second must be finite and > 0.")
        self._samples_per_second = samples_per_second
        self._sample_interval_s = 1.0 / samples_per_second
        if not math.isfinite(self._sample_interval_s) or self._sample_interval_s <= 0.0:
            raise ValueError("samples_per_second is outside the supported range.")
        self._next_window_start_s = _finite_time(start_s, name="start_s")

    @property
    def samples_per_second(self) -> float:
        """Return the target input sampling rate."""
        return self._samples_per_second

    @property
    def sample_interval_s(self) -> float:
        """Return the interval between adjacent input samples."""
        return self._sample_interval_s

    @property
    def next_window_start_s(self) -> float:
        """Return the start of the next unsampled virtual-time window."""
        return self._next_window_start_s

    def next_window(
        self,
        sample_count: int,
        *,
        input_times_s: Iterable[float] = (),
    ) -> InputWindow:
        """Advance and return one sampling window.

        Args:
            sample_count: Number of evenly spaced samples in the window.
            input_times_s: Timestamps for newly accepted input. These may move
                the window forward, but never backward.

        Returns:
            The selected window and its sample locations.

        Raises:
            TypeError: ``sample_count`` is not an integer.
            ValueError: A count or timestamp is invalid.
        """
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise TypeError("sample_count must be an integer.")
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1.")

        input_times = tuple(
            _finite_time(value, name="input_times_s") for value in input_times_s
        )
        duration_s = sample_count * self._sample_interval_s
        if not math.isfinite(duration_s):
            raise ValueError("sample_count produces a non-finite window duration.")
        start_s = self._next_window_start_s
        nominal_end_s = start_s + duration_s
        if not math.isfinite(nominal_end_s):
            raise ValueError("sample_count produces a non-finite window end.")
        if input_times:
            earliest_input_s = min(input_times)
            latest_input_s = max(input_times)
            if latest_input_s > nominal_end_s:
                start_s = max(
                    start_s,
                    earliest_input_s,
                    latest_input_s - duration_s,
                )

        end_s = start_s + duration_s
        if not math.isfinite(end_s):
            raise ValueError("input_times_s produce a non-finite window end.")
        sample_times_s = tuple(
            start_s + (index + 1) * self._sample_interval_s
            for index in range(sample_count)
        )
        self._next_window_start_s = end_s
        return InputWindow(
            start_s=start_s,
            end_s=end_s,
            sample_times_s=sample_times_s,
        )

    def reset(self, *, start_s: float = 0.0) -> None:
        """Restart the virtual clock at ``start_s``."""
        self._next_window_start_s = _finite_time(start_s, name="start_s")


def _finite_time(value: float, *, name: str) -> float:
    """Return ``value`` as a finite nonnegative timestamp."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must contain only finite nonnegative timestamps.")
    return parsed


__all__ = ["InputWindow", "RealtimeInputTimeline"]
