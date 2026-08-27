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

"""Correlation of user input events with generated output frames."""

import math
from bisect import bisect_right
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.step_result import InputEventTrace


@dataclass(frozen=True, slots=True)
class _PendingInputEventTrace:
    """Correlated input waiting for its first affected output frame."""

    effective_at_s: float | None
    """Output-clock time, or ``None`` for the next produced frame."""

    event_id: str
    """Browser-generated correlation ID."""


class InputEventTraceTracker:
    """Correlate tagged input events with frames across output chunks."""

    def __init__(self) -> None:
        """Create an empty tracker."""
        self._pending: deque[_PendingInputEventTrace] = deque()

    @property
    def pending_count(self) -> int:
        """Return the number of inputs awaiting an affected frame."""
        return len(self._pending)

    def track(
        self,
        event: UserInputEvent,
        *,
        effective_at_s: float | None,
    ) -> None:
        """Track a correlated event until an affected frame is available.

        Args:
            event: Input event carrying the browser correlation ID. Events
                without an ID are ignored.
            effective_at_s: Event time on the output-frame clock, or ``None``
                to acknowledge the next produced frame.

        Raises:
            ValueError: The correlation ID or effective time is invalid.
        """
        event_id = event.event_id
        if event_id is None:
            return
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event.event_id must be a non-empty string.")
        if effective_at_s is not None:
            effective_at_s = _finite_nonnegative_time(
                effective_at_s,
                name="effective_at_s",
            )
        self._pending.append(
            _PendingInputEventTrace(
                effective_at_s=effective_at_s,
                event_id=event_id,
            )
        )

    def resolve(
        self,
        frame_times_s: Sequence[float],
    ) -> tuple[InputEventTrace, ...]:
        """Resolve inputs to the first frame strictly after each event.

        Inputs later than this chunk remain pending for a future call. Inputs
        tracked without an effective time resolve to frame zero.

        Args:
            frame_times_s: Strictly increasing, nonnegative output-frame times
                for one chunk.

        Returns:
            Traces resolved against this chunk, in input arrival order.

        Raises:
            ValueError: Frame times are negative, non-finite, or not strictly
                increasing.
        """
        frame_times = tuple(
            _finite_nonnegative_time(frame_time, name="frame_times_s")
            for frame_time in frame_times_s
        )
        if any(
            current <= previous
            for previous, current in zip(frame_times, frame_times[1:], strict=False)
        ):
            raise ValueError("frame_times_s must be strictly increasing.")

        resolved: list[InputEventTrace] = []
        carried: deque[_PendingInputEventTrace] = deque()
        while self._pending:
            pending = self._pending.popleft()
            frame_index = (
                0
                if pending.effective_at_s is None
                else bisect_right(frame_times, pending.effective_at_s)
            )
            if frame_index == len(frame_times):
                carried.append(pending)
                continue
            resolved.append(
                InputEventTrace(
                    event_id=pending.event_id,
                    frame_index=frame_index,
                )
            )
        self._pending.extend(carried)
        return tuple(resolved)

    def clear(self) -> None:
        """Discard all pending input traces."""
        self._pending.clear()


def _finite_nonnegative_time(value: float, *, name: str) -> float:
    """Return ``value`` as a finite nonnegative timestamp."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must contain only finite nonnegative timestamps.")
    return parsed


__all__ = ["InputEventTraceTracker"]
