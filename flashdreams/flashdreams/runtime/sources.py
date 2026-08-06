# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-input sources the standard loop pulls per step, for replay and live runs."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime.inputs import TimeWindow, UserInputEvent, UserInputs


@runtime_checkable
class UserInputSource(Protocol):
    """Something the standard loop can slice a user-input window out of.

    Deliberately the same method name a :class:`UserInputs` batch already
    exposes, so a fully-known replay trace satisfies this protocol as-is and
    only live producers need a dedicated implementation.
    """

    def window(self, time_window: TimeWindow) -> UserInputs:
        """Return the events falling inside ``time_window``."""
        ...


class QueuedUserInputSource:
    """Live event queue fed by a transport and drained by the standard loop.

    Producers append timestamped events as they arrive; the loop asks for one
    window per step. Events older than a requested window are discarded when
    that window is taken, which bounds the queue for arbitrarily long
    sessions -- so windows must be requested in non-decreasing order, as the
    standard loop does.

    Safe to append from a transport thread while the loop reads.
    """

    def __init__(
        self,
        *,
        snapshot: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._events: deque[UserInputEvent] = deque()
        self._snapshot: Mapping[str, Any] = dict(snapshot or {})
        self._metadata: Mapping[str, Any] = dict(metadata or {})
        self._latest_timestamp_s = -math.inf
        self._lock = threading.Lock()

    def append(self, event: UserInputEvent) -> None:
        """Queue one event.

        Raises:
            ValueError: ``event`` predates an already-queued event. Producers
                should stamp from a single monotonic clock; an out-of-order
                arrival would otherwise silently reorder control state.
        """
        with self._lock:
            if event.timestamp_s < self._latest_timestamp_s:
                raise ValueError(
                    "QueuedUserInputSource events must arrive in non-decreasing "
                    f"timestamp order; got {event.timestamp_s} after "
                    f"{self._latest_timestamp_s}."
                )
            self._events.append(event)
            self._latest_timestamp_s = event.timestamp_s

    def extend(self, events: Iterable[UserInputEvent]) -> None:
        """Queue several events in order."""
        for event in events:
            self.append(event)

    def set_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Replace the derived snapshot returned alongside future windows."""
        with self._lock:
            self._snapshot = dict(snapshot)

    def reset(self) -> None:
        """Drop queued events and the ordering watermark for a new rollout."""
        with self._lock:
            self._events.clear()
            self._latest_timestamp_s = -math.inf

    def window(self, time_window: TimeWindow) -> UserInputs:
        """Return events inside ``time_window``, discarding anything older."""
        with self._lock:
            while self._events and self._events[0].timestamp_s < time_window.start_s:
                self._events.popleft()
            events = tuple(
                event
                for event in self._events
                if time_window.contains(event.timestamp_s)
            )
            snapshot = self._snapshot
            metadata = self._metadata
        return UserInputs(events=events, snapshot=snapshot, metadata=metadata)

    @property
    def pending_count(self) -> int:
        """Number of events currently retained, for tests and diagnostics."""
        with self._lock:
            return len(self._events)


__all__ = ["QueuedUserInputSource", "UserInputSource"]
