# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input events shared by the model and UI loops."""

import threading

from flashdreams.runtime_v2.user_input_event import (
    ResetUserInputEvent,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class EventBuffer:
    """Keep input events until every loop has read them.

    Input is collected once, on the thread running the UI, but both loops need it
    and they read at different rates. So this holds a flat list of events plus a
    cursor per registered reader, hands each reader only what it has not seen,
    and drops the prefix they have all passed.

    It also counts resets. Every :class:`ResetUserInputEventData` appended bumps
    :attr:`generation`, which the loops and the presentation manager compare
    against their own; that counter is how a reset reaches all of them without
    any of them talking to each other.
    """

    def __init__(self) -> None:
        self._events: list[UserInputEvent] = []
        self._base_index = 0
        self._reader_indexes: dict[int, int] = {}
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def generation(self) -> int:
        """Return the current reset generation."""
        with self._lock:
            return self._generation

    def register(self, reader_id: int) -> None:
        """Register a reader before input collection begins."""
        with self._lock:
            self._reader_indexes.setdefault(reader_id, self._base_index)

    def append(self, events: UserInputEvents) -> None:
        """Add a batch of client input events."""
        received = events.get_events()
        with self._lock:
            self._events.extend(received)
            self._generation += sum(
                isinstance(event, ResetUserInputEvent) for event in received
            )

    def read(self, reader_id: int) -> tuple[UserInputEvents, int]:
        """Return unread events and advance ``reader_id`` to the buffer end."""
        with self._lock:
            event_index = self._reader_indexes.setdefault(reader_id, self._base_index)
            relative_index = event_index - self._base_index
            events = list(self._events[relative_index:])
            self._reader_indexes[reader_id] = self._base_index + len(self._events)
            return UserInputEvents(events), self._generation

    def unregister(self, reader_id: int) -> None:
        """Stop retaining events for ``reader_id``."""
        with self._lock:
            self._reader_indexes.pop(reader_id, None)

    def collect_garbage(self) -> int:
        """Delete events read by every registered reader."""
        with self._lock:
            if not self._reader_indexes:
                removed = len(self._events)
            else:
                removed = min(self._reader_indexes.values()) - self._base_index
            if removed <= 0:
                return 0
            del self._events[:removed]
            self._base_index += removed
            return removed

    def clear(self) -> None:
        """Remove all retained events and readers."""
        with self._lock:
            self._events.clear()
            self._reader_indexes.clear()
            self._base_index = 0


__all__ = ["EventBuffer"]
