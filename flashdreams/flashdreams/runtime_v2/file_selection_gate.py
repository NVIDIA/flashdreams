# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-at-a-time client file-selector queue shared by every window."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, replace

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueuedFileSelection:
    """One file-selector request waiting to occupy the client dialog."""

    request_id: str
    """Stable selector-slot id for the later selected-files input event."""

    initial_path: str | None
    """Directory hint for the selector, or ``None`` for the window default."""

    accept: tuple[str, ...]
    """Filename suffixes the client may choose, empty for any type."""

    max_bytes: int | None
    """Requested size budget in bytes, or ``None`` for the window ceiling."""

    generation: int = 0
    """Client generation that admitted this request. Stale pickers do not match."""


class FileSelectionGate:
    """Admit one client file selector at a time.

    A duplicate ``request_id`` that is already active or queued is ignored.
    A different id waits until :meth:`complete` for the active request.
    :meth:`drain` finishes the current client generation so a late picker
    cannot complete a later request that reused the id.
    """

    def __init__(self) -> None:
        """Create an empty gate."""
        self._lock = threading.Lock()
        self._active: QueuedFileSelection | None = None
        self._queued: deque[QueuedFileSelection] = deque()
        self._generation = 0

    def submit(self, pending: QueuedFileSelection) -> QueuedFileSelection | None:
        """Record ``pending`` and return it when it should start now.

        Args:
            pending: Request the window wants to show.

        Returns:
            ``pending`` when no selector is active, otherwise ``None`` when the
            id is a duplicate or the request was queued behind the active one.
        """
        with self._lock:
            if self._contains(pending.request_id):
                _LOGGER.warning(
                    "Ignoring duplicate file-selection request id %r.",
                    pending.request_id,
                )
                return None
            stamped = replace(pending, generation=self._generation)
            if self._active is None:
                self._active = stamped
                return stamped
            self._queued.append(stamped)
            return None

    def complete(
        self, request_id: str, generation: int | None = None
    ) -> QueuedFileSelection | None:
        """Release ``request_id`` and return the next request to start.

        Args:
            request_id: Id of the selector that just emitted an event.
            generation: Client generation of that selector, or ``None`` to
                match only the id.

        Returns:
            The next queued request, or ``None`` when this id was not active,
            the generation does not match, or the queue is empty.
        """
        with self._lock:
            if self._active is None or self._active.request_id != request_id:
                return None
            if generation is not None and self._active.generation != generation:
                return None
            self._active = None
            if not self._queued:
                return None
            nxt = self._queued.popleft()
            self._active = nxt
            return nxt

    def is_current(self, request_id: str, generation: int) -> bool:
        """Return whether ``request_id`` is still the active selector for ``generation``."""
        with self._lock:
            return (
                self._active is not None
                and self._active.request_id == request_id
                and self._active.generation == generation
            )

    def drain(self) -> tuple[QueuedFileSelection, ...]:
        """Return and drop remaining requests, then start a new client generation."""
        with self._lock:
            items: list[QueuedFileSelection] = []
            if self._active is not None:
                items.append(self._active)
            items.extend(self._queued)
            self._active = None
            self._queued.clear()
            self._generation += 1
            return tuple(items)

    def clear(self) -> None:
        """Drop remaining requests and start a new client generation."""
        self.drain()

    def _contains(self, request_id: str) -> bool:
        if self._active is not None and self._active.request_id == request_id:
            return True
        return any(item.request_id == request_id for item in self._queued)
