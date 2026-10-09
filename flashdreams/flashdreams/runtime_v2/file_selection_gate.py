# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-at-a-time client file-selector queue shared by every window."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, replace

from flashdreams.runtime_v2.selected_file import FileSelectionRequest

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueuedFileSelection:
    """One file-selector request waiting to become active."""

    request_id: str
    """Stable id for which button this request is for."""

    initial_path: str | None
    """Directory hint for the selector, or ``None`` for the window default."""

    accept: tuple[str, ...]
    """Filename suffixes the client may choose, empty for any type."""

    max_bytes: int | None
    """Requested size budget in bytes, or ``None`` for the window ceiling."""

    multiple: bool = False
    """Whether the selector may return more than one file."""

    generation: int = 0
    """Client generation that admitted this request. Stale pickers do not match."""

    @classmethod
    def from_request(cls, request: FileSelectionRequest) -> QueuedFileSelection:
        """Build a queued selection from a UI-loop request."""
        return cls(
            request_id=request.request_id,
            initial_path=request.initial_path,
            accept=request.accept,
            max_bytes=request.max_bytes,
            multiple=request.multiple,
        )


class FileSelectionGate:
    """Admit one client file selector at a time.

    A duplicate ``request_id`` that is already active or queued is ignored.
    A different id waits until :meth:`complete` for the active request.
    Callers emit a selected-files event only after :meth:`complete` claims
    the slot. :meth:`drain` finishes the current client generation so late
    in-flight work cannot complete a later request that reused the id.
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
    ) -> tuple[bool, QueuedFileSelection | None]:
        """Claim ``request_id`` if it is still current and return the next start.

        Callers emit a result only when this returns ``claimed=True``. Client-gone
        :meth:`drain` wins the race against in-flight work.

        Args:
            request_id: Id of the in-flight selector.
            generation: Client generation of that selector, or ``None`` to
                match only the id.

        Returns:
            ``(True, next)`` when this in-flight work still owned the slot.
            ``next`` is the request that should start now, or ``None`` if the
            queue is empty. ``(False, None)`` when the id is not current.
        """
        with self._lock:
            if self._active is None or self._active.request_id != request_id:
                return False, None
            if generation is not None and self._active.generation != generation:
                return False, None
            self._active = None
            if not self._queued:
                return True, None
            nxt = self._queued.popleft()
            self._active = nxt
            return True, nxt

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
