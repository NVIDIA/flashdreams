# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-at-a-time client file-selector queue shared by every window."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass

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


class FileSelectionGate:
    """Admit one client file selector at a time.

    A duplicate ``request_id`` that is already active or queued is ignored.
    A different id waits until :meth:`complete` for the active request.
    """

    def __init__(self) -> None:
        """Create an empty gate."""
        self._lock = threading.Lock()
        self._active: QueuedFileSelection | None = None
        self._queued: deque[QueuedFileSelection] = deque()

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
            if self._active is None:
                self._active = pending
                return pending
            self._queued.append(pending)
            return None

    def complete(self, request_id: str) -> QueuedFileSelection | None:
        """Release ``request_id`` and return the next request to start.

        Args:
            request_id: Id of the selector that just emitted an event.

        Returns:
            The next queued request, or ``None`` when this id was not active or
            the queue is empty.
        """
        with self._lock:
            if self._active is None or self._active.request_id != request_id:
                return None
            self._active = None
            if not self._queued:
                return None
            nxt = self._queued.popleft()
            self._active = nxt
            return nxt

    def clear(self) -> None:
        """Drop the active request and every queued request."""
        with self._lock:
            self._active = None
            self._queued.clear()

    def _contains(self, request_id: str) -> bool:
        if self._active is not None and self._active.request_id == request_id:
            return True
        return any(item.request_id == request_id for item in self._queued)
