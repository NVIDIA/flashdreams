# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-at-a-time client file-selector queue shared by every window."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

from flashdreams.runtime_v2.selected_file import FileSelectionRequest

_LOGGER = logging.getLogger(__name__)

FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS = 30.0
"""Seconds to wait for the host to show a picker before answering ``UNAVAILABLE``.

The deadline starts when a request is handed to the host. After the host reports
that the picker is presented, the user may take as long as they want. The
deadline does not close a picker that is already open.
"""


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


@dataclass(frozen=True, slots=True)
class FileSelectionAdmission:
    """What the window should do with one submitted file-selector request."""

    started: QueuedFileSelection | None = None
    """Request to hand to the host now, or ``None`` when it is not starting."""

    unavailable: bool = False
    """Answer this submit with ``UNAVAILABLE`` and do not open another picker."""


class FileSelectionGate:
    """Run one file picker at a time; later requests wait in a queue.

    - A repeated ``request_id`` is ignored.
    - :meth:`complete` frees the slot and says whether to deliver the result.
    - :meth:`expire_presentation` answers a picker that never appeared, but
      keeps the slot until the host finishes, so two pickers never overlap.
    - :meth:`drain` drops everything, so late results cannot match new requests.
    """

    def __init__(self) -> None:
        """Create an empty gate."""
        self._lock = threading.Lock()
        self._active: QueuedFileSelection | None = None
        self._queued: deque[QueuedFileSelection] = deque()
        self._generation = 0
        self._presented = False
        self._answered = False

    def submit(self, pending: QueuedFileSelection) -> FileSelectionAdmission:
        """Record ``pending`` and return whether it should start now.

        Args:
            pending: Request the window wants to show.

        Returns:
            An admission whose ``started`` request is handed to the host when
            no selector is active. ``unavailable`` is set when a previous
            picker was already answered but the host has not finished.
            Otherwise the request was queued or ignored as a duplicate.
        """
        with self._lock:
            if self._contains(pending.request_id):
                _LOGGER.warning(
                    "Ignoring duplicate file-selection request id %r.",
                    pending.request_id,
                )
                return FileSelectionAdmission()
            if self._active is not None and self._answered:
                return FileSelectionAdmission(unavailable=True)
            stamped = replace(pending, generation=self._generation)
            if self._active is None:
                self._active = stamped
                self._presented = False
                self._answered = False
                return FileSelectionAdmission(started=stamped)
            self._queued.append(stamped)
            return FileSelectionAdmission()

    def mark_presented(self, request_id: str, generation: int) -> bool:
        """Record that the host showed ``request_id``.

        Returns:
            ``True`` when this request is still waiting for that signal.
        """
        with self._lock:
            if not self._is_active(request_id, generation) or self._answered:
                return False
            self._presented = True
            return True

    def may_show(self, request_id: str, generation: int) -> bool:
        """Return whether the host should still be asked to show this request."""
        with self._lock:
            return self._is_active(request_id, generation) and not self._answered

    def expire_presentation(
        self, pending: QueuedFileSelection
    ) -> tuple[QueuedFileSelection, ...]:
        """Answer an unpresented request and every request waiting behind it.

        The active slot stays occupied until :meth:`complete` so the window
        does not open another picker while this host might still show one.

        Args:
            pending: The admitted request object, so a later pick that reuses
                its id and generation does not match.

        Returns:
            Requests the window should answer as ``UNAVAILABLE``. Empty when
            the request is no longer current or the host already presented it.
        """
        with self._lock:
            if self._active is not pending or self._presented or self._answered:
                return ()
            self._answered = True
            expired = [self._active, *self._queued]
            self._queued.clear()
            return tuple(item for item in expired if item is not None)

    def complete(
        self, request_id: str, generation: int | None = None
    ) -> tuple[bool, QueuedFileSelection | None]:
        """Release ``request_id`` if it is still current and return the next start.

        Callers emit the host result only when this returns ``deliver=True``.
        A request already answered by :meth:`expire_presentation` still releases
        the slot, but its host result is not delivered. Client-gone
        :meth:`drain` wins the race against in-flight work.

        Args:
            request_id: Id of the in-flight selector.
            generation: Client generation of that selector, or ``None`` to
                match only the id.

        Returns:
            ``(True, next)`` when the host result should be delivered.
            ``next`` is the request that should start now, or ``None`` if the
            queue is empty. ``(False, next)`` when the slot was released but
            the app already has a result. ``(False, None)`` when the id is
            not current.
        """
        with self._lock:
            if self._active is None or self._active.request_id != request_id:
                return False, None
            if generation is not None and self._active.generation != generation:
                return False, None
            deliver = not self._answered
            self._active = None
            self._presented = False
            self._answered = False
            if not self._queued:
                return deliver, None
            nxt = self._queued.popleft()
            self._active = nxt
            return deliver, nxt

    @property
    def generation(self) -> int:
        """Client generation that new requests are stamped with."""
        with self._lock:
            return self._generation

    def is_current(self, request_id: str, generation: int) -> bool:
        """Return whether ``request_id`` is still the active selector for ``generation``."""
        with self._lock:
            return (
                self._active is not None
                and self._active.request_id == request_id
                and self._active.generation == generation
            )

    def drain(self) -> tuple[QueuedFileSelection, ...]:
        """Return unanswered requests, then start a new client generation.

        A request already answered by :meth:`expire_presentation` is dropped
        here without being returned again.
        """
        with self._lock:
            items: list[QueuedFileSelection] = []
            if self._active is not None and not self._answered:
                items.append(self._active)
            items.extend(self._queued)
            self._active = None
            self._queued.clear()
            self._presented = False
            self._answered = False
            self._generation += 1
            return tuple(items)

    def clear(self) -> None:
        """Drop remaining requests and start a new client generation."""
        self.drain()

    def _contains(self, request_id: str) -> bool:
        if self._active is not None and self._active.request_id == request_id:
            return True
        return any(item.request_id == request_id for item in self._queued)

    def _is_active(self, request_id: str, generation: int) -> bool:
        return (
            self._active is not None
            and self._active.request_id == request_id
            and self._active.generation == generation
        )


class FileSelectionPresentationTimer:
    """Fail one request when the host does not show its picker in time."""

    def __init__(
        self,
        gate: FileSelectionGate,
        *,
        timeout_seconds: float,
        on_expired: Callable[[tuple[QueuedFileSelection, ...]], None],
    ) -> None:
        """Arm deadlines against ``gate``.

        Args:
            gate: Queue whose active request this timer watches.
            timeout_seconds: Presentation deadline. Must be positive.
            on_expired: Called with the requests to answer as ``UNAVAILABLE``.

        Raises:
            ValueError: ``timeout_seconds`` is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("presentation_timeout_seconds must be > 0.")
        self._gate = gate
        self._timeout_seconds = timeout_seconds
        self._on_expired = on_expired
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._token: QueuedFileSelection | None = None

    def arm(self, pending: QueuedFileSelection) -> None:
        """Start the presentation deadline for ``pending``."""
        self.cancel()
        timer = threading.Timer(self._timeout_seconds, self._fire, args=(pending,))
        timer.daemon = True
        with self._lock:
            self._timer = timer
            self._token = pending
        timer.start()

    def presented(self, request_id: str, generation: int) -> None:
        """Cancel the deadline when the host has shown this request."""
        if self._gate.mark_presented(request_id, generation):
            self._cancel_if(request_id, generation)

    def finished(self, request_id: str, generation: int) -> None:
        """Cancel the deadline because the host finished this request."""
        self._cancel_if(request_id, generation)

    def cancel(self) -> None:
        """Cancel the current deadline, if one is armed."""
        with self._lock:
            timer = self._timer
            self._timer = None
            self._token = None
        if timer is not None:
            timer.cancel()

    def _cancel_if(self, request_id: str, generation: int) -> None:
        with self._lock:
            token = self._token
            if (
                token is None
                or token.request_id != request_id
                or token.generation != generation
            ):
                return
            timer = self._timer
            self._timer = None
            self._token = None
        if timer is not None:
            timer.cancel()

    def _fire(self, pending: QueuedFileSelection) -> None:
        with self._lock:
            if self._token is not pending:
                return
            self._timer = None
            self._token = None
        expired = self._gate.expire_presentation(pending)
        if not expired:
            return
        _LOGGER.warning(
            "File selector %r was not presented within %g seconds.",
            pending.request_id,
            self._timeout_seconds,
        )
        self._on_expired(expired)
