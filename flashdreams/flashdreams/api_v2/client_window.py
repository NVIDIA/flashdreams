# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import final
from urllib.parse import urlsplit

from numpy import uint64

from flashdreams.runtime_v2.selected_file import (
    FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS,
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
    clamp_selected_file_max_bytes,
    selected_files_policy_status,
)
from flashdreams.runtime_v2.user_input_event import SelectedFilesUserInputEvent

from .input_source import InputSource
from .output_sink import OutputSink
from .user_input_event import UserInputEvent

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClientWindowLink:
    """Clickable HTTP link over a normalized rectangle of the output content."""

    label: str
    """Accessible link label, containing one to 256 characters."""

    url: str
    """Absolute HTTP(S) destination without user information or control characters."""

    bounds: tuple[float, float, float, float]
    """Normalized ``(left, top, right, bottom)`` bounds within the output content."""

    def __post_init__(self) -> None:
        """Validate immutable bounds and a safe browser destination.

        Raises:
            TypeError: A field has an unsupported type.
            ValueError: A label, destination, or rectangle is invalid.
        """
        if not isinstance(self.label, str) or not isinstance(self.url, str):
            raise TypeError("Link label and URL must be strings.")
        if not self.label.strip() or len(self.label) > 256:
            raise ValueError("Link label must contain one to 256 characters.")
        if (
            not self.url
            or len(self.url) > 4096
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in self.url)
            or self.url != self.url.strip()
            or "\\" in self.url
        ):
            raise ValueError(
                "Link URL contains invalid characters or exceeds 4096 characters."
            )
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(char.isspace() for char in parsed.netloc)
        ):
            raise ValueError(
                "Link URL must be absolute HTTP(S) without user information."
            )
        # Accessing port also rejects malformed or out-of-range ports.
        parsed.port
        if (
            not isinstance(self.bounds, tuple)
            or len(self.bounds) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in self.bounds
            )
        ):
            raise TypeError("Link bounds must be a tuple of four numbers.")
        if not all(0 <= value <= 1 and math.isfinite(value) for value in self.bounds):
            raise ValueError("Link bounds must be finite and within [0, 1].")
        left, top, right, bottom = self.bounds
        if left >= right or top >= bottom:
            raise ValueError("Link bounds must have positive width and height.")


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with a session's description, then reads input
    and writes results until the run ends. A window stays open across a session
    reset/replacement. A window may be opened/closed by the runtime during a session
    reset/replacement.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    The runtime makes every call on a window from the UI thread. A window still
    needs locking where its backend delivers input from another thread.

    File selection: this class owns the rules (one picker at a time, the
    presentation deadline, result checks, stale results). A window with a
    picker implements :meth:`_request_file_selector` and
    :meth:`_cancel_active_file_selectors`, and calls
    :meth:`_expire_file_selectors` when input is read. Its reports must reach
    this class on the UI thread. :meth:`OutputSink.open` drops events queued
    for the previous session. A window wrapping another forwards
    :meth:`request_file_selector` and :meth:`flush_file_selectors`.

    Created by the runtime, never by an application.
    """

    # Optional to implement
    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        """Replace clickable output regions; an empty tuple clears all links."""
        pass

    # Optional to implement
    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Show or hide the cursor for this client window."""
        pass

    # Optional to implement
    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Release or capture pointer motion for this client window."""
        pass

    # Optional to implement
    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Resize this client window without replacing it.

        Backends whose output dimensions cannot change may leave this unimplemented.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.
        """
        pass

    # Plain class defaults so subclasses need no constructor call; assigning
    # one on an instance shadows it for that window only.
    _active_file_selector: tuple[int, FileSelectionRequest] | None = None
    _file_selector_deadline: float | None = None
    _last_file_selector_id = 0
    _file_selector_clock: Callable[[], float] = staticmethod(time.monotonic)

    def request_file_selector(self, request: FileSelectionRequest) -> None:
        """Show a file selector; its result arrives later as an input event.

        The result is one ``SelectedFilesUserInputEvent`` from
        :meth:`InputSource.get_user_input_events` carrying
        ``request.request_id``. A request made while a picker is open is
        answered as ``BUSY``.

        Args:
            request: Selector request queued by the UI loop.
        """
        self._expire_file_selectors()
        if self._active_file_selector is not None:
            self._put_user_input_event(
                _selected_files_event(request, SelectedFilesStatus.BUSY)
            )
            return
        selector_id = self._last_file_selector_id + 1
        self._last_file_selector_id = selector_id
        self._active_file_selector = (selector_id, request)
        self._file_selector_deadline = (
            self._file_selector_clock() + FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS
        )
        try:
            self._request_file_selector(selector_id, request)
        except Exception:
            _LOGGER.exception("File selector %r failed to start.", request.request_id)
            self._complete_file_selector(selector_id, SelectedFilesStatus.UNAVAILABLE)

    def flush_file_selectors(self) -> None:
        """Forget the open picker without a result, for a session that ended.

        The window withdraws a picker still on screen where it can; whatever
        that picker returns is dropped.
        """
        active = self._active_file_selector
        self._active_file_selector = None
        self._file_selector_deadline = None
        if active is not None:
            self._cancel_active_file_selectors()

    # Optional to implement
    def _request_file_selector(
        self, selector_id: int, request: FileSelectionRequest
    ) -> None:
        """Show a picker for ``request`` without blocking.

        Call :meth:`_file_selector_opened` once it is shown, then
        :meth:`_complete_file_selector` with ``selector_id`` when the user
        finishes, both on the UI thread. The default has no picker and
        completes as ``UNAVAILABLE``.
        """
        del request
        self._complete_file_selector(selector_id, SelectedFilesStatus.UNAVAILABLE)

    # Optional to implement
    def _cancel_active_file_selectors(self) -> None:
        """Withdraw the picker that was flushed or that timed out."""
        pass

    @abstractmethod
    def _put_user_input_event(self, event: UserInputEvent) -> None:
        """Queue ``event`` for :meth:`InputSource.get_user_input_events`.

        Required even for a window without a picker, which answers every
        request through it. Only this class calls it, on the UI thread, and it
        must not call back into file selection.
        """

    @final
    def _file_selector_opened(self, selector_id: int) -> None:
        """Stop the deadline because the picker for ``selector_id`` is shown."""
        active = self._active_file_selector
        if active is not None and active[0] == selector_id:
            self._file_selector_deadline = None

    @final
    def _complete_file_selector(
        self,
        selector_id: int,
        status: SelectedFilesStatus,
        files: tuple[SelectedFile, ...] = (),
    ) -> None:
        """Report the result for ``selector_id``; a stale id is dropped.

        A result that breaks the request's rules (type, size, count) is
        reported by the status it earns instead.
        """
        active = self._active_file_selector
        if active is None or active[0] != selector_id:
            return
        self._active_file_selector = None
        self._file_selector_deadline = None
        request = active[1]
        status, files = _checked_result(request, status, files)
        self._put_user_input_event(_selected_files_event(request, status, files))

    @final
    def _expire_file_selectors(self) -> None:
        """Complete a picker not shown by its deadline as ``UNAVAILABLE``.

        Windows with a picker call this each time input is read.
        """
        active = self._active_file_selector
        deadline = self._file_selector_deadline
        if active is None or deadline is None or self._file_selector_clock() < deadline:
            return
        _LOGGER.warning(
            "File selector %r was not shown within %g seconds.",
            active[1].request_id,
            FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS,
        )
        self._complete_file_selector(active[0], SelectedFilesStatus.UNAVAILABLE)
        self._cancel_active_file_selectors()


def _selected_files_event(
    request: FileSelectionRequest,
    status: SelectedFilesStatus,
    files: tuple[SelectedFile, ...] = (),
) -> SelectedFilesUserInputEvent:
    """Return the event answering ``request``."""
    return SelectedFilesUserInputEvent(
        timestamp=uint64(0),
        request_id=request.request_id,
        status=status,
        files=files,
    )


def _checked_result(
    request: FileSelectionRequest,
    status: SelectedFilesStatus,
    files: tuple[SelectedFile, ...],
) -> tuple[SelectedFilesStatus, tuple[SelectedFile, ...]]:
    """Return the status and files a window's result earns under ``request``."""
    if status is SelectedFilesStatus.BUSY:
        status = SelectedFilesStatus.UNAVAILABLE
    if status is not SelectedFilesStatus.OK:
        return status, ()
    if not files or (len(files) > 1 and not request.multiple):
        _LOGGER.warning("File selector %r returned no valid set.", request.request_id)
        return SelectedFilesStatus.UNAVAILABLE, ()
    files = tuple(replace(chosen, name=_base_name(chosen.name)) for chosen in files)
    rejected = selected_files_policy_status(
        [(chosen.name, len(chosen.data)) for chosen in files],
        accept=request.accept,
        max_bytes=clamp_selected_file_max_bytes(request.max_bytes),
    )
    if rejected is not None:
        return rejected, ()
    return status, files


def _base_name(name: str) -> str:
    """Return ``name`` without any POSIX or Windows directory part."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return "upload" if base in ("", ".", "..") else base
