# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

import logging
import threading
import time
from abc import ABC
from collections.abc import Callable
from typing import final

from numpy import uint64

from flashdreams.runtime_v2.selected_file import (
    FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS,
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.user_input_event import SelectedFilesUserInputEvent

from .input_source import InputSource
from .output_sink import OutputSink
from .user_input_event import UserInputEvent

_LOGGER = logging.getLogger(__name__)


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with a session's description, then reads input
    and writes results until the run ends. A window stays open across a session
    reset/replacement. A window may be opened/closed by the runtime during a session
    reset/replacement.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    The runtime makes every call on a window from one thread. A window still
    needs locking where its backend or file picker delivers input from another
    thread, such as :meth:`_put_user_input_event`.

    Created by the runtime, never by an application.
    """

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

    # File selection. The public methods below hold the rules: one picker at a
    # time, a deadline until the picker is shown, and one result per request.
    # A window with a picker implements the ``_request_file_selector`` hook and
    # reports through ``_file_selector_opened`` and ``_complete_file_selector``.
    # A window that wraps another overrides the public methods to forward them.

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
        ``request.request_id``. While a picker is open, another request is
        ignored. A picker not shown within
        :data:`FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS` completes as
        ``UNAVAILABLE``; once shown, the user may take as long as they want.

        Args:
            request: Selector request queued by the UI loop.
        """
        self._expire_file_selectors()
        with _FILE_SELECTOR_LOCK:
            if self._active_file_selector is not None:
                return
            selector_id = self._last_file_selector_id + 1
            self._last_file_selector_id = selector_id
            self._active_file_selector = (selector_id, request)
            self._file_selector_deadline = (
                self._file_selector_clock()
                + FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS
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
        with _FILE_SELECTOR_LOCK:
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
        finishes, from any thread. The default has no picker and completes as
        ``UNAVAILABLE``.
        """
        del request
        self._complete_file_selector(selector_id, SelectedFilesStatus.UNAVAILABLE)

    # Optional to implement
    def _cancel_active_file_selectors(self) -> None:
        """Withdraw the picker that was flushed or that timed out."""
        pass

    # Optional to implement
    def _put_user_input_event(self, event: UserInputEvent) -> None:
        """Queue ``event`` for :meth:`InputSource.get_user_input_events`.

        Every window implements this, including one without a picker, or the
        ``UNAVAILABLE`` answer to its requests is lost. Only window code calls
        it. It may run on any thread and runs under the file-selector lock, so
        it must not call back into file selection.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _put_user_input_event to "
            "report file selections."
        )

    @final
    def _file_selector_opened(self, selector_id: int) -> None:
        """Stop the deadline because the picker for ``selector_id`` is shown."""
        with _FILE_SELECTOR_LOCK:
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
        """Report the result for ``selector_id``; a stale id is dropped."""
        with _FILE_SELECTOR_LOCK:
            active = self._active_file_selector
            if active is None or active[0] != selector_id:
                return
            self._active_file_selector = None
            self._file_selector_deadline = None
            # Queued under the lock so flush_file_selectors, which runs before
            # the next session's open(), cannot slip in before the put.
            self._put_user_input_event(
                SelectedFilesUserInputEvent(
                    timestamp=uint64(0),
                    request_id=active[1].request_id,
                    status=status,
                    files=files,
                )
            )

    @final
    def _expire_file_selectors(self) -> None:
        """Complete a picker not shown by its deadline as ``UNAVAILABLE``.

        Windows with a picker call this each time input is read.
        """
        with _FILE_SELECTOR_LOCK:
            active = self._active_file_selector
            deadline = self._file_selector_deadline
            if (
                active is None
                or deadline is None
                or self._file_selector_clock() < deadline
            ):
                return
        _LOGGER.warning(
            "File selector %r was not shown within %g seconds.",
            active[1].request_id,
            FILE_SELECTION_PRESENTATION_TIMEOUT_SECONDS,
        )
        self._complete_file_selector(active[0], SelectedFilesStatus.UNAVAILABLE)
        self._cancel_active_file_selectors()


# ponytail: one lock for every window keeps subclasses constructor-free; it only
# guards a few assignments and one event put. Upgrade path: a per-window lock
# set in ``__init__``.
_FILE_SELECTOR_LOCK = threading.Lock()
