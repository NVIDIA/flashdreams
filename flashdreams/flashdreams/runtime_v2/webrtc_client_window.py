# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC client window for the v2 runtime."""

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import replace

from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.file_selection_gate import (
    FileSelectionGate,
    QueuedFileSelection,
)
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    SelectedFilesStatus,
    SelectedFilesUserInputEvent,
    UserInputEvent,
    normalize_selected_file_accept,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class WebRTCClientWindow(IClientWindow):
    """Client window streaming a run to a browser.

    A thin pairing of the :class:`IClientWindow` protocol with
    :class:`WebRTCServer`, which does the serving. Browser events arrive on the
    server's own thread, so they are queued here and handed over in batches when
    the session asks, as the protocol requires.

    The server timestamps arrivals against one stable clock. This window clears
    input buffered during a session handoff, then rebases later events to the
    new session's clock. Input aimed at a completed UI cannot accidentally act
    on its replacement.

    Disconnecting releases only that browser's peer connection. The server and
    current session stay available for a refreshed or replacement client until
    the application is explicitly stopped. Leftover and in-flight file picks
    complete as unavailable; a replacement viewer does not inherit them.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        """Create the WebRTC backend.

        Construction is specific to this implementation; it is not part of the
        ``IClientWindow`` protocol.

        Args:
            host: Interface on which the HTTP server listens.
            port: Listening port. Zero asks the operating system to choose one.
            startup_timeout_seconds: Maximum time to wait for server startup.
        """
        self._input_events: deque[UserInputEvent] = deque()
        self._hide_cursor = False
        self._lock_cursor_to_window = False
        self._file_gate = FileSelectionGate()
        self._input_lock = threading.Lock()
        # Offset from the server's stable clock to the current session's clock.
        self._session_event_offset_us = uint64(0)
        self.server = WebRTCServer(
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
        )

        def handle_input(event: UserInputEvent) -> None:
            """Buffer one backend event for the ``InputSource`` protocol."""
            # TODO: do we really need to buffer all events? Some mouse moves may be superseded by later ones.
            nxt = None
            with self._input_lock:
                if isinstance(event, SelectedFilesUserInputEvent):
                    claimed, nxt = self._file_gate.complete(
                        event.request_id, event.generation
                    )
                    if not claimed:
                        return
                self._input_events.append(event)
            if nxt is not None:
                self._start_file_selection(nxt)

        self.server.register_input_callback(handle_input)
        self.server.register_file_selection_current_callback(self._file_gate.is_current)
        self.server.register_file_selection_invalidate_callback(
            self._invalidate_file_selections
        )

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Show or hide the cursor in the browser window."""
        self.server.configure_cursor(
            hide_cursor=hide_cursor,
            lock_cursor_to_window=self._lock_cursor_to_window,
        )
        self._hide_cursor = hide_cursor

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Release or capture pointer motion in the browser window."""
        self.server.configure_cursor(
            hide_cursor=self._hide_cursor,
            lock_cursor_to_window=lock_cursor_to_window,
        )
        self._lock_cursor_to_window = lock_cursor_to_window

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Change the active WebRTC stream dimensions.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.

        Raises:
            RuntimeError: The server is closed or has no open session.
            TimeoutError: The active video track cannot be reset in time.
        """
        self.server.request_new_window_size(new_window_size)

    def request_selected_files(
        self,
        request_id: str,
        initial_path: str | None = None,
        *,
        accept: Sequence[str] = (),
        max_bytes: int | None = None,
    ) -> None:
        """Ask the connected browser to open a file selector.

        A second distinct ``request_id`` waits until the active selector
        completes so the viewer shows one dialog at a time. A dropped peer
        completes leftovers and in-flight picks as unavailable.

        Args:
            request_id: Stable selector-slot id from the UI control.
            initial_path: Ignored by the browser picker; kept so the window
                signature matches :meth:`IClientWindow.request_selected_files`.
            accept: Filename suffixes such as ``.png``. Empty allows any type.
            max_bytes: Maximum file size in bytes, or ``None`` for the ceiling.
        """
        started = self._file_gate.submit(
            QueuedFileSelection(
                request_id=request_id,
                initial_path=initial_path,
                accept=normalize_selected_file_accept(accept),
                max_bytes=max_bytes,
            )
        )
        if started is not None:
            self._start_file_selection(started)

    def _invalidate_file_selections(self, deliver_unavailable: bool) -> None:
        """Finish this client generation; optionally report leftovers as unavailable."""
        leftover = self._file_gate.drain()
        if not deliver_unavailable or not leftover:
            return
        timestamp = self.server.event_timestamp_us()
        with self._input_lock:
            for item in leftover:
                self._input_events.append(
                    SelectedFilesUserInputEvent(
                        timestamp=timestamp,
                        request_id=item.request_id,
                        status=SelectedFilesStatus.UNAVAILABLE,
                        generation=item.generation,
                    )
                )

    def _start_file_selection(self, pending: QueuedFileSelection) -> None:
        """Arm the WebRTC server for one file-selector request."""
        self.server.request_selected_files(
            pending.request_id,
            initial_path=pending.initial_path,
            accept=pending.accept,
            max_bytes=pending.max_bytes,
            generation=pending.generation,
        )

    def open(self, session_desc: SessionDesc) -> None:
        """Implement ``OutputSink.open`` by configuring WebRTC output.

        Args:
            session_desc: Resolved dimensions, frame rate, and tensor layout.
        """
        self.server.open(session_desc)
        session_event_offset_us = self.server.event_timestamp_us()
        with self._input_lock:
            self._input_events.clear()
            self._session_event_offset_us = session_event_offset_us

    def get_user_input_events(self) -> UserInputEvents:
        """Implement ``InputSource.get_user_input_events`` for browser input.

        Returns:
            Buffered browser events in timestamp order, each returned once.
        """
        with self._input_lock:
            events = list(self._input_events)
            self._input_events.clear()
            session_event_offset_us = int(self._session_event_offset_us)
        return UserInputEvents(
            [
                replace(
                    event,
                    timestamp=uint64(
                        max(
                            0,
                            int(event.get_timestamp()) - session_event_offset_us,
                        )
                    ),
                )
                for event in events
            ]
        )

    def write(self, result: StepResult) -> None:
        """Materialize and queue one UI-composited frame for the browser.

        Args:
            result: One UI-composited frame matching the opened session.
        """
        self.server.write(result)

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Return sender-queue diagnostics."""
        return self.server.metrics_snapshot()

    def close(self) -> None:
        """Implement ``OutputSink.close`` by releasing WebRTC resources."""
        self._invalidate_file_selections(False)
        self.server.close()
