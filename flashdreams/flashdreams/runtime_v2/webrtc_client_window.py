# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC client window for the v2 runtime."""

import threading
from collections import deque
from dataclasses import replace

from numpy import uint64

from flashdreams.api_v2.client_window import ClientWindowLink, IClientWindow
from flashdreams.runtime_v2.selected_file import (
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class WebRTCClientWindow(IClientWindow):
    """Client window streaming a run to a browser.

    A thin pairing of the :class:`IClientWindow` protocol with
    :class:`WebRTCServer`, which does the serving. Browser events arrive on the
    server's own thread, so they are queued here and handed over in batches when
    the session asks, as the protocol requires. File-selector reports are
    queued the same way and applied on the UI thread when input is read.

    The server timestamps arrivals against one stable clock. This window clears
    input buffered during a session handoff, then rebases later events to the
    new session's clock. Input aimed at a completed UI cannot accidentally act
    on its replacement.

    Disconnecting releases only that browser's peer connection. The server and
    current session stay available for a refreshed or replacement client until
    the application is explicitly stopped.
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
        self._file_selector_reports: deque[
            tuple[int, SelectedFilesStatus, tuple[SelectedFile, ...]]
        ] = deque()
        self._hide_cursor = False
        self._lock_cursor_to_window = False
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
            with self._input_lock:
                self._input_events.append(event)

        def report_file_selector(
            selector_id: str,
            status: SelectedFilesStatus,
            files: tuple[SelectedFile, ...],
        ) -> None:
            """Hand a server-thread picker result to the UI thread."""
            with self._input_lock:
                self._file_selector_reports.append((int(selector_id), status, files))

        self.server.register_input_callback(handle_input)
        self.server.register_file_selector_callback(report_file_selector)

    def request_links(self, links: tuple[ClientWindowLink, ...]) -> None:
        """Replace links opened locally by the browser viewing this window."""
        self.server.configure_links(links)

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

    def _request_file_selector(
        self, selector_id: int, request: FileSelectionRequest
    ) -> None:
        """Ask the connected browser to open a file selector."""
        self.server.request_file_selector(str(selector_id), request)

    def _cancel_active_file_selectors(self) -> None:
        """Refuse a later upload for the selector that was forgotten."""
        self.server.cancel_file_selectors()

    def put_user_input_event(self, event: UserInputEvent) -> None:
        """Queue ``event`` on the server's clock, like browser input."""
        event = replace(event, timestamp=self.server.event_timestamp_us())
        with self._input_lock:
            self._input_events.append(event)

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
            reports = list(self._file_selector_reports)
            self._file_selector_reports.clear()
        for selector_id, status, files in reports:
            self._complete_file_selector(selector_id, status, files)
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
        self.server.close()
