# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC client window for the v2 runtime."""

import threading
import time
from collections import deque

from flashdreams.api_v2.client_window import IClientWindow
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
    the session asks, as the protocol requires.

    A browser that disconnects becomes a close event, so a run through this
    window ends on its own even when the session would generate forever.
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
        self._input_events: deque[tuple[UserInputEvent, int]] = deque()
        self._input_lock = threading.Lock()
        self.server = WebRTCServer(
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
        )

        def handle_input(event: UserInputEvent) -> None:
            """Buffer one backend event for the ``InputSource`` protocol."""
            received_at_ns = time.monotonic_ns()
            with self._input_lock:
                self._input_events.append((event, received_at_ns))

        self.server.register_input_callback(handle_input)

    def open(self, session_desc: SessionDesc) -> None:
        """Implement ``OutputSink.open`` by configuring WebRTC output.

        Args:
            session_desc: Resolved dimensions, frame rate, and tensor layout.
        """
        self.server.open(session_desc)

    def get_user_input_events(self) -> UserInputEvents:
        """Implement ``InputSource.get_user_input_events`` for browser input.

        Returns:
            Buffered browser events in timestamp order, each returned once.
        """
        with self._input_lock:
            received = list(self._input_events)
            self._input_events.clear()
        events = [event for event, _ in received]
        return UserInputEvents(
            events,
            received_at_ns={
                id(event): received_at_ns for event, received_at_ns in received
            },
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
