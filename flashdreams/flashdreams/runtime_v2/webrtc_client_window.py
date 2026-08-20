# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC client window for the v2 runtime."""

import queue

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class WebRTCClientWindow(IClientWindow):
    """Implement ``IClientWindow`` with WebRTC input and presentation."""

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
        self._input_events: queue.SimpleQueue[UserInputEvent] = queue.SimpleQueue()
        self.server = WebRTCServer(
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
        )

        def handle_input(event: UserInputEvent) -> None:
            """Buffer one backend event for the ``InputSource`` protocol."""
            self._input_events.put(event)

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
        events = []
        while True:
            try:
                events.append(self._input_events.get_nowait())
            except queue.Empty:
                return UserInputEvents(events)

    def write(self, result: StepResult) -> None:
        """Implement ``OutputSink.write`` by delivering a result to the browser.

        Args:
            result: Generated frames matching the opened session.
        """
        self.server.write(result)

    def close(self) -> None:
        """Implement ``OutputSink.close`` by releasing WebRTC resources."""
        self.server.close()
