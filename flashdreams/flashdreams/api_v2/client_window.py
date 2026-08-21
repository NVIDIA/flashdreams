# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC, abstractmethod

from .input_source import InputSource
from .output_sink import OutputSink


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with the session's description, then reads input
    and writes results until the run ends. A window stays open across a session
    reset. When the client asks for a replacement session, the runtime closes the
    old session and opens the same window with the replacement's description. A
    persistent window can also remain open between sessions while the runtime
    waits for another client request.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    One runtime thread at a time makes every call on a window, so an
    implementation needs no locking except when its backend delivers input from
    another thread.

    Created by the runtime, never by an application.
    """

    keeps_open_between_sessions: bool = False
    """Whether sessions start on demand and the window persists between them.

    When false, the runtime starts the resolved initial session immediately and
    returns after it ends unless the client requested a replacement. When true,
    the runtime opens the window before creating a session, waits for a client
    request, and returns to waiting after completion or disconnection.
    """

    @abstractmethod
    def close(self) -> None:
        """Release this window's resources.

        This must be safe before :meth:`OutputSink.open` and after an earlier
        call. The session loop performs the meaningful close on its I/O thread;
        the application runner calls it again as a lifetime-cleanup fallback.
        """
        ...
