# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC

from .input_source import InputSource
from .output_sink import OutputSink


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with the session's description, then reads input
    and writes results until the run ends. A window stays open across a session
    reset. When the client asks for a replacement session, the runtime closes the
    old session and opens the same window with the replacement's description. A
    session-serving runner can also leave the window open between sessions while
    it waits for another client request.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    One I/O thread at a time makes every call on a window, so an implementation
    needs no locking except when its backend delivers input from another thread.

    Created by the runtime, never by an application.
    """
