# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC

from .input_source import InputSource
from .output_sink import OutputSink


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with the session's description, then reads input
    and writes results until the run ends, and closes it then. A window stays open
    across a session reset.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    One thread makes every call on a window, so an implementation needs no
    locking except when its backend delivers input from another thread.

    Created by the runtime, never by an application.
    """

    def discard_input_event_ids(self, event_ids: tuple[str, ...]) -> None:
        """Report traced input that cannot reach this window.

        Non-interactive windows may ignore this notification. Interactive
        windows use it to terminate client-side latency samples when
        presentation drops a model frame before :meth:`OutputSink.write`.

        Args:
            event_ids: Browser correlation IDs whose acknowledgement frames
                were discarded.
        """
        del event_ids
