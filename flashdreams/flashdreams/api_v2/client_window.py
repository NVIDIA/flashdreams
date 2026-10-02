# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC

from .input_source import InputSource
from .output_sink import OutputSink


class IClientWindow(InputSource, OutputSink, ABC):
    """Handle application input and output for one client window.

    The runtime opens the window with a session's description, then reads input
    and writes results until the run ends. A window stays open across a session
    reset/replacement. A window may be opened/closed by the runtime during a session
    reset/replacement.

    A window does not describe the output shape. The session does, and the window
    is given that description in :meth:`OutputSink.open`.

    One thread makes every call on a window, so an implementation needs no
    locking except when its backend delivers input from another thread.

    Created by the runtime, never by an application.
    """

    def request_reset(self) -> None:
        """Queue a synthetic ``ResetUserInputEvent`` in this window's input.

        Return the event once through ``get_user_input_events``, together with
        any other pending input, using the window's input timestamp clock.
        The runtime handles it like a reset event received from the client.

        Raises:
            NotImplementedError: This window cannot synthesize reset events.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support reset requests."
        )

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
