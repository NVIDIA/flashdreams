# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC

from flashdreams.runtime_v2.selected_file import FileSelectionRequest

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

    # Optional to implement
    def request_selected_files(self, request: FileSelectionRequest) -> None:
        """Request to open a file selector local to the client this window drives.

        The result arrives later through
        :meth:`InputSource.get_user_input_events` as
        :class:`~flashdreams.runtime_v2.user_input_event.SelectedFilesUserInputEvent`.

        Args:
            request: Selector request queued by the UI loop.
        """
        pass
