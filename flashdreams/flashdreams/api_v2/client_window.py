# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window abstract interface."""

from abc import ABC
from collections.abc import Sequence

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
    def request_selected_files(
        self,
        request_id: str,
        initial_path: str | None = None,
        *,
        accept: Sequence[str] = (),
        max_bytes: int | None = None,
    ) -> None:
        """Ask this client to pick files and report them as input events.

        The chosen files arrive later through
        :meth:`InputSource.get_user_input_events`. Windows classify the choice
        with
        :func:`~flashdreams.runtime_v2.user_input_event.selected_file_policy_status`
        (type before size) and clamp size to the 32 MiB ceiling. The event
        carries a ``status`` so cancel, oversize, and disallowed type are
        distinct. Applications read that ``status``; they do not re-check type
        or size. A second call with the same ``request_id`` while the first is
        still active or queued is ignored. Use one stable id per UI control so
        extra clicks while the picker is opening do not enqueue more dialogs.
        A new id on every click is a new control and queues another dialog.
        A different ``request_id`` waits until the active selector completes,
        so two controls still run one dialog at a time. Every accepted request
        later produces one event. ``cancelled`` is only a user dismiss. If the
        interactive client is gone while the session continues (dropped WebRTC
        peer, closed native window), leftovers complete as ``unavailable``. A
        picker that outlives that client cannot complete a later request that
        reused the id. Window ``close`` / session replacement finish leftovers
        without delivering them to the next session.

        Args:
            request_id: Stable id for this selector slot, typically one per
                UI control. Reuse it after the matching event arrives.
            initial_path: Directory the selector should start in; ``None`` lets
                the window choose a default.
            accept: Filename suffixes such as ``.png``. Empty allows any type.
            max_bytes: Maximum file size in bytes, or ``None`` for the ceiling.
        """
        pass
