# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window composed with an additional input source."""

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.input_source import InputSource
from flashdreams.api_v2.input_source import SessionInputSource
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class CompositeClientWindow(IClientWindow):
    """Add an input source to an existing client window."""

    def __init__(
        self,
        client_window: IClientWindow,
        input_source: InputSource,
    ) -> None:
        self._client_window = client_window
        self._input_source = input_source

    @property
    def client_window(self) -> IClientWindow:
        """Return the wrapped client window."""
        return self._client_window

    def get_user_input_events(self) -> UserInputEvents:
        """Return wrapped-window input followed by additional input."""
        events = self._client_window.get_user_input_events().get_events()
        events.extend(self._input_source.get_user_input_events().get_events())
        return UserInputEvents(events)

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Delegate cursor visibility."""
        self._client_window.request_hide_cursor(hide_cursor)

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Delegate cursor locking."""
        self._client_window.request_lock_cursor_to_window(lock_cursor_to_window)

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Delegate window resizing."""
        self._client_window.request_new_window_size(new_window_size)

    def open(self, session_desc: SessionDesc) -> None:
        """Open the wrapped window and any session-aware input source."""
        self._client_window.open(session_desc)
        if isinstance(self._input_source, SessionInputSource):
            self._input_source.open()

    def write(self, result: StepResult) -> None:
        """Delegate output."""
        self._client_window.write(result)

    def close(self) -> None:
        """Close the wrapped window and any session-aware input source."""
        try:
            self._client_window.close()
        finally:
            if isinstance(self._input_source, SessionInputSource):
                self._input_source.close()


__all__ = ["CompositeClientWindow"]