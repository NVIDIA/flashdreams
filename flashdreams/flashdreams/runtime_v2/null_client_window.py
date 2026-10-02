# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window that discards output and accepts synthetic reset requests."""

from time import monotonic_ns

from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import ResetUserInputEvent, UserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class NullClientWindow(IClientWindow):
    """Discard output and report only explicitly requested reset events."""

    def __init__(self) -> None:
        self._session_started_ns: int | None = None
        self._input_events: list[UserInputEvent] = []

    def request_reset(self) -> None:
        """Queue a reset using the current session's input clock.

        Raises:
            RuntimeError: The window is not open.
        """
        started_ns = self._session_started_ns
        if started_ns is None:
            raise RuntimeError("Open the client window before requesting a reset.")
        self._input_events.append(
            ResetUserInputEvent(
                timestamp=uint64((monotonic_ns() - started_ns) // 1_000)
            )
        )

    def get_user_input_events(self) -> UserInputEvents:
        """Return requested resets once, or an empty input batch."""
        events = UserInputEvents(self._input_events)
        self._input_events.clear()
        return events

    def open(self, session_desc: SessionDesc) -> None:
        """Accept a session without opening an output."""
        del session_desc
        self._session_started_ns = monotonic_ns()
        self._input_events.clear()

    def write(self, result: StepResult) -> None:
        """Discard one result."""
        del result

    def close(self) -> None:
        """Discard input belonging to the completed session."""
        self._session_started_ns = None
        self._input_events.clear()
