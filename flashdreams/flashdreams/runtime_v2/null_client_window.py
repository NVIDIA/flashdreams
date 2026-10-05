# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window that discards output and reports no input."""

from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.file_selection_gate import (
    FileSelectionGate,
    QueuedFileSelection,
)
from flashdreams.runtime_v2.selected_file import (
    FileSelectionRequest,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    SelectedFilesUserInputEvent,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class NullClientWindow(IClientWindow):
    """Discard output and report no input.

    File-selector requests complete immediately as unavailable. This window
    has no interactive client.
    """

    def __init__(self) -> None:
        """Create a window that drops output."""
        self._pending_events: list[UserInputEvent] = []
        self._file_gate = FileSelectionGate()

    def get_user_input_events(self) -> UserInputEvents:
        """Return file-selector results queued since the previous poll."""
        events = self._pending_events
        self._pending_events = []
        return UserInputEvents(events)

    def request_selected_files(self, request: FileSelectionRequest) -> None:
        """Complete a file-selector request as unavailable."""
        started = self._file_gate.submit(QueuedFileSelection.from_request(request))
        while started is not None:
            claimed, nxt = self._file_gate.complete(
                started.request_id, started.generation
            )
            if claimed:
                self._pending_events.append(
                    SelectedFilesUserInputEvent(
                        timestamp=uint64(0),
                        request_id=started.request_id,
                        status=SelectedFilesStatus.UNAVAILABLE,
                        _generation=started.generation,
                    )
                )
            started = nxt

    def open(self, session_desc: SessionDesc) -> None:
        """Accept a session without opening an output."""
        del session_desc
        self._file_gate.clear()
        self._pending_events.clear()

    def write(self, result: StepResult) -> None:
        """Discard one result."""
        del result

    def close(self) -> None:
        """Close without releasing any resources."""
        self._file_gate.clear()
