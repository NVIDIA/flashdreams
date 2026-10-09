# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window that writes an MP4 file."""

from pathlib import Path

from numpy import uint64

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.file_selection_gate import (
    FileSelectionGate,
    QueuedFileSelection,
)
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
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


class Mp4ClientWindow(IClientWindow):
    """Write UI frames to an MP4 file.

    The session must finish on its own because this window never sends a close
    event. Use ``BackpressureMode.BLOCK`` with
    ``PresentationMode.ON_DEMAND`` to write every frame once.

    File-selector requests complete immediately as unavailable. This window
    writes the run to a file; it has no interactive client.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: MP4 file to write. Parent directories are created.
        """
        self._path = Path(path)
        self._video_sink = Mp4OutputSink(path)
        self._pending_events: list[UserInputEvent] = []
        self._file_gate = FileSelectionGate()

    @property
    def path(self) -> Path:
        """Return the output path."""
        return self._path

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

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Change the MP4 presentation dimensions without replacing the session.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.

        Raises:
            RuntimeError: The window is not open or video migration fails.
            ValueError: An expanded MP4 canvas dimension is odd and cannot be
                encoded as ``yuv420p``.
        """
        self._video_sink.request_new_window_size(new_window_size)

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to write a session's output."""
        self._file_gate.clear()
        self._pending_events.clear()
        self._video_sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Encode one frame produced by the UI thread."""
        self._video_sink.write(result)

    def close(self) -> None:
        """Finish the MP4 file."""
        self._video_sink.close()
        self._file_gate.clear()
