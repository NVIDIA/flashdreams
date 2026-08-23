# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client window that writes an MP4 file."""

from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.mp4_output_sink import Mp4OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class Mp4ClientWindow(IClientWindow):
    """Write UI frames to an MP4 file and report no input.

    The session must finish on its own because this window never sends a close
    event. Use ``BLOCK`` or ``LOSSLESS`` to keep every frame.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: MP4 file to write. Parent directories are created.
        """
        self._path = Path(path)
        self._video_sink = Mp4OutputSink(path)

    @property
    def path(self) -> Path:
        """Return the output path."""
        return self._path

    def get_user_input_events(self) -> UserInputEvents:
        """Return an empty input batch."""
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to write a session's output."""
        self._video_sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Encode one frame produced by the UI thread."""
        self._video_sink.write(result)

    def close(self) -> None:
        """Finish the MP4 file."""
        self._video_sink.close()
