# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MP4 output paired with descriptor-backed synthetic input."""

from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.input_source import SessionInputSource
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class SyntheticInputMp4ClientWindow(IClientWindow):
    """Delegate MP4 output while replaying synthetic input on UI polls."""

    def __init__(
        self,
        output: Mp4ClientWindow,
        input_source: SessionInputSource,
    ) -> None:
        self._output = output
        self._input_source = input_source

    @property
    def path(self) -> Path:
        """Return the delegated MP4 path."""
        return self._output.path

    def get_user_input_events(self) -> UserInputEvents:
        """Return due descriptor events."""
        return self._input_source.get_user_input_events()

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Delegate MP4 canvas resizing."""
        self._output.request_new_window_size(new_window_size)

    def open(self, session_desc: SessionDesc) -> None:
        """Open MP4 output, then begin the session-relative input clock."""
        self._output.open(session_desc)
        self._input_source.open()

    def write(self, result: StepResult) -> None:
        """Delegate frame encoding."""
        self._output.write(result)

    def close(self) -> None:
        """Close MP4 output and clear the input source session."""
        try:
            self._output.close()
        finally:
            self._input_source.close()


__all__ = ["SyntheticInputMp4ClientWindow"]