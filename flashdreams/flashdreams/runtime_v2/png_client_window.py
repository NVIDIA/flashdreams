# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless client window writing generated frames as PNG files."""

from pathlib import Path

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.png_output_sink import PngOutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class PngClientWindow(IClientWindow):
    """Write generated frames as PNG and report no input."""

    def __init__(self, path: str | Path) -> None:
        self._sink = PngOutputSink(path)

    @property
    def path(self) -> Path:
        """Return the first PNG output path."""
        return self._sink.path

    def get_user_input_events(self) -> UserInputEvents:
        """Return an empty input batch."""
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare the PNG sink for a session."""
        self._sink.open(session_desc)

    def write(self, result: StepResult) -> None:
        """Write generated frames to PNG files."""
        self._sink.write(result)

    def close(self) -> None:
        """Close the PNG sink."""
        self._sink.close()


__all__ = ["PngClientWindow"]
