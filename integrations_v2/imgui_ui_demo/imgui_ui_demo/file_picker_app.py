# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui file-picker application for the v2 loop runtime."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.selected_file import (
    MAX_SELECTED_FILE_BYTES,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import SelectedFilesUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_BREATHE_PERIOD_S = 3.0
"""Seconds for one black → gray → black cycle in the demo background."""

_BREATHE_BLACK = -1.0
"""Model-frame value for black in ``[-1, 1]``."""

_BREATHE_GRAY = -0.25
"""Brightest gray in the cycle. Stops short of white."""

_FILE_SELECTION_STATUS_TEXT = {
    SelectedFilesStatus.CANCELLED: (
        "SelectedFilesStatus.CANCELLED: user dismissed the selector."
    ),
    SelectedFilesStatus.TOO_LARGE: (
        "SelectedFilesStatus.TOO_LARGE: file exceeded this request's "
        f"max_file_bytes={MAX_SELECTED_FILE_BYTES}."
    ),
    SelectedFilesStatus.DISALLOWED_TYPE: (
        "SelectedFilesStatus.DISALLOWED_TYPE: suffix not in accept=('.bin', '.raw')."
    ),
    SelectedFilesStatus.UNAVAILABLE: (
        "SelectedFilesStatus.UNAVAILABLE: picker could not complete "
        "(no interactive client, gone client, failed chooser, or unreadable file)."
    ),
}
_FILE_SELECTION_ERROR_COLOR = (1.0, 0.25, 0.25, 1.0)
"""RGBA used for cancelled and other failed selector results."""

_OPEN_FILE_A_REQUEST_ID = "open-file-a"
"""Selector slot for Open file (A). Mash uses this same id."""

_OPEN_FILE_B_REQUEST_ID = "open-file-b"
"""Selector slot for Open file (B). Queued behind A when both are requested."""

_OPEN_FILE_A_LABEL = "Open file (A)"
"""ImGui label for selector slot A."""

_OPEN_FILE_B_LABEL = "Open file (B)"
"""ImGui label for selector slot B."""

_OPEN_A_THEN_B_LABEL = "Open A then B"
"""ImGui label that requests A then B in the same UI tick."""

_OPEN_FILES_REQUEST_ID = "open-files"
"""Selector slot for Open files. Asks for more than one file."""

_OPEN_FILES_LABEL = "Open files"
"""ImGui label for a multi-file selector slot."""


class BreathingBackgroundModelLoop(IModelLoop[tuple[SessionDesc, torch.device | str]]):
    """Generate a black-to-gray breathing background for the file-picker demo."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return one background frame along a 3s black-gray cycle."""
        del events
        desc, device = self.state
        phase = (step_index / desc.frames_per_second_for_step) * (
            2.0 * math.pi / _BREATHE_PERIOD_S
        )
        mix = 0.5 - 0.5 * math.cos(phase)
        level = _BREATHE_BLACK + (_BREATHE_GRAY - _BREATHE_BLACK) * mix
        return [
            StepResult(
                step_index=step_index,
                output=torch.full(
                    (1, 3, desc.video_height, desc.video_width),
                    level,
                    dtype=torch.float32,
                    device=device,
                ),
                frame_count=1,
                output_layout=desc.output_layout,
            )
        ]

    def reset(self) -> None:
        return


@dataclass(slots=True)
class FilePickerState:
    """Last file-selector result shown by the demo overlay."""

    status: str = "No file selected."
    """Human-readable status of the last selector result."""


class FilePickerImGuiUILoop(ImGuiUILoop[FilePickerState]):
    """Request a ``.bin`` / ``.raw`` pick and show the chosen name and size."""

    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw the Open-file controls and apply this tick's selector results."""
        del step_index, events
        imgui.set_next_window_pos(imgui.ImVec2(16.0, 16.0), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(520.0, 300.0), imgui.Cond_.once)
        imgui.begin("File picker")
        try:
            clicked_a = imgui.button(_OPEN_FILE_A_LABEL)
            clicked_b = imgui.button(_OPEN_FILE_B_LABEL)
            clicked_a_then_b = imgui.button(_OPEN_A_THEN_B_LABEL)
            clicked_files = imgui.button(_OPEN_FILES_LABEL)
            self._apply_demo_pick(
                self._request_demo_file(
                    _OPEN_FILE_A_REQUEST_ID,
                    open=clicked_a or clicked_a_then_b,
                )
            )
            self._apply_demo_pick(
                self._request_demo_file(
                    _OPEN_FILE_B_REQUEST_ID,
                    open=clicked_b or clicked_a_then_b,
                )
            )
            self._apply_demo_pick(
                self._request_demo_file(
                    _OPEN_FILES_REQUEST_ID,
                    open=clicked_files,
                    multiple=True,
                )
            )
            if self.state.status in _FILE_SELECTION_STATUS_TEXT.values():
                imgui.push_style_color(
                    imgui.Col_.text,
                    imgui.ImVec4(*_FILE_SELECTION_ERROR_COLOR),
                )
                try:
                    imgui.text_wrapped(self.state.status)
                finally:
                    imgui.pop_style_color()
            else:
                imgui.text(self.state.status)
        finally:
            imgui.end()
        return self.presented_model_frame()

    def _request_demo_file(
        self,
        request_id: str,
        *,
        open: bool,
        multiple: bool = False,
    ) -> SelectedFilesUserInputEvent | None:
        """Poll one ``.bin`` / ``.raw`` selector slot and optionally start it."""
        return self.file_selector(
            request_id,
            open=open,
            accept=(".bin", ".raw"),
            max_file_bytes=MAX_SELECTED_FILE_BYTES,
            multiple=multiple,
        )

    def _apply_demo_pick(self, event: SelectedFilesUserInputEvent | None) -> None:
        """Show this tick's selector result on the overlay."""
        if event is None:
            return
        if event.status is not SelectedFilesStatus.OK:
            self.state.status = _FILE_SELECTION_STATUS_TEXT[event.status]
            return
        self.state.status = ", ".join(
            f"{chosen.name} ({len(chosen.data)} bytes)" for chosen in event.files
        )

    def reset(self) -> None:
        """Clear the last selector result for a new generation."""
        self.state.status = "No file selected."
        super().reset()


class FilePickerSession(ISession):
    """Run an ImGui file picker over a generated background."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one file-picker session.

        Args:
            session_desc: Output dimensions and loop frequencies.
            device: Device used for the background model frame.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The file-picker demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._device = device

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the file-picker UI and background model loops."""
        self.register_ui_loop(
            FilePickerImGuiUILoop,
            state=FilePickerState(),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_model_loop(
            BreathingBackgroundModelLoop,
            state=(self._session_desc, self._device),
        )


class FilePickerApplication(IApplication):
    """Create ImGui UI file-picker sessions."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The file-picker demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return SessionDesc(video_width=640, video_height=480)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized file-picker session."""
        return FilePickerSession(session_desc)


def create_app() -> IApplication:
    """Return a new file-picker application."""
    return FilePickerApplication()
