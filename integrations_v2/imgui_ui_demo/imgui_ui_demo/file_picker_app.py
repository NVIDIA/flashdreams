# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui file-picker application for the v2 loop runtime."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_event import SelectedFilesUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .text_input_app import BackgroundModelLoop


@dataclass(slots=True)
class FilePickerState:
    """Last file-selector result shown by the demo overlay."""

    status: str = "No file selected."
    """Human-readable status of the last selector result."""


class FilePickerImGuiUILoop(ImGuiUILoop[FilePickerState]):
    """Request a client file pick and display the chosen name and size."""

    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw the Open-file control and apply any selected-files events."""
        del step_index
        for event in events.get_events():
            if not isinstance(event, SelectedFilesUserInputEvent):
                continue
            if not event.files:
                self.state.status = "Cancelled."
                continue
            chosen = event.files[0]
            self.state.status = f"{chosen.name} ({len(chosen.data)} bytes)"

        imgui.set_next_window_pos(imgui.ImVec2(16.0, 16.0), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(420.0, 130.0), imgui.Cond_.once)
        imgui.begin("File picker")
        try:
            if imgui.button("Open file"):
                self.request_selected_files(uuid.uuid4().hex)
            imgui.text(self.state.status)
        finally:
            imgui.end()
        return self.presented_model_frame()

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
            BackgroundModelLoop,
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
