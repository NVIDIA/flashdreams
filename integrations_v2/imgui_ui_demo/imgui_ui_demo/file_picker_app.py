# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui file-picker application for the v2 loop runtime."""

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.selected_file import SelectedFilesStatus
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
    SelectedFilesStatus.BUSY: (
        "SelectedFilesStatus.BUSY: another picker was already open."
    ),
    SelectedFilesStatus.TOO_LARGE: (
        "SelectedFilesStatus.TOO_LARGE: a file was over 32 MiB, the files "
        "together were over 256 MiB, or too many files were chosen."
    ),
    SelectedFilesStatus.DISALLOWED_TYPE: (
        "SelectedFilesStatus.DISALLOWED_TYPE: suffix not in accept=('.bin', '.raw')."
    ),
    SelectedFilesStatus.UNAVAILABLE: (
        "SelectedFilesStatus.UNAVAILABLE: picker could not complete "
        "(no or gone client, failed chooser, or unreadable file)."
    ),
}
_FILE_SELECTION_ERROR_COLOR = (1.0, 0.25, 0.25, 1.0)
"""RGBA used for cancelled and other failed selector results."""

_OPEN_FILE_A_REQUEST_ID = "open-file-a"
"""Selector slot for Open file (A)."""

_OPEN_FILE_B_REQUEST_ID = "open-file-b"
"""Selector slot for Open file (B)."""

_OPEN_FILE_A_LABEL = "Open file (A)"
"""ImGui label for selector slot A."""

_OPEN_FILE_B_LABEL = "Open file (B)"
"""ImGui label for selector slot B."""

_OPEN_FILES_REQUEST_ID = "open-files"
"""Selector slot for Open files. Asks for more than one file."""

_OPEN_FILES_LABEL = "Open files"
"""ImGui label for a multi-file selector slot."""

_OPEN_FILES_MAX_FILE_COUNT = 8
"""Open files' ``max_file_count``, raised from the default of one."""

_FILE_SELECTOR_SLOTS = (
    (_OPEN_FILE_A_LABEL, _OPEN_FILE_A_REQUEST_ID, 1),
    (_OPEN_FILE_B_LABEL, _OPEN_FILE_B_REQUEST_ID, 1),
    (_OPEN_FILES_LABEL, _OPEN_FILES_REQUEST_ID, _OPEN_FILES_MAX_FILE_COUNT),
)
"""``(label, request_id, max_file_count)`` for each Open button, in display order."""

_MAX_FILE_BYTES = 32 * 1024 * 1024
"""Each button's ``max_file_bytes`` (32 MiB), lowered from the 1 GiB default."""

_MAX_TOTAL_BYTES = 256 * 1024 * 1024
"""Each button's ``max_total_bytes`` (256 MiB), lowered from the 1 GiB default.
Only Open files can pick enough to reach it."""

_NO_FILE_SELECTED = "No file selected."
"""Status shown for a button before its first result."""

_RESTART_DELAY_S = 5.0
"""Seconds before a requested restart, long enough to open a picker first."""


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
    """Last file-selector result for each Open button."""

    statuses: dict[str, str] = field(default_factory=dict)
    """Status text by ``request_id``. A missing id has no result yet."""

    restart_at: float | None = None
    """``time.monotonic()`` deadline for a requested session restart."""


def _status_text(event: SelectedFilesUserInputEvent) -> str:
    """Return the overlay text for one selector result."""
    if event.status is not SelectedFilesStatus.OK:
        return _FILE_SELECTION_STATUS_TEXT[event.status]
    return ", ".join(
        f"{chosen.name} ({len(chosen.data)} bytes)" for chosen in event.files
    )


class FilePickerImGuiUILoop(ImGuiUILoop[FilePickerState]):
    """Request ``.bin`` / ``.raw`` picks and show each button's own result."""

    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw the Open buttons and one status line per button."""
        del step_index
        for event in events.get_events():
            if isinstance(event, SelectedFilesUserInputEvent):
                self.state.statuses[event.request_id] = _status_text(event)
        imgui.set_next_window_pos(imgui.ImVec2(16.0, 16.0), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(520.0, 300.0), imgui.Cond_.once)
        imgui.begin("File picker")
        try:
            for label, request_id, max_file_count in _FILE_SELECTOR_SLOTS:
                if imgui.button(label):
                    self.request_file_selector(
                        request_id,
                        accept=(".bin", ".raw"),
                        max_file_bytes=_MAX_FILE_BYTES,
                        max_total_bytes=_MAX_TOTAL_BYTES,
                        max_file_count=max_file_count,
                    )
            if self.state.restart_at is None:
                if imgui.button(f"Restart session in {_RESTART_DELAY_S:.0f}s"):
                    self.state.restart_at = time.monotonic() + _RESTART_DELAY_S
            else:
                remaining = self.state.restart_at - time.monotonic()
                if remaining <= 0:
                    self.request_new_session(self.session_desc)
                imgui.text(f"Restarting in {max(remaining, 0.0):.1f}s...")
            for label, request_id, _ in _FILE_SELECTOR_SLOTS:
                status = self.state.statuses.get(request_id, _NO_FILE_SELECTED)
                line = f"{label}: {status}"
                if status not in _FILE_SELECTION_STATUS_TEXT.values():
                    imgui.text(line)
                    continue
                imgui.push_style_color(
                    imgui.Col_.text,
                    imgui.ImVec4(*_FILE_SELECTION_ERROR_COLOR),
                )
                try:
                    imgui.text_wrapped(line)
                finally:
                    imgui.pop_style_color()
        finally:
            imgui.end()
        return self.presented_model_frame()

    def reset(self) -> None:
        """Clear every button's result for a new generation."""
        self.state.statuses.clear()
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
