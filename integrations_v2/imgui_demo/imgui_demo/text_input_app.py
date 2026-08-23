# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dear ImGui text-input application for the v2 threaded runtime."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .common import (
    DEFAULT_SESSION_DESC,
    BackgroundModelState,
    BackgroundModelThread,
    require_tchw,
)


@dataclass(slots=True)
class TextInputState:
    """Editable state owned by the ImGui UI thread."""

    text: str = ""
    """Current contents of the input widget."""

    request_focus: bool = True
    """Whether the next frame should focus the input widget."""


class TextInputImGUIThread(ImGUIThread[TextInputState]):
    """Draw an editable text field from UI-thread-owned state."""

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw the text input and its current value."""
        del step_index, events
        imgui.set_next_window_pos((16, 16), imgui.Cond_.once)
        imgui.set_next_window_size((420, 130), imgui.Cond_.once)
        imgui.begin("Text input")
        imgui.text("Type into the field:")
        if self.state.request_focus:
            imgui.set_keyboard_focus_here()
            self.state.request_focus = False
        changed, value = imgui.input_text("##text", self.state.text)
        if changed:
            self.state.text = value
        imgui.text(f"Value: {self.state.text}")
        imgui.end()
        return self.presented_model_frame()

    def reset(self) -> None:
        """Clear the input for a new generation."""
        self.state.text = ""
        self.state.request_focus = True
        super().reset()


class TextInputSession(ISession):
    """Run an ImGui text field over a generated background."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one text-input session.

        Args:
            session_desc: Output dimensions and thread frequencies.
            device: Device used for the background model frame.
        """
        require_tchw(session_desc, "text-input")
        self._session_desc = session_desc
        self._device = device

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the text UI and background model threads."""
        self.register_ui_thread(
            TextInputImGUIThread,
            state=TextInputState(),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_model_thread(
            BackgroundModelThread,
            state=BackgroundModelState(
                session_desc=self._session_desc,
                device=self._device,
            ),
        )


class TextInputApplication(IApplication):
    """Create ImGui text-input sessions."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The text-input demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return DEFAULT_SESSION_DESC

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized text-input session."""
        return TextInputSession(session_desc)


def create_app() -> IApplication:
    """Return a new text-input application."""
    return TextInputApplication()
