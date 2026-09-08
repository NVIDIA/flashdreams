# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy UI text-input application for the v2 loop runtime."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class BackgroundModelLoop(IModelLoop[tuple[SessionDesc, torch.device | str]]):
    """Generate a dark background beneath the text-input UI layer."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return one dark background frame."""
        del events
        desc, device = self.state
        return [
            StepResult(
                step_index=step_index,
                output=torch.full(
                    (1, 3, desc.video_height, desc.video_width),
                    -0.85,
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
class TextInputState:
    """Editable state owned by the SlangPy UI loop."""

    text: str = ""
    """Current contents of the input widget."""

    input_widget: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy text-input widget."""

    value_widget: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy text widget showing the current value."""


class TextInputSlangPyUILoop(SlangPyUILoop[TextInputState]):
    """Draw an editable text field from UI-loop-owned state."""

    def step_ui(
        self,
        ui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw the text input and its current value."""
        del step_index, events
        if self.state.input_widget is None:
            window = ui.Window(
                ui.screen,
                "Text input",
                position=(16, 16),
                size=(420, 130),
            )
            ui.Text(window, "Type into the field:")
            self.state.input_widget = ui.InputText(
                window,
                "Text",
                self.state.text,
                self._set_text,
            )
            self.state.value_widget = ui.Text(window, f"Value: {self.state.text}")
        return self.presented_model_frame()

    def _set_text(self, value: str) -> None:
        self.state.text = value
        if self.state.value_widget is not None:
            self.state.value_widget.text = f"Value: {value}"

    def reset(self) -> None:
        """Clear the input for a new generation."""
        self.state.text = ""
        if self.state.input_widget is not None:
            self.state.input_widget.value = ""
        if self.state.value_widget is not None:
            self.state.value_widget.text = "Value: "
        super().reset()


class TextInputSession(ISession):
    """Run a SlangPy text field over a generated background."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one text-input session.

        Args:
            session_desc: Output dimensions and loop frequencies.
            device: Device used for the background model frame.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The text-input demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._device = device

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the text UI and background model loops."""
        self.register_ui_loop(
            TextInputSlangPyUILoop,
            state=TextInputState(),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_model_loop(
            BackgroundModelLoop,
            state=(self._session_desc, self._device),
        )


class TextInputApplication(IApplication):
    """Create SlangPy UI text-input sessions."""

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The text-input demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return SessionDesc(video_width=640, video_height=480)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized text-input session."""
        return TextInputSession(session_desc)


def create_app() -> IApplication:
    """Return a new text-input application."""
    return TextInputApplication()
