# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy UI keyboard signaling across the v2 loop boundary."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEventData,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(slots=True)
class ColorToggleModelState:
    """Color state owned exclusively by the model loop."""

    session_desc: SessionDesc
    """Output dimensions and layout for generated frames."""

    device: torch.device | str
    """Device used for model output."""

    blue: bool = False
    """Whether the next model frame is blue instead of red."""

    def _toggle_color(self) -> None:
        self.blue = not self.blue


class ColorToggleModelLoop(IModelLoop[ColorToggleModelState]):
    """Generate a solid red or blue frame from model-loop-owned state."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return the color selected through the UI loop."""
        del events
        return [
            StepResult(
                step_index=step_index,
                output=_color_frame(self.state),
                frame_count=1,
                output_layout=self.state.session_desc.output_layout,
            )
        ]

    def reset(self) -> None:
        """Restore red as the initial color."""
        self.state.blue = False


@dataclass(slots=True)
class ColorToggleUIState:
    """UI state containing the model loop that receives key signals."""

    model_loop: IModelLoop[ColorToggleModelState]
    """Model loop whose state is updated through :func:`invoke_async`."""

    instructions: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy instruction text."""


class ColorToggleSlangPyUILoop(SlangPyUILoop[ColorToggleUIState]):
    """Send ``W`` key presses from SlangPy UI to the model loop."""

    def step_ui(
        self,
        ui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Queue one model color toggle for every ``W`` press."""
        del step_index
        if self.state.instructions is None:
            window = ui.Window(
                ui.screen,
                "invoke_async",
                position=(16, 16),
                size=(300, 80),
            )
            self.state.instructions = ui.Text(window, "Press W to toggle red / blue")

        for event in events.get_events():
            data = event.get_event_data()
            if (
                isinstance(data, KeyboardUserInputEventData)
                and data.state is KeyboardInputState.PRESSED
                and data.key.lower() == "w"
            ):
                invoke_async(self.state.model_loop, lambda state: state._toggle_color())
        return self.presented_model_frame()


class ColorToggleSession(ISession):
    """Connect a SlangPy keyboard UI to a color-generating model loop."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one color-toggle session.

        Args:
            session_desc: Output dimensions and loop frequencies.
            device: Device used for model output.

        Raises:
            ValueError: ``session_desc`` does not request ``tchw`` output.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The invoke-async demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._device = device

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the model loop before wiring it into the UI state."""
        model_loop = self.register_model_loop(
            ColorToggleModelLoop,
            state=ColorToggleModelState(
                session_desc=self._session_desc,
                device=self._device,
            ),
        )
        self.register_ui_loop(
            ColorToggleSlangPyUILoop,
            state=ColorToggleUIState(model_loop=model_loop),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )


class ColorToggleApplication(IApplication):
    """Create SlangPy UI sessions that signal color toggles to the model loop."""

    def __init__(self, *, device: torch.device | str = "cuda") -> None:
        """Configure the device used by color-toggle sessions."""
        self._device = device

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The invoke-async demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return SessionDesc(video_width=640, video_height=480)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized color-toggle session."""
        return ColorToggleSession(session_desc, device=self._device)


def _color_frame(state: ColorToggleModelState) -> Tensor:
    color = (-1.0, -1.0, 1.0) if state.blue else (1.0, -1.0, -1.0)
    return (
        torch.tensor(color, dtype=torch.float32, device=state.device)
        .view(1, 3, 1, 1)
        .expand(
            1,
            3,
            state.session_desc.video_height,
            state.session_desc.video_width,
        )
    )


def create_app() -> IApplication:
    """Return a new color-toggle application."""
    return ColorToggleApplication()
