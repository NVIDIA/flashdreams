# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy UI application displaying model-loop output."""

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

_COLORS = (
    (1.0, -1.0, -1.0, 1.0),
    (-1.0, 1.0, -1.0, 0.5),
    (-1.0, -1.0, 1.0, 0.5),
)
"""Normalized RGBA colors emitted as separate model result layers."""

_CHANNEL_NAMES = ("Red", "Green", "Blue")
"""Labels for the model result channels selectable in the UI."""

_FADE_FRAME_COUNT = 60
"""Frames in each full-intensity-to-black model chunk."""


@dataclass(slots=True)
class ModelOutputUIState:
    """Model-channel selection owned by the UI loop."""

    channel_index: int = 0
    """Index of the model result channel composited beneath the UI."""

    selector: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy channel selector."""

    status: Any | None = field(default=None, init=False, repr=False)
    """Retained SlangPy waiting-status text."""


class ModelOutputSlangPyUILoop(SlangPyUILoop[ModelOutputUIState]):
    """Draw the newest model-loop frame in a SlangPy window."""

    def step_ui(
        self,
        ui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw the latest model output when one is available."""
        del step_index, events
        if self.state.selector is None:
            window = ui.Window(
                ui.screen,
                "Model output channels",
                position=(16, 16),
                size=(300, 100),
            )
            self.state.selector = ui.ComboBox(
                window,
                "Channel",
                self.state.channel_index,
                self._select_channel,
                _CHANNEL_NAMES,
            )
            self.state.status = ui.Text(window, "Waiting for model generation...")
        frame = self.presented_model_frame(self.state.channel_index)
        assert self.state.status is not None
        self.state.status.visible = frame is None
        return frame

    def _select_channel(self, channel_index: int) -> None:
        self.state.channel_index = channel_index


@dataclass(frozen=True, slots=True)
class ModelOutputState:
    """Output configuration owned by the model loop."""

    session_desc: SessionDesc
    """Output layout and dimensions for generated frames."""

    device: torch.device | str
    """Device used for model output."""


class ModelOutputLoop(IModelLoop[ModelOutputState]):
    """Generate one independently selectable fade chunk per color."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return one repeating fade chunk per selectable color."""
        del events
        desc = self.state.session_desc
        intensity = torch.linspace(
            1.0,
            0.0,
            _FADE_FRAME_COUNT,
            dtype=torch.float32,
            device=self.state.device,
        ).view(_FADE_FRAME_COUNT, 1, 1, 1)
        black = torch.full(
            (1, 3, 1, 1),
            -1.0,
            dtype=torch.float32,
            device=self.state.device,
        )
        return [
            StepResult(
                step_index=step_index,
                output=torch.cat(
                    (
                        (
                            black
                            + (
                                torch.tensor(
                                    color[:3],
                                    dtype=torch.float32,
                                    device=self.state.device,
                                ).view(1, 3, 1, 1)
                                - black
                            )
                            * intensity
                        ).expand(
                            _FADE_FRAME_COUNT,
                            3,
                            desc.video_height,
                            desc.video_width,
                        ),
                        torch.full(
                            (
                                _FADE_FRAME_COUNT,
                                1,
                                desc.video_height,
                                desc.video_width,
                            ),
                            color[3],
                            dtype=torch.float32,
                            device=self.state.device,
                        ),
                    ),
                    dim=1,
                ),
                frame_count=_FADE_FRAME_COUNT,
                output_layout=desc.output_layout,
            )
            for color in _COLORS
        ]

    def reset(self) -> None:
        return


class ModelOutputSession(ISession):
    """Display a selected channel from each model-generated chunk."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        """Configure one model-output session.

        Args:
            session_desc: Output dimensions and loop frequencies.
            device: Device used for model output.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The model-output demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._device = device

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the frame-displaying UI and model loops."""
        self.register_ui_loop(
            ModelOutputSlangPyUILoop,
            state=ModelOutputUIState(),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_model_loop(
            ModelOutputLoop,
            state=ModelOutputState(
                session_desc=self._session_desc,
                device=self._device,
            ),
        )


class ModelOutputApplication(IApplication):
    """Create sessions that display model output through SlangPy UI."""

    def __init__(self, *, device: torch.device | str = "cuda") -> None:
        """Configure the device used by model-output sessions."""
        self._device = device

    def init(self, commandline_args: Sequence[str]) -> None:
        """Reject application-specific arguments."""
        if commandline_args:
            raise ValueError("The model-output demo takes no application arguments.")

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return SessionDesc(video_width=640, video_height=480)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized model-output session."""
        return ModelOutputSession(
            session_desc,
            device=self._device,
        )


def create_app() -> IApplication:
    """Return a new model-output application."""
    return ModelOutputApplication()
