# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImGui application displaying output produced by its model thread."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.api_v2.thread import IThread
from flashdreams.runtime_v2.imgui_thread import ImGUIThread
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .common import DEFAULT_SESSION_DESC, require_tchw

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
    """Model-channel selection owned by the UI thread."""

    image_size: tuple[float, float]
    """Width and height used inside the ImGui window."""

    channel_index: int = 0
    """Index of the model result channel drawn in the window."""


class ModelOutputImGUIThread(ImGUIThread[ModelOutputUIState]):
    """Draw the newest model-thread frame in an ImGui window."""

    def draw_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> None:
        """Draw the latest model output when one is available."""
        del step_index, events
        image_width, image_height = self.state.image_size
        imgui.set_next_window_pos((16, 16), imgui.Cond_.once)
        imgui.set_next_window_size(
            (image_width + 32, image_height + 64),
            imgui.Cond_.once,
        )
        imgui.begin("Model output channels")
        changed, channel_index = imgui.combo(
            "Channel",
            self.state.channel_index,
            _CHANNEL_NAMES,
        )
        if changed:
            self.state.channel_index = channel_index
        if not self.draw_presented_model_frame(
            self.state.channel_index,
            image_width,
            image_height,
        ):
            imgui.text("Waiting for model generation...")
        imgui.end()


@dataclass(frozen=True, slots=True)
class ModelOutputState:
    """Output configuration owned by the model thread."""

    session_desc: SessionDesc
    """Output layout and dimensions for generated frames."""

    device: torch.device | str
    """Device used for model output."""


class ModelOutputThread(IThread[ModelOutputState]):
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
        use_imgui: bool = True,
    ) -> None:
        """Configure one model-output session.

        Args:
            session_desc: Output dimensions and thread frequencies.
            device: Device used for model output.
            use_imgui: Whether to register the channel-selecting ImGui UI.
        """
        require_tchw(session_desc, "model-output")
        self._session_desc = session_desc
        self._device = device
        self._use_imgui = use_imgui

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the frame-displaying UI and model threads."""
        if self._use_imgui:
            self.register_ui_thread(
                ModelOutputImGUIThread,
                state=ModelOutputUIState(image_size=_image_size(self._session_desc)),
                width=self._session_desc.video_width,
                height=self._session_desc.video_height,
            )
        self.register_model_thread(
            ModelOutputThread,
            state=ModelOutputState(
                session_desc=self._session_desc,
                device=self._device,
            ),
        )


class ModelOutputApplication(IApplication):
    """Create sessions that display model output through ImGui."""

    def __init__(self, *, device: torch.device | str = "cuda") -> None:
        """Configure the device used by model-output sessions."""
        self._device = device
        self._use_imgui = True

    def init(self, commandline_args: Sequence[str]) -> None:
        """Select ImGui or the session's default model-output blitter."""
        parser = argparse.ArgumentParser(prog="imgui-model-output")
        parser.add_argument(
            "--no-ui",
            action="store_true",
            help="Omit UI registration and use the default all-channel blitter.",
        )
        self._use_imgui = not parser.parse_args(list(commandline_args)).no_ui

    def session_desc(self) -> SessionDesc:
        """Return the demo's established dimensions and rates."""
        return DEFAULT_SESSION_DESC

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized model-output session."""
        return ModelOutputSession(
            session_desc,
            device=self._device,
            use_imgui=self._use_imgui,
        )


def _image_size(session_desc: SessionDesc) -> tuple[float, float]:
    width = session_desc.video_width
    height = session_desc.video_height
    scale = min(max(1, width - 64) / width, max(1, height - 96) / height)
    return width * scale, height * scale


def create_app() -> IApplication:
    """Return a new model-output application."""
    return ModelOutputApplication()
