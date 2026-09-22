# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImGui window and render-target resize application for the v2 loop runtime."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .text_input_app import BackgroundModelLoop


@dataclass(frozen=True, slots=True)
class ScheduledResize:
    """One window or UI resize scheduled from the first UI frame."""

    after_ui_loops: int
    """Number of UI loops to complete before requesting the resize."""

    size: tuple[int, int]
    """Requested ``(width, height)`` in pixels."""

    resize_ui: bool = False
    """Whether to resize the UI render target instead of the presenter."""


@dataclass(slots=True)
class WindowSizeState:
    """Editable window or UI render-target dimensions."""

    new_x: int = 800
    """Requested window width in pixels."""

    new_y: int = 600
    """Requested window height in pixels."""

    scheduled_resizes: tuple[ScheduledResize, ...] = ()
    """Programmatic resizes ordered by UI loop count."""

    next_scheduled_resize: int = 0
    """Index of the next programmatic resize to request."""


class WindowSizeImGuiUILoop(ImGuiUILoop[WindowSizeState]):
    """Resize the native window or UI render target from two integer inputs."""

    def step_ui(
        self,
        imgui: Any,
        step_index: int,
        events: UserInputEvents,
    ) -> Tensor | None:
        """Draw size inputs and queue a resize when requested."""
        del events
        if self.state.next_scheduled_resize < len(self.state.scheduled_resizes):
            scheduled_resize = self.state.scheduled_resizes[
                self.state.next_scheduled_resize
            ]
            if step_index >= scheduled_resize.after_ui_loops:
                self.state.new_x, self.state.new_y = scheduled_resize.size
                if scheduled_resize.resize_ui:
                    self.resize_ui_loop(*scheduled_resize.size)
                else:
                    self.request_new_window_size(scheduled_resize.size)
                self.state.next_scheduled_resize += 1

        imgui.set_next_window_pos(imgui.ImVec2(16.0, 16.0), imgui.Cond_.once)
        imgui.set_next_window_size(imgui.ImVec2(360.0, 175.0), imgui.Cond_.once)
        imgui.begin("Window size")
        try:
            _, self.state.new_x = imgui.input_int("New X", self.state.new_x)
            _, self.state.new_y = imgui.input_int("New Y", self.state.new_y)
            if self.state.new_x <= 0 or self.state.new_y <= 0:
                imgui.text("New X and New Y must be positive.")
            elif imgui.button("Resize window"):
                self.request_new_window_size((self.state.new_x, self.state.new_y))
            elif imgui.button("Resize UI"):
                self.resize_ui_loop(self.state.new_x, self.state.new_y)
        finally:
            imgui.end()
        return self.presented_model_frame()

    def reset(self) -> None:
        """Reset the renderer for a new session generation."""
        super().reset()


class WindowSizeSession(ISession):
    """Run the ImGui window-size controls over a generated background."""

    def __init__(
        self,
        session_desc: SessionDesc,
        *,
        device: torch.device | str = "cuda",
        scheduled_resizes: tuple[ScheduledResize, ...] = (),
    ) -> None:
        """Configure one window-size session.

        Args:
            session_desc: Output dimensions and loop frequencies.
            device: Device used for the background model frame.
            scheduled_resizes: Programmatic window or UI resizes.

        Raises:
            ValueError: The output layout is not ``tchw``.
        """
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                "The window-size demo requires tchw output, got "
                f"{session_desc.output_layout.value}."
            )
        self._session_desc = session_desc
        self._device = device
        self._scheduled_resizes = scheduled_resizes

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved session description."""
        return self._session_desc

    def init(self) -> None:
        """Register the resize UI and background model loops."""
        self.register_ui_loop(
            WindowSizeImGuiUILoop,
            state=WindowSizeState(scheduled_resizes=self._scheduled_resizes),
            width=self._session_desc.video_width,
            height=self._session_desc.video_height,
        )
        self.register_model_loop(
            BackgroundModelLoop,
            state=(self._session_desc, self._device),
        )


class WindowSizeApplication(IApplication):
    """Create ImGui window and render-target resize sessions."""

    def __init__(self) -> None:
        self._scheduled_resizes: tuple[ScheduledResize, ...] = ()

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse optional programmatic window and UI resizes."""
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 imgui-ui-window-size --",
            description="Request window or UI resizes manually or on a schedule.",
        )
        parser.add_argument(
            "--resize-after-ui-loops",
            action="append",
            default=[],
            type=_non_negative_ui_loops,
            metavar="N",
            help="Completed UI loops before a resize. Repeat with --resize-to-size.",
        )
        parser.add_argument(
            "--resize-ui-after-ui-loops",
            action="append",
            default=[],
            type=_non_negative_ui_loops,
            metavar="N",
            help=(
                "Completed UI loops before a UI resize. Repeat with --resize-to-size."
            ),
        )
        parser.add_argument(
            "--resize-to-size",
            action="append",
            default=[],
            type=_window_size,
            metavar="WIDTHxHEIGHT",
            help="Size paired with the preceding resize schedule. Repeatable.",
        )
        raw_args = list(commandline_args)
        schedule_options = _require_adjacent_resize_pairs(raw_args, parser)
        parsed = parser.parse_args(raw_args)
        resize_count = len(parsed.resize_after_ui_loops) + len(
            parsed.resize_ui_after_ui_loops
        )
        if resize_count != len(parsed.resize_to_size):
            parser.error(
                "Resize schedules and --resize-to-size must be supplied the same "
                "number of times."
            )
        window_offsets = iter(parsed.resize_after_ui_loops)
        ui_offsets = iter(parsed.resize_ui_after_ui_loops)
        self._scheduled_resizes = tuple(
            sorted(
                (
                    ScheduledResize(
                        after_ui_loops=(
                            next(ui_offsets)
                            if option == "--resize-ui-after-ui-loops"
                            else next(window_offsets)
                        ),
                        size=size,
                        resize_ui=option == "--resize-ui-after-ui-loops",
                    )
                    for option, size in zip(
                        schedule_options,
                        parsed.resize_to_size,
                        strict=True,
                    )
                ),
                key=lambda resize: resize.after_ui_loops,
            )
        )

    def session_desc(self) -> SessionDesc:
        """Return the demo's fixed rendering dimensions and rates."""
        return SessionDesc(video_width=640, video_height=480)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized window-size session."""
        return WindowSizeSession(
            session_desc,
            scheduled_resizes=self._scheduled_resizes,
        )


def create_app() -> IApplication:
    """Return a new window-size application."""
    return WindowSizeApplication()


def _non_negative_ui_loops(value: str) -> int:
    """Parse a non-negative UI loop count."""
    try:
        ui_loops = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("N must be an integer.") from error
    if ui_loops < 0:
        raise argparse.ArgumentTypeError("N must be >= 0.")
    return ui_loops


def _require_adjacent_resize_pairs(
    arguments: list[str],
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    """Require every schedule offset to be followed by its size option."""
    schedule_options = (
        "--resize-after-ui-loops",
        "--resize-ui-after-ui-loops",
    )
    paired_schedule_options: list[str] = []
    paired_size_indices: set[int] = set()
    for index, argument in enumerate(arguments):
        if argument in schedule_options:
            size_option_index = index + 2
            if (
                size_option_index >= len(arguments)
                or arguments[size_option_index] != "--resize-to-size"
            ):
                parser.error(
                    f"{argument} N must be immediately followed by "
                    "--resize-to-size WIDTHxHEIGHT."
                )
            paired_schedule_options.append(argument)
            paired_size_indices.add(size_option_index)
        elif argument == "--resize-to-size" and index not in paired_size_indices:
            parser.error(
                "--resize-to-size WIDTHxHEIGHT must immediately follow "
                "a resize schedule."
            )
    return tuple(paired_schedule_options)


def _window_size(value: str) -> tuple[int, int]:
    """Parse a positive ``WIDTHxHEIGHT`` target size."""
    dimensions = value.lower().split("x")
    if len(dimensions) != 2:
        raise argparse.ArgumentTypeError("Size must use WIDTHxHEIGHT.")
    try:
        width, height = (int(dimension) for dimension in dimensions)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "WIDTH and HEIGHT must be integers."
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be > 0.")
    return width, height
