# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable camera and auxiliary-sensor grid for Interactive Drive integrations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .app import InteractiveDriveUILoop, InteractiveDriveUIState
from .config import PresentationLayout


@dataclass(frozen=True, slots=True)
class CameraView:
    """Camera identity and caption, in the model's channel order."""

    name: str
    label: str
    yaw_degrees: float = 0.0
    caption: str = ""
    fov_degrees: float = 90.0


def validate_cameras(cameras: Sequence[CameraView]) -> None:
    if not cameras or len({c.name for c in cameras}) != len(cameras):
        raise ValueError("Camera streams need unique names and at least one camera")


@dataclass(frozen=True, slots=True)
class AuxiliaryView:
    """A non-camera stream appended after the camera channels."""

    name: str
    label: str
    caption: str = ""


@dataclass(slots=True)
class MultiviewUIState(InteractiveDriveUIState):
    """Camera metadata and presentation selection alongside shared drive state."""

    cameras: tuple[CameraView, ...] = ()
    """Camera metadata in model-channel order."""
    visible_count: int = 7
    """Number of cameras shown in grid mode."""
    focus_channel: int = -1
    """Single channel to enlarge, or -1 for the grid layout."""
    auxiliary_views: tuple[AuxiliaryView, ...] = ()
    """Additional synchronized streams, shown in the same grid or individually focused."""
    preview: bool = False
    """Identify synthetic output explicitly in the HUD."""


def visible_channels(cameras: tuple[CameraView, ...], count: int) -> tuple[int, ...]:
    """Preserve camera rig order when selecting a prefix for display."""
    validate_cameras(cameras)
    if not 1 <= count <= len(cameras):
        raise ValueError("Visible count must lie within the available camera count")
    return tuple(range(count))


class MultiviewUILoop(InteractiveDriveUILoop):
    """Render one model channel per camera, preserving camera/stream identity.

    Register with ``MultiviewUIState``. Model loops return synchronized TCHW
    channels and publish ``DriveTelemetry`` with ``invoke_async``. UI selection
    never changes the generated channel order.
    """

    state: MultiviewUIState
    """UI-thread camera selection and inherited driving HUD state."""

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Present labeled camera and auxiliary channels in the shared rig grid."""
        del step_index
        state = self.state
        self._apply_inputs(events)
        width, height = self.get_ui_loop_size()
        sidebar = min(320.0, width * 0.4)
        area_width = width - sidebar
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
        )
        self._panel(imgui, "Sensor controls", area_width, 0, sidebar, height, flags)
        try:
            imgui.text(state.title.upper())
            changed, focus = imgui.combo(
                "Focus",
                state.focus_channel + 1,
                ["Grid"]
                + [c.label for c in state.cameras]
                + [v.label for v in state.auxiliary_views],
            )
            if changed:
                state.focus_channel = focus - 1
            if state.focus_channel >= 0:
                if imgui.button("Return to grid"):
                    state.focus_channel = -1
            else:
                _, state.visible_count = imgui.slider_int(
                    "Camera views", state.visible_count, 1, len(state.cameras)
                )
            imgui.text_wrapped("Display only; all configured streams keep generating.")
            self._draw_view_controls(imgui)
            imgui.separator()
            self._draw_drive_hud(imgui)
            if imgui.button("Restart rollout"):
                self._restart()
            imgui.text_wrapped(state.status)
            self._draw_help(imgui)
            imgui.text_wrapped(
                "Cameras follow rig order. Hover a title for its full caption."
            )
        finally:
            imgui.end()
        self._panel(
            imgui,
            "Sensor background",
            0,
            0,
            area_width,
            height,
            flags | imgui.WindowFlags_.no_inputs,
        )
        imgui.end()
        metadata = (*state.cameras, *state.auxiliary_views)
        channels = (
            (state.focus_channel,)
            if state.focus_channel >= 0
            else (
                *visible_channels(state.cameras, state.visible_count),
                *range(len(state.cameras), len(metadata)),
            )
        )
        layout = PresentationLayout.grid(len(channels))
        for cell, channel in enumerate(channels):
            view = metadata[channel]
            row, column = divmod(cell, layout.columns)
            w, h = area_width / layout.columns, height / layout.rows
            self._panel(imgui, f"Stream##{view.name}", column * w, row * h, w, h, flags)
            try:
                imgui.text(view.label)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"{view.name}\n{view.caption}")
                caption = h >= 8 * imgui.get_text_line_height_with_spacing()
                self._draw_stream(
                    imgui,
                    channel,
                    f"stream-{view.name}",
                    "Waiting for sensor stream...",
                    reserve_caption=caption,
                )
                if caption:
                    imgui.text(" ".join(view.caption.splitlines()))
            finally:
                imgui.end()
        return None

    def _draw_stream(
        self,
        imgui: Any,
        channel: int,
        key: str,
        waiting: str,
        *,
        reserve_caption: bool = True,
    ) -> None:
        """Draw any synchronized image channel with its original aspect ratio."""
        try:
            frame = self.presented_model_frame(channel)
        except IndexError:
            frame = None
        if frame is None:
            imgui.text_wrapped(waiting)
            return
        if frame.is_floating_point():
            frame = ((frame.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        available = imgui.get_content_region_avail()
        width = max(1.0, available.x)
        caption_height = (
            imgui.get_text_line_height_with_spacing() if reserve_caption else 0
        )
        height = max(1.0, available.y - caption_height)
        scale = min(width / frame.shape[-1], height / frame.shape[-2])
        imgui.image(
            key,
            frame.permute(1, 2, 0),
            size=(frame.shape[-1] * scale, frame.shape[-2] * scale),
        )

    def _draw_view_controls(self, imgui: Any) -> None:
        """Allow application-specific display controls in the shared sidebar."""

    def _apply_inputs(self, events: UserInputEvents) -> None:
        del events

    def _draw_drive_hud(self, imgui: Any) -> None:
        """Allow a host application to supply its own controls."""

    def _draw_help(self, imgui: Any) -> None:
        imgui.text_wrapped(
            "Focus enlarges a camera or sensor without changing generation."
        )

    @staticmethod
    def _panel(
        imgui: Any,
        title: str,
        x: float,
        y: float,
        width: float,
        height: float,
        flags: Any,
    ) -> None:
        imgui.set_next_window_pos(imgui.ImVec2(x + 4, y + 4), imgui.Cond_.always)
        imgui.set_next_window_size(
            imgui.ImVec2(max(1, width - 8), max(1, height - 8)), imgui.Cond_.always
        )
        imgui.begin(title, flags=flags)

    def reset(self) -> None:
        """Clear telemetry while preserving camera and sensor selection."""
        super().reset()
        self.state.telemetry = None
