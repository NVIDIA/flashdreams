# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bearing-aware camera and auxiliary-sensor viewer for Interactive Drive."""

from __future__ import annotations

import math
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
    if any(not math.isfinite(c.yaw_degrees) for c in cameras):
        raise ValueError("Camera bearings must be finite")


def sensor_layout(
    cameras: Sequence[CameraView], auxiliary_count: int = 0, *, bearings: bool = True
) -> dict[int, tuple[float, float, float, float]]:
    """Return normalized sensor rectangles keyed by original channel index.

    Bearing rows run front to rear. Left/right streams stay in their respective
    screen halves; co-located lenses remain distinct. Auxiliary streams occupy
    the middle row between the sides. One camera and one auxiliary stream use
    equal columns. Grid mode preserves model order.
    """
    validate_cameras(cameras)
    if auxiliary_count < 0:
        raise ValueError("Auxiliary count must be non-negative")
    count = len(cameras) + auxiliary_count
    if not bearings or count == 1 or (len(cameras) == 1 and auxiliary_count == 1):
        grid = PresentationLayout.grid(count)
        return {
            i: (
                i % grid.columns / grid.columns,
                i // grid.columns / grid.rows,
                1 / grid.columns,
                1 / grid.rows,
            )
            for i in range(count)
        }
    rows: list[list[list[int]]] = [[[], [], []] for _ in range(3)]
    for i, camera in enumerate(cameras):
        yaw = (camera.yaw_degrees + 180) % 360 - 180
        row = 0 if abs(yaw) <= 45 else 1 if abs(yaw) < 120 else 2
        side = 1 if abs(yaw) in (0, 180) else 0 if yaw > 0 else 2
        rows[row][side].append(i)
    # A one-sided four-camera rig fits two rows without reserving an empty side.
    for side in (0, 2):
        opposite = 2 - side
        if (
            len(cameras) == 4
            and auxiliary_count <= 1
            and [len(g) for g in rows[0]] == [0, 1, 0]
            and len(rows[1][side]) == len(rows[2][side]) == len(rows[2][1]) == 1
            and not rows[1][opposite]
            and not rows[2][opposite]
        ):
            front, cross, rear, corner = (
                rows[0][1][0],
                rows[1][side][0],
                rows[2][1][0],
                rows[2][side][0],
            )
            split = 2 / 3 if auxiliary_count else 0.5
            result = {
                front: (0, 0, split, 0.5),
                cross: (split, 0, 1 - split, 0.5),
                rear: (0, 0.5, 1 - split, 0.5),
                corner: (split, 0.5, 1 - split, 0.5),
            }
            if auxiliary_count:
                result[4] = (1 / 3, 0.5, 1 / 3, 0.5)
            if side == 0:
                result = {i: (1 - x - w, y, w, h) for i, (x, y, w, h) in result.items()}
            return result
    for row in rows:
        for group in row:
            group.sort(
                key=lambda i: (
                    -(cameras[i].yaw_degrees + 180) % 360,
                    cameras[i].fov_degrees,
                    cameras[i].name,
                )
            )
    rows[1][1] = list(range(len(cameras), count))
    rows = [row for row in rows if any(row)]
    rectangles = {}
    weights = []
    for left, center, right in rows:
        if center and center[0] >= len(cameras) and (left or right):
            weights.append(max(len(left), len(center), len(right)) / 3)
        else:
            weights.append(1 / max(1, 2 * max(len(left), len(right)) + len(center)))
    total = sum(weights)
    y = 0.0
    for (left, center, right), weight in zip(rows, weights, strict=True):
        height = weight / total
        if center and center[0] >= len(cameras) and (left or right):
            for column, group in enumerate((left, center, right)):
                for j, channel in enumerate(group):
                    rectangles[channel] = (
                        column / 3,
                        y + j * height / len(group),
                        1 / 3,
                        height / len(group),
                    )
        else:
            columns = max(1, 2 * max(len(left), len(right)) + len(center))
            for group, start in (
                (left, 0),
                (center, (columns - len(center)) / 2),
                (right, columns - len(right)),
            ):
                for j, channel in enumerate(group):
                    rectangles[channel] = (
                        (start + j) / columns,
                        y,
                        1 / columns,
                        height,
                    )
        y += height
    return rectangles


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
    visible_count: int = 0
    """Number of cameras shown in overview mode; zero initializes to the full rig."""
    focus_channel: int = -1
    """Single channel to enlarge, or -1 for the overview layout."""
    auxiliary_views: tuple[AuxiliaryView, ...] = ()
    """Configured sensors, always shown in the overview or individually focused."""
    bearing_layout: bool = True
    """Use physical bearing rows; false retains the original model-order grid."""
    preview: bool = False
    """Identify synthetic output explicitly in the HUD."""

    def __post_init__(self) -> None:
        if self.visible_count == 0:
            self.visible_count = len(self.cameras)


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
        """Present labeled sensor channels and shared display controls."""
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
            imgui.text_wrapped(state.title.upper())
            imgui.text(
                "SENSOR OVERVIEW" if state.focus_channel < 0 else "FOCUSED SENSOR"
            )
            imgui.separator()
            imgui.set_next_item_width(-1)
            changed, focus = imgui.combo(
                "##Focus",
                state.focus_channel + 1,
                ["Overview"]
                + [c.label for c in state.cameras]
                + [v.label for v in state.auxiliary_views],
            )
            if changed:
                state.focus_channel = focus - 1
            if state.focus_channel >= 0:
                if imgui.button("Return to overview"):
                    state.focus_channel = -1
            else:
                changed, layout_index = imgui.combo(
                    "Layout",
                    0 if state.bearing_layout else 1,
                    ["Bearings", "Grid (model order)"],
                )
                if changed:
                    state.bearing_layout = layout_index == 0
            imgui.text(f"Cameras: {len(state.cameras)}")
            self._draw_view_controls(imgui)
            imgui.separator()
            self._draw_drive_hud(imgui)
            if imgui.button("Restart rollout"):
                self._restart()
            imgui.text_wrapped(state.status)
            self._draw_help(imgui)
            imgui.separator()
            imgui.text_wrapped(
                "Click a sensor title to enlarge. Return to overview to restore coverage."
            )
            imgui.text_wrapped(
                f"Camera left / {state.auxiliary_views[0].label} right. Images are not mirrored."
                if state.visible_count == 1 and len(state.auxiliary_views) == 1
                else "Front above / rear below. Left and right are vehicle-relative. Images are not mirrored."
                if state.bearing_layout
                else "Grid follows model channel order."
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
        if state.focus_channel >= 0:
            rectangles = {state.focus_channel: (0, 0, 1, 1)}
        else:
            rectangles = sensor_layout(
                state.cameras[: state.visible_count],
                len(state.auxiliary_views),
                bearings=state.bearing_layout,
            )
            rectangles = {channels[i]: rect for i, rect in rectangles.items()}
        for channel in channels:
            view = metadata[channel]
            x, y, w, h = rectangles[channel]
            x, y, w, h = x * area_width, y * height, w * area_width, h * height
            self._panel(imgui, f"Stream##{channel}", x, y, w, h, flags)
            try:
                if imgui.selectable(f"{view.label}  [+]##focus-{channel}", False)[0]:
                    state.focus_channel = (
                        channel if state.focus_channel != channel else -1
                    )
                if imgui.is_item_hovered():
                    bearing = (
                        f" | yaw {view.yaw_degrees:+g} deg | FOV {view.fov_degrees:g} deg"
                        if isinstance(view, CameraView)
                        else ""
                    )
                    imgui.set_tooltip(
                        f"{view.name}{bearing}\n{view.caption}\nClick to toggle focus"
                    )
                caption = state.focus_channel == channel
                self._draw_stream(
                    imgui,
                    channel,
                    f"stream-{channel}",
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
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + (width - frame.shape[-1] * scale) / 2
        )
        imgui.set_cursor_pos_y(
            imgui.get_cursor_pos_y() + (height - frame.shape[-2] * scale) / 2
        )
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
