# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live driving controls and camera streams for the shared sensor viewer."""

from __future__ import annotations

import math
from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch
from interactive_drive.app import (
    InteractiveDriveSession,
    InteractiveDriveUIState,
    InteractiveDriveUILoop,
)
from interactive_drive.config import PresentationLayout
from interactive_drive.core import (
    InteractiveDriveModelLoop,
    InteractiveDriveModelState,
    _frame_chunk_tensor,
    _output_cuda_device,
)
from interactive_drive.multiview import AuxiliaryView, MultiviewUIState
from interactive_drive.types import FrameChunk
from torch import Tensor

from cosmos3_nano_sim_auto.impl.ui import camera_views_for_names
from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop


class MultiviewDriveModelLoop(InteractiveDriveModelLoop):
    """Present each camera and optional LiDAR stream in its own channel."""

    def _frame_chunk_tensors(self, chunk: FrameChunk) -> list[Tensor]:
        state = self.state
        if state.config.no_ui:
            return super()._frame_chunk_tensors(chunk)
        if not chunk.frames:
            return []
        return [
            _frame_chunk_tensor(
                chunk,
                state.view_mode,
                layout=PresentationLayout(1, 1, view),
                output_device=_output_cuda_device(state.config.app),
            )
            for view in range(chunk.frames[0].views)
        ]


class MultiviewDriveUILoop(ImGuiUILoop[MultiviewUIState]):
    """Keep live controls active while the viewer arranges or focuses sensors."""

    state: MultiviewUIState
    """Live controls and stable sensor-channel identities."""

    _restart = InteractiveDriveUILoop._restart
    _toggle_view = InteractiveDriveUILoop._toggle_view
    _draw_scene_controls = InteractiveDriveUILoop._draw_scene_controls
    _progress = staticmethod(InteractiveDriveUILoop._progress)

    def __init__(self, **kwargs: Any) -> None:
        """Use the proportional font already bundled with the UI dependency."""
        super().__init__(**kwargs)
        package = find_spec("imgui_bundle")
        if package is not None and package.origin is not None:
            font = (
                Path(package.origin).parent / "assets/fonts/Roboto/Roboto-Regular.ttf"
            )
            if font.is_file():
                self.renderer.font_path = str(font)

    @staticmethod
    def _text(
        imgui: Any, text: str, x: float, y: float, size: float, color: int = 0xFFF0EAE4
    ) -> None:
        imgui.get_window_draw_list().add_text(
            imgui.get_font(), size, imgui.ImVec2(x, y), color, text
        )

    @staticmethod
    def _fill_image(
        imgui: Any, key: str, pixels: Any, width: float, height: float
    ) -> None:
        """Fill a tile with a centered crop, preserving the image's aspect ratio."""
        ratio = pixels.shape[1] / pixels.shape[0]
        target = width / height
        u = max(0.0, (1 - target / ratio) / 2)
        v = max(0.0, (1 - ratio / target) / 2)
        imgui.image(
            key,
            pixels,
            size=(width, height),
            uv0=imgui.ImVec2(u, v),
            uv1=imgui.ImVec2(1 - u, 1 - v),
        )

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Present a full-canvas driving view with a compact instrument rail."""
        del step_index
        self._apply_inputs(events)
        state = self.state
        width, height = self.get_ui_loop_size()
        scale = max(0.5, min(width / 1600, height / 900))
        gap, header = 12 * scale, 68 * scale
        rail = width * 0.235
        left_width = width - rail - 3 * gap
        metadata = (*state.cameras, *state.auxiliary_views)
        front = next(
            (
                i
                for i, c in enumerate(state.cameras)
                if c.name == "camera_front_wide_120fov"
            ),
            0,
        )
        main = state.focus_channel if state.focus_channel >= 0 else front
        others = [i for i in range(len(metadata)) if i != main]
        strip = (
            min(height * 0.21, left_width / max(1, len(others)) * 480 / 832)
            if others
            else 0
        )
        bottom = 27 * scale
        main_height = height - header - strip - bottom - 2 * gap
        rail_x = left_width + 2 * gap
        controls_height = min(height * 0.35, 300 * scale)
        colors = (
            (imgui.Col_.window_bg, (0.035, 0.047, 0.063, 1)),
            (imgui.Col_.text, (0.9, 0.93, 0.96, 1)),
            (imgui.Col_.button, (0.10, 0.14, 0.18, 1)),
            (imgui.Col_.button_hovered, (0.17, 0.25, 0.30, 1)),
            (imgui.Col_.button_active, (0.15, 0.40, 0.40, 1)),
            (imgui.Col_.frame_bg, (0.12, 0.16, 0.20, 1)),
            (imgui.Col_.popup_bg, (0.065, 0.085, 0.11, 1)),
        )
        for color, rgba in colors:
            imgui.push_style_color(color, imgui.ImVec4(*rgba))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0)
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 7 * scale)
        imgui.set_next_window_pos(imgui.ImVec2(0, 0), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(width, height), imgui.Cond_.always)
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.begin("Driving cockpit", flags=flags)
        try:
            imgui.get_style().font_scale_main = max(0.8, scale)
            self._text(imgui, "FLASHDREAMS", gap, 20 * scale, 23 * scale)
            self._text(
                imgui,
                f"LIVE / {len(state.cameras):02d} CAMERAS",
                206 * scale,
                24 * scale,
                14 * scale,
                0xFFC5DFC0,
            )
            for index, label in enumerate(("Front Wide", "Restart", "Settings")):
                imgui.set_cursor_pos(
                    imgui.ImVec2(width - (3 - index) * 112 * scale - gap, 14 * scale)
                )
                if imgui.button(label, imgui.ImVec2(104 * scale, 36 * scale)):
                    if index == 0:
                        state.focus_channel = -1
                        main = front
                        others = [i for i in range(len(metadata)) if i != main]
                    elif index == 1:
                        self._restart()
                    else:
                        imgui.open_popup("Drive settings")
            self._draw_camera(
                imgui,
                main,
                metadata[main].label,
                gap,
                header,
                left_width,
                main_height,
                scale,
                True,
            )
            thumb_width = (left_width - max(0, len(others) - 1) * gap) / max(
                1, len(others)
            )
            for index, channel in enumerate(others):
                self._draw_camera(
                    imgui,
                    channel,
                    metadata[channel].label,
                    gap + index * (thumb_width + gap),
                    header + main_height + gap,
                    thumb_width,
                    strip,
                    scale,
                    False,
                )
            self._draw_instruments(imgui, rail_x, header, rail, controls_height, scale)
            map_y = header + controls_height + gap
            map_height = height - map_y - bottom - gap
            draw = imgui.get_window_draw_list()
            draw.add_rect_filled(
                imgui.ImVec2(rail_x, map_y),
                imgui.ImVec2(rail_x + rail, map_y + map_height),
                0xFF19130F,
                10 * scale,
            )
            imgui.set_cursor_pos(imgui.ImVec2(rail_x, map_y))
            map_channel = len(metadata)
            try:
                map_frame = self.presented_model_frame(map_channel)
            except IndexError:
                map_frame = None
            if map_frame is not None:
                self._fill_image(
                    imgui, "drive-minimap", map_frame.permute(1, 2, 0), rail, map_height
                )
                center = imgui.ImVec2(rail_x + rail / 2, map_y + map_height / 2)
                draw.add_triangle_filled(
                    imgui.ImVec2(center.x, center.y - 9 * scale),
                    imgui.ImVec2(center.x - 6 * scale, center.y + 7 * scale),
                    imgui.ImVec2(center.x + 6 * scale, center.y + 7 * scale),
                    0xFFB4D3A7,
                )
            else:
                self._text(
                    imgui,
                    "Waiting for World Scenario",
                    rail_x + 16 * scale,
                    map_y + map_height / 2,
                    15 * scale,
                )
            self._text(
                imgui,
                "WORLD SCENARIO / FOLLOW HEADING",
                rail_x + 16 * scale,
                map_y + 16 * scale,
                13 * scale,
            )
            self._text(
                imgui,
                "W/S  DRIVE    A/D  STEER    SPACE  BRAKE    R  RESTART",
                gap,
                height - 19 * scale,
                12 * scale,
                0xFF9E8F7F,
            )
            if state.telemetry is not None and state.telemetry.warning:
                self._text(
                    imgui,
                    "OUTSIDE MAPPED AREA / R TO RESTART",
                    gap + left_width / 2,
                    height - 19 * scale,
                    12 * scale,
                    0xFF748BFF,
                )
            if imgui.begin_popup("Drive settings"):
                try:
                    self._draw_view_controls(imgui)
                    self._draw_drive_hud(imgui)
                    self._draw_help(imgui)
                    imgui.text_wrapped(state.status)
                finally:
                    imgui.end_popup()
        finally:
            imgui.end()
            imgui.pop_style_var(3)
            imgui.pop_style_color(len(colors))
        return None

    def _draw_camera(
        self,
        imgui: Any,
        channel: int,
        label: str,
        x: float,
        y: float,
        width: float,
        height: float,
        scale: float,
        primary: bool,
    ) -> None:
        """Draw an edge-to-edge sensor tile with an overlaid identity label."""
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(
            imgui.ImVec2(x, y), imgui.ImVec2(x + width, y + height), 0xFF27201A
        )
        try:
            frame = self.presented_model_frame(channel)
        except IndexError:
            frame = None
        imgui.set_cursor_pos(imgui.ImVec2(x, y))
        if frame is not None:
            if frame.is_floating_point():
                frame = ((frame.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            self._fill_image(
                imgui, f"stream-{channel}", frame.permute(1, 2, 0), width, height
            )
        else:
            self._text(
                imgui, "Waiting for camera", x + 16 * scale, y + height / 2, 16 * scale
            )
        imgui.set_cursor_pos(imgui.ImVec2(x, y))
        if imgui.invisible_button(f"camera-{channel}", imgui.ImVec2(width, height)):
            self.state.focus_channel = (
                channel if channel != self.state.focus_channel else -1
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label}: click to use as main camera")
        tag_width = min(width, (len(label) * 8 + 50) * scale)
        draw.add_rect_filled(
            imgui.ImVec2(x + 10 * scale, y + 10 * scale),
            imgui.ImVec2(x + tag_width, y + 40 * scale),
            0xDB1D1711,
            6 * scale,
        )
        self._text(imgui, label.upper(), x + 20 * scale, y + 17 * scale, 14 * scale)
        if primary:
            draw.add_rect(
                imgui.ImVec2(x, y),
                imgui.ImVec2(x + width, y + height),
                0xFFB4D3A7,
                thickness=2 * scale,
            )

    def _draw_instruments(
        self, imgui: Any, x: float, y: float, width: float, height: float, scale: float
    ) -> None:
        """Draw compact speed, steering and pedal instruments from live input."""
        state = self.state
        command = state.drive_input.command()
        telemetry = state.telemetry
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + width, y + height),
            0xFF241C15,
            10 * scale,
        )
        speed = abs(telemetry.speed_mps) * 3.6 if telemetry is not None else 0
        self._text(
            imgui, "VEHICLE", x + 20 * scale, y + 16 * scale, 13 * scale, 0xFFB8A28D
        )
        self._text(imgui, f"{speed:.0f}", x + 16 * scale, y + 40 * scale, 83 * scale)
        self._text(
            imgui, "KM/H", x + 20 * scale, y + 130 * scale, 13 * scale, 0xFFB8A28D
        )
        self._text(
            imgui,
            "R" if telemetry is not None and telemetry.reverse else "D",
            x + width - 62 * scale,
            y + 52 * scale,
            44 * scale,
            0xFFB4D3A7,
        )
        radius = 31 * scale
        center = np.array([x + 51 * scale, y + height - 66 * scale])
        draw.add_circle(imgui.ImVec2(*center), radius, 0xFFD2C3B1, thickness=4 * scale)
        angle = -command.steer * math.pi / 2
        for offset in (0, math.pi, math.pi / 2):
            end = center + radius * np.array(
                [math.cos(angle + offset), math.sin(angle + offset)]
            )
            draw.add_line(
                imgui.ImVec2(*center), imgui.ImVec2(*end), 0xFFD2C3B1, 3 * scale
            )
        self._text(
            imgui,
            f"{command.steer:+.2f}",
            x + 24 * scale,
            y + height - 23 * scale,
            12 * scale,
            0xFFB8A28D,
        )
        for index, (label, value, color) in enumerate(
            (
                ("THROTTLE", command.throttle, 0xFFB4D3A7),
                ("BRAKE", command.brake, 0xFF748BFF),
            )
        ):
            bx = x + 112 * scale
            by = y + height - (94 - index * 48) * scale
            bar_width = max(1.0, width - 134 * scale)
            self._text(
                imgui, f"{label}  {value * 100:.0f}%", bx, by, 12 * scale, 0xFFB8A28D
            )
            draw.add_rect_filled(
                imgui.ImVec2(bx, by + 20 * scale),
                imgui.ImVec2(bx + bar_width, by + 26 * scale),
                0xFF493C2F,
                3 * scale,
            )
            if value > 0:
                draw.add_rect_filled(
                    imgui.ImVec2(bx, by + 20 * scale),
                    imgui.ImVec2(bx + bar_width * value, by + 26 * scale),
                    color,
                    3 * scale,
                )

    def _apply_inputs(self, events: UserInputEvents) -> None:
        self.state.drive_input.apply(events)

    def _draw_view_controls(self, imgui: Any) -> None:
        state = self.state
        self._draw_scene_controls(imgui)
        if imgui.button(f"View: {state.view_mode.upper()}"):
            self._toggle_view()
        if state.show_postprocess_toggle:
            enabled = (
                state.telemetry.postprocess_enabled
                if state.telemetry is not None
                else state.postprocess_enabled
            )
            changed, enabled = imgui.checkbox("Post-processing", enabled)
            if changed:
                state.postprocess_enabled = bool(enabled)
                invoke_async(
                    state.model_loop,
                    lambda model_state, enabled=bool(enabled): (
                        model_state.set_postprocess_enabled(enabled)
                    ),
                )

    def _draw_drive_hud(self, imgui: Any) -> None:
        state = self.state
        command = state.drive_input.command()
        telemetry = state.telemetry
        if telemetry is None:
            imgui.text_wrapped("Waiting for the first driving block...")
        else:
            imgui.text(f"Block {telemetry.blocks_generated}")
            imgui.text(f"Speed  {abs(telemetry.speed_mps) * 3.6:5.1f} km/h")
            imgui.text(f"Gear   {'R' if telemetry.reverse else 'D'}")
            imgui.text(f"Generation  {telemetry.model_loop_ms / 1000:.2f} s")
            if telemetry.warning:
                imgui.push_style_color(
                    imgui.Col_.text, imgui.ImVec4(1.0, 0.25, 0.25, 1.0)
                )
                try:
                    imgui.text_wrapped("MAP COVERAGE WARNING")
                    imgui.text_wrapped(telemetry.warning)
                finally:
                    imgui.pop_style_color()
        imgui.text(f"Input  {state.drive_input.source()}")
        imgui.text(f"Steer  {command.steer:+.2f}")
        self._progress(imgui, "Throttle", command.throttle)
        self._progress(imgui, "Brake", command.brake)

    def reset(self) -> None:
        """Clear the vehicle marker until the restarted drive produces frames."""
        super().reset()
        self.state.telemetry = None

    def _draw_help(self, imgui: Any) -> None:
        imgui.text_wrapped(
            "W drives forward; S reverses; Space brakes; A/D steer. "
            "R restarts the rollout. Gamepads and wheels are supported. "
            "1: RGB, 2: HD map."
        )


class MultiviewDriveSession(InteractiveDriveSession):
    """Bind the live model's sensor channels to the bearing-aware viewer."""

    model_loop_type = MultiviewDriveModelLoop
    ui_loop_type = MultiviewDriveUILoop
    auxiliary_views: tuple[AuxiliaryView, ...] = ()
    """Optional sensor metadata following the camera channels."""

    def _create_ui_state(
        self, model_loop: IModelLoop[InteractiveDriveModelState]
    ) -> MultiviewUIState:
        base = super()._create_ui_state(model_loop)
        selected = self._config.app.presentation.view_index
        return MultiviewUIState(
            **{
                item.name: getattr(base, item.name)
                for item in fields(InteractiveDriveUIState)
            },
            cameras=camera_views_for_names(self._config.app.camera_names),
            auxiliary_views=self.auxiliary_views,
            focus_channel=-1 if selected is None else selected,
        )
