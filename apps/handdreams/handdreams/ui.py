# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small API-v2-native HandDreams status overlay."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache

import numpy as np
import torch
from flashdreams_controller_input.events import (
    ControllerConnectionStatus,
    ControllerFrameEvent,
    ControllerStatusEvent,
)
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from flashdreams.api_v2.client_window import ClientWindowLink
from flashdreams.api_v2.loop import IUILoop, invoke_async
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    TouchUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .alignment import CALIBRATION_TRIGGER_THRESHOLD, AlignmentClaw, AlignmentPreview
from .controller_preview import (
    ControllerPreview,
    ControllerPreviewCalibration,
    trigger_preview,
)
from .scenes import ScenePreview
from .tutorials import TutorialVideo


class HandDreamsPhase(str, Enum):
    SELECTING = "selecting"
    LOADING = "loading"
    WAITING = "waiting"
    CALIBRATING = "calibrating"
    ALIGNING = "aligning"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class HandDreamsUIStatus:
    """Immutable controller and scene snapshot sent to the UI thread."""

    phase: HandDreamsPhase = HandDreamsPhase.WAITING
    left_tracked: bool = False
    right_tracked: bool = False
    left_open: bool = False
    right_open: bool = False
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_battery: float | None = None
    right_battery: float | None = None
    current_scene: str = ""
    scene_catalog: tuple[str, ...] = ()
    alignment: AlignmentPreview | None = None
    """Model-owned frozen setup geometry; never a generated action."""
    calibration_progress: float = 0.0
    """Model-confirmed fraction of the continuous trigger hold."""
    is_emulated: bool = False
    """Model-confirmed virtual-controller setup policy, never inferred from a route."""
    replay: bool = False
    """Recorded playback without controller setup or tracking guidance."""


@dataclass(slots=True)
class HandDreamsUIState:
    """State owned exclusively by :class:`HandDreamsUILoop`."""

    status: HandDreamsUIStatus = HandDreamsUIStatus()
    scene_previews: tuple[ScenePreview, ...] = ()
    stage_width: int = 640
    stage_height: int = 360
    show_postprocess_toggle: bool = False
    postprocess_enabled: bool = False
    open_claw_threshold: float = 0.1
    pointer: tuple[float, float] | None = None
    highlighted_scene: str = ""
    """Keyboard selection awaiting confirmation, independent of the active scene."""
    scene_strip_cache_key: tuple[object, ...] | None = None
    scene_strip_cache: Tensor | None = None
    controller_frame: ControllerFrameEvent | None = None
    """Latest UI input, independent of model generation and calibration updates."""
    connection_status: ControllerConnectionStatus = ControllerConnectionStatus()
    """Source setup guidance retained across scene changes and model resets."""
    connection_links: tuple[ClientWindowLink, ...] = ()
    """Last published click regions; only changes need a window request."""
    controller_preview: ControllerPreview | None = None
    """UI-owned projector using the model's last accepted calibration."""
    tutorial: TutorialVideo = field(default_factory=TutorialVideo)
    """UI-owned bounded decoder for the two guided-setup clips."""
    live_started_at: float | None = None
    """Monotonic time of the first presented live frame; ``None`` before playback."""

    @property
    def selected_scene(self) -> str:
        """Return the highlighted scene, falling back to the active scene."""
        if self.highlighted_scene in self.status.scene_catalog:
            return self.highlighted_scene
        return self.status.current_scene

    def update_status(
        self,
        *,
        phase: HandDreamsPhase,
        current_scene: str,
        scene_catalog: tuple[str, ...],
        alignment: AlignmentPreview | None = None,
        calibration_progress: float = 0.0,
        controller_calibration: ControllerPreviewCalibration | None = None,
        is_emulated: bool = False,
    ) -> None:
        """Update model-owned state without overwriting newer controller input."""
        if self.status.phase is HandDreamsPhase.LOADING:
            return
        if controller_calibration is None:
            self.controller_preview = None
        elif (
            self.controller_preview is None
            or self.controller_preview.calibration is not controller_calibration
        ):
            self.controller_preview = ControllerPreview(controller_calibration)
        self.status = replace(
            self.status,
            phase=phase,
            current_scene=current_scene,
            scene_catalog=scene_catalog,
            alignment=alignment,
            calibration_progress=calibration_progress,
            is_emulated=is_emulated,
        )

    def observe_events(self, events: UserInputEvents) -> None:
        """Reflect source setup, controller tracking, triggers, and battery."""
        if self.status.replay:
            return
        status = self.status
        for event in events.get_events():
            if isinstance(event, ControllerStatusEvent):
                self.connection_status = event.status
                continue
            if not isinstance(event, ControllerFrameEvent):
                continue
            self.controller_frame = event
            if event.action == "disconnected":
                status = replace(
                    status,
                    left_tracked=False,
                    right_tracked=False,
                    left_open=False,
                    right_open=False,
                    left_trigger=0.0,
                    right_trigger=0.0,
                    left_battery=None,
                    right_battery=None,
                    alignment=None,
                    calibration_progress=0.0,
                    is_emulated=False,
                )
                continue
            left, right = event.left, event.right
            assert left is not None and right is not None
            status = replace(
                status,
                left_tracked=left.tracked,
                right_tracked=right.tracked,
                left_open=left.tracked and left.trigger <= self.open_claw_threshold,
                right_open=right.tracked and right.trigger <= self.open_claw_threshold,
                left_trigger=left.trigger,
                right_trigger=right.trigger,
                left_battery=left.battery,
                right_battery=right.battery,
            )
        self.status = status

    def reset_interaction(self) -> None:
        """Clear transient UI input and presentation state."""
        self.pointer = None
        self.highlighted_scene = ""
        self.controller_frame = None
        self.controller_preview = None
        self.live_started_at = None
        self.tutorial.close()


_BAR = (18, 21, 27)
_PANEL = (27, 31, 39)
_WELL = (42, 47, 58)
_TEXT = (236, 239, 244)
_MUTED = (165, 171, 184)
_ACCENT = (45, 154, 230)
_LOW_BATTERY = 0.25
_SCENE_PAGE_SIZE = 5
_CLAW_EDGES = (
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (4, 6),
    (3, 5),
    (5, 7),
    (2, 8),
    (3, 9),
    (8, 9),
)


def scene_strip_height(width: int) -> int:
    """Scale the V1-style selector with the presentation width."""
    return max(96, round(int(width) * 27 / 160))


def status_message(status: HandDreamsUIStatus) -> str:
    """Describe the next operator action without implying video is generating."""
    if status.phase is HandDreamsPhase.SELECTING:
        return "Choose a scene"
    if status.phase is HandDreamsPhase.LOADING:
        return "Loading scene..." if status.current_scene else "Returning to scenes..."
    if status.replay:
        return "Replay"
    if not (status.left_tracked and status.right_tracked):
        if status.phase is HandDreamsPhase.LIVE:
            return "Tracking lost"
        if status.left_tracked:
            return "Waiting for right controller"
        if status.right_tracked:
            return "Waiting for left controller"
        return "Waiting for controllers"
    if status.phase is HandDreamsPhase.LIVE:
        if any(
            battery is not None and battery <= _LOW_BATTERY
            for battery in (status.left_battery, status.right_battery)
        ):
            return "Live - low controller battery"
        return "Live"
    if status.phase is HandDreamsPhase.CALIBRATING:
        if status.is_emulated:
            return (
                "Release both triggers to start"
                if status.calibration_progress >= 1.0
                else "Hold either trigger for 1 second"
            )
        return "Close both claws for 1 second"
    if status.phase is HandDreamsPhase.WAITING:
        return "Preparing scene"
    if not (status.left_open and status.right_open):
        return "Open both claws"
    if status.phase is HandDreamsPhase.ALIGNING:
        if status.alignment is not None and status.alignment.aligned:
            return "Hold still"
        return "Match claw outlines"
    return "Starting demo"


@lru_cache(maxsize=64)
def _status_text(message: str, width: int, height: int) -> Tensor:
    image = Image.new("RGB", (width, height), _BAR)
    ImageDraw.Draw(image).text((6, 3), message, fill=(230, 234, 240))
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def draw_status_overlay(
    frame: Tensor,
    status: HandDreamsUIStatus,
    *,
    waiting_for_video: bool = False,
    controller_frame: ControllerFrameEvent | None = None,
    tutorial_frame: Image.Image | None = None,
    connection_status: ControllerConnectionStatus = ControllerConnectionStatus(),
) -> Tensor:
    """Draw guided setup or live status without changing the model frame."""
    if (
        frame.ndim != 3
        or int(frame.shape[0]) not in (1, 3, 4)
        or min(frame.shape[-2:]) < 1
    ):
        raise ValueError(
            f"HandDreams UI frame must be [C,H,W], got {tuple(frame.shape)}"
        )
    output = frame.clone()
    height, width = int(output.shape[-2]), int(output.shape[-1])
    if (
        status.phase
        in (
            HandDreamsPhase.SELECTING,
            HandDreamsPhase.LOADING,
            HandDreamsPhase.WAITING,
        )
        or waiting_for_video
    ):
        _draw_waiting_screen(
            output,
            status,
            waiting_for_video=waiting_for_video,
            connection_status=connection_status,
        )
        return output
    if status.phase in (HandDreamsPhase.CALIBRATING, HandDreamsPhase.ALIGNING):
        if width >= 320 and height >= 200:
            _draw_setup_screen(output, status, controller_frame, tutorial_frame)
            return output
    message = status_message(status)
    if message in ("Live", "Replay"):
        return output
    bar_h = min(18, height)
    text = _status_text(message, width, bar_h).to(
        device=output.device, dtype=output.dtype
    )
    if output.dtype.is_floating_point:
        text = text / 127.5 - 1.0
    if output.shape[0] == 1:
        output[0, :bar_h] = text.float().mean(dim=0).to(output.dtype)
    else:
        output[:3, :bar_h] = text
    return output


def _draw_waiting_screen(
    output: Tensor,
    status: HandDreamsUIStatus,
    *,
    waiting_for_video: bool,
    connection_status: ControllerConnectionStatus,
) -> None:
    """Show the next setup action on a neutral stage."""
    height, width = map(int, output.shape[-2:])
    panel = Image.new("RGB", (width, height), (14, 16, 20))
    draw = ImageDraw.Draw(panel)
    if status.phase is HandDreamsPhase.SELECTING:
        title = "Choose a scene"
        detail = "Click a scene, or use arrow keys then Enter."
    elif status.phase is HandDreamsPhase.LOADING:
        title = status_message(status)
        detail = (
            "Preparing the selected scenario."
            if status.current_scene
            else "Finishing the current scene."
        )
    elif waiting_for_video and status.replay:
        title = "Starting replay..."
        detail = "The first generated frame may take a moment."
    elif waiting_for_video:
        tracked = status.left_tracked and status.right_tracked
        title = "Starting demo..." if tracked else "Tracking lost"
        detail = (
            "The first generated frame may take a moment."
            if tracked
            else "Restore controller tracking to continue."
        )
    else:
        title, detail = "Preparing scene...", status_message(status)
    show_connections = (
        not status.replay
        and not (status.left_tracked and status.right_tracked)
        and connection_status != ControllerConnectionStatus()
    )
    draw.text(
        (width * 0.5, height * (0.14 if show_connections else 0.43)),
        title,
        font=_font(max(12, width // 32), bold=True),
        fill=_TEXT,
        anchor="mm",
    )
    if detail:
        draw.text(
            (width * 0.5, height * (0.23 if show_connections else 0.55)),
            detail,
            font=_font(max(8, width // 64)),
            fill=_MUTED,
            anchor="mm",
        )
    if status.phase is HandDreamsPhase.SELECTING and not status.replay:
        for index, side in enumerate(("left", "right")):
            tracked = getattr(status, f"{side}_tracked")
            battery = getattr(status, f"{side}_battery")
            x, y = (
                width * (0.25 + 0.34 * index),
                height * (0.32 if show_connections else 0.72),
            )
            radius = max(2, width // 200)
            indicator_x = x - radius * 3
            draw.ellipse(
                (
                    indicator_x - radius,
                    y - radius,
                    indicator_x + radius,
                    y + radius,
                ),
                fill=(118, 208, 110) if tracked else _MUTED,
            )
            draw.text(
                (x, y),
                f"{side.title()}: {'Tracked' if tracked else 'Not tracked'}",
                font=_font(max(9, width // 53)),
                fill=_TEXT if tracked else _MUTED,
                anchor="lm",
            )
            if battery is not None:
                draw.text(
                    (x, height * (0.39 if show_connections else 0.79)),
                    f"Battery: {battery:.0%}",
                    font=_font(max(8, width // 64)),
                    fill=(235, 182, 65) if battery <= _LOW_BATTERY else _MUTED,
                    anchor="lm",
                )
    if show_connections:
        _draw_connection_details(panel, connection_status)
    _paste_panel(output, panel)


def _draw_connection_details(
    panel: Image.Image, status: ControllerConnectionStatus
) -> None:
    """Show headset status and one labeled Isaac Teleop connection row."""
    width, height = panel.size
    draw = ImageDraw.Draw(panel)
    font = _font(max(8, min(18, width // 45, height // 20)))
    draw.text(
        (width / 2, height * 0.57),
        _ellipsize(
            f"Headset: {status.headset or 'Waiting...'}",
            font,
            max(1, int(width * 0.88)),
        ),
        font=font,
        fill=_MUTED,
        anchor="mm",
    )
    links = _connection_links(status, width, height)
    if not links:
        draw.text(
            (width / 2, height * 0.70),
            _ellipsize(
                f"Isaac Teleop Web Client: {status.web_client or 'Waiting...'}",
                font,
                max(1, int(width * 0.88)),
            ),
            font=font,
            fill=_MUTED,
            anchor="mm",
        )
        return
    (link,) = links
    left, top, right, bottom = link.bounds
    position = (left * width + 4, (top + bottom) * height / 2)
    draw.text(
        position, "Isaac Teleop Web Client: ", font=font, fill=_MUTED, anchor="rm"
    )
    draw.text(position, "Open Link", font=font, fill=_ACCENT, anchor="lm")
    draw.line(
        (left * width + 4, bottom * height - 4, right * width - 4, bottom * height - 4),
        fill=_ACCENT,
    )


def _connection_links(
    status: ControllerConnectionStatus, width: int, height: int
) -> tuple[ClientWindowLink, ...]:
    """Match the click region to just the Open Link text on the neutral stage."""
    if not status.web_client_url or width < 160 or height < 160:
        return ()
    size = max(8, min(18, width // 45, height // 20))
    font = _font(size)
    prefix_width = font.getlength("Isaac Teleop Web Client: ")
    link_width = font.getlength("Open Link")
    left = (width - prefix_width - link_width) / 2 + prefix_width
    half_height = (size + 12) / (2 * height)
    return (
        ClientWindowLink(
            label="Open Isaac Teleop Web Client",
            url=status.web_client_url,
            bounds=(
                (left - 4) / width,
                0.70 - half_height,
                (left + link_width + 4) / width,
                0.70 + half_height,
            ),
        ),
    )


def _draw_setup_screen(
    output: Tensor,
    status: HandDreamsUIStatus,
    controller_frame: ControllerFrameEvent | None,
    tutorial_frame: Image.Image | None,
) -> None:
    height, width = map(int, output.shape[-2:])
    panel = Image.new("RGB", (width, height), (14, 16, 20))
    draw = ImageDraw.Draw(panel)
    calibrating = status.phase is HandDreamsPhase.CALIBRATING
    ready = status.left_tracked and status.right_tracked
    green, red = (118, 208, 110), (245, 65, 65)
    title = "Try the robot grippers" if calibrating else "Match the outlines"
    draw.text(
        (width * 0.5, height * 0.07),
        title,
        font=_font(max(16, width // 44), bold=True),
        fill=_TEXT,
        anchor="mm",
    )
    draw.text(
        (width * 0.965, height * 0.025),
        "XR ready" if ready else status_message(status),
        font=_font(max(8, width // 100), bold=True),
        fill=green if ready else _MUTED,
        anchor="ra",
    )
    bounds = (width * 0.08, height * 0.23, width * 0.86, height * 0.68)
    geometry = Image.new("RGBA", panel.size)
    claw_draw = ImageDraw.Draw(geometry)
    preview = status.alignment
    midpoint = (
        (np.asarray(preview.left_target.position) + preview.right_target.position) / 2
        if preview is not None
        else np.zeros(3)
    )
    # The trigger tutorial follows the hands' heading; alignment stays calibrated.
    tutorial_claws = trigger_preview(controller_frame) if calibrating else (None, None)
    view = np.array([[1, 0, 0], [0, 0.6, -0.8], [0, 0.8, 0.6]])
    for index, (side, normal) in enumerate(
        (("left", (40, 190, 235)), ("right", (245, 120, 70)))
    ):
        tracked = getattr(status, f"{side}_tracked")
        trigger = (
            min(1.0, max(0.0, getattr(status, f"{side}_trigger"))) if tracked else 0.0
        )
        hand = getattr(preview, side) if preview else None
        color = normal if tracked else _MUTED
        label = f"{side.title()}: not tracked"
        if calibrating:
            hand = tutorial_claws[index]
            position = (
                view @ hand.position
                if hand is not None
                else np.array([-0.18 if index == 0 else 0.18, 0, 0])
            )
            rotation = view @ (
                np.asarray(hand.rotation).reshape(3, 3)
                if hand is not None
                else np.diag([-1, -1, 1])
            )
            hand = AlignmentClaw(tuple(position), tuple(rotation.flat))
            _draw_setup_claw(
                claw_draw, hand, np.zeros(3), bounds, trigger, color, width
            )
        else:
            if preview:
                _draw_setup_claw(
                    claw_draw,
                    getattr(preview, f"{side}_target"),
                    midpoint,
                    bounds,
                    trigger,
                    (142, 148, 158),
                    width,
                    dashed=True,
                )
            if tracked and hand is not None:
                opened = getattr(status, f"{side}_open")
                color = red if not opened else green if hand.aligned else normal
                label = f"{side.title()}: " + (
                    "open claw"
                    if not opened
                    else "aligned"
                    if hand.aligned
                    else "move into guide"
                )
                _draw_setup_claw(
                    claw_draw, hand, midpoint, bounds, trigger, color, width
                )
            draw.text(
                (width * (0.36 + 0.25 * index), height * 0.77),
                label,
                font=_font(max(9, width // 90), bold=True),
                fill=color,
                anchor="mm",
            )
        if calibrating:
            left = int(width * (0.27 + 0.27 * index))
            top, card_width, card_height = (
                int(height * 0.72),
                int(width * 0.20),
                int(height * 0.09),
            )
            color = (
                green
                if tracked and trigger >= CALIBRATION_TRIGGER_THRESHOLD
                else _MUTED
            )
            draw.rounded_rectangle(
                (left, top, left + card_width, top + card_height),
                radius=6,
                fill=_PANEL,
                outline=color,
                width=max(1, width // 640),
            )
            label = (
                f"{side.title()} trigger" if tracked else f"{side.title()}: not tracked"
            )
            draw.text(
                (left + 8, top + 5),
                label,
                font=_font(max(9, width // 90), bold=True),
                fill=color,
            )
            bar = (
                left + 8,
                top + card_height - 10,
                left + card_width - 8,
                top + card_height - 5,
            )
            draw.rectangle(bar, fill=_WELL)
            if trigger > 0:
                draw.rectangle(
                    (bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * trigger, bar[3]),
                    fill=color,
                )

    left, top, right, bottom = (round(value) for value in bounds)
    clip_bounds = (left, top, right, bottom)
    geometry = geometry.crop(clip_bounds)
    panel.paste(geometry, clip_bounds[:2], geometry)

    if calibrating:
        footer = (
            status_message(status)
            if status.is_emulated
            else "Hold down both triggers for 1 second"
        )
        held_trigger = (max if status.is_emulated else min)(
            status.left_trigger, status.right_trigger
        )
        progress = (
            status.calibration_progress
            if ready
            and (
                held_trigger >= CALIBRATION_TRIGGER_THRESHOLD
                or status.is_emulated
                and status.calibration_progress >= 1.0
            )
            else 0.0
        )
    else:
        footer = (
            "Hold still"
            if ready
            and status.left_open
            and status.right_open
            and preview is not None
            and preview.aligned
            else "Keep both claws open"
        )
        progress = (
            preview.progress
            if ready and preview is not None and status.left_open and status.right_open
            else 0.0
        )
    progress = min(1.0, max(0.0, progress))
    draw.text(
        (width * 0.5, height * 0.93),
        footer,
        font=_font(max(11, width // 58), bold=True),
        fill=_TEXT,
        anchor="mm",
    )
    bar = (width * 0.30, height * 0.88, width * 0.70, height * 0.89)
    draw.rectangle(bar, fill=_WELL)
    if progress > 0:
        draw.rectangle(
            (bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * progress, bar[3]),
            fill=green if calibrating else (235, 182, 65),
        )
    if tutorial_frame is not None:
        left = width - tutorial_frame.width - int(width * 0.025)
        top = int(height * 0.22)
        mask = Image.new("L", tutorial_frame.size)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, tutorial_frame.width - 1, tutorial_frame.height - 1),
            radius=8,
            fill=255,
        )
        panel.paste(tutorial_frame, (left, top), mask)
        draw.rounded_rectangle(
            (left, top, left + tutorial_frame.width, top + tutorial_frame.height),
            radius=8,
            outline=_ACCENT,
            width=max(1, width // 640),
        )
    _paste_panel(output, panel)


def _claw_points(trigger: float) -> np.ndarray:
    """Return shared setup and HUD claw joints in meters."""
    trigger = min(1.0, max(0.0, trigger))
    half = 0.055 * (1 - trigger) + 0.018 * trigger
    return np.array(
        [
            (0, 0, -0.07),
            (0, 0, 0),
            (-0.055, 0, 0.035),
            (0.055, 0, 0.035),
            (-0.065, 0, 0.12),
            (0.065, 0, 0.12),
            (-half, 0, 0.21),
            (half, 0, 0.21),
            (-0.04, 0.035, 0.04),
            (0.04, 0.035, 0.04),
        ]
    )


def _draw_setup_claw(
    draw: ImageDraw.ImageDraw,
    claw: AlignmentClaw,
    midpoint: np.ndarray,
    bounds: tuple[float, float, float, float],
    trigger: float,
    color: tuple[int, int, int],
    canvas_width: int,
    *,
    dashed: bool = False,
) -> None:
    points = (
        _claw_points(trigger) @ np.asarray(claw.rotation).reshape(3, 3).T
        + claw.position
        - midpoint
    )
    left, top, right, bottom = bounds
    focal = min(right - left, bottom - top) * 0.82
    projected = points[:, :2] * focal / np.maximum(points[:, 2:] + 0.75, 0.12)
    projected += ((left + right) / 2, top + (bottom - top) * 0.48)
    stroke = max(1, canvas_width // 640)
    for first, second in _CLAW_EDGES:
        start, end = projected[first], projected[second]
        if dashed:
            segments = max(
                1, int(np.linalg.norm(end - start) / max(3, canvas_width // 160))
            )
            for index in range(0, segments, 2):
                draw.line(
                    (
                        *(start + (end - start) * index / segments),
                        *(start + (end - start) * min(index + 1, segments) / segments),
                    ),
                    fill=color,
                    width=stroke,
                )
        else:
            draw.line((*start, *end), fill=color, width=stroke)
    if not dashed:
        radius = max(2, canvas_width // 400)
        for x, y in projected:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _paste_panel(output: Tensor, panel: Image.Image) -> None:
    width, height = panel.size
    pixels = (
        torch.from_numpy(np.array(panel, copy=True))
        .permute(2, 0, 1)
        .to(device=output.device, dtype=output.dtype)
    )
    if output.dtype.is_floating_point:
        pixels = pixels / 127.5 - 1.0
    left = (int(output.shape[-1]) - width) // 2
    if output.shape[0] == 1:
        output[0, -height:, left : left + width] = (
            pixels.float().mean(0).to(output.dtype)
        )
    else:
        output[:3, -height:, left : left + width] = pixels


def _draw_claw(
    draw: ImageDraw.ImageDraw,
    claw: AlignmentClaw,
    midpoint: np.ndarray,
    center: np.ndarray,
    scale: float,
    vertical_axis: int,
    color: tuple[int, int, int],
    width: int,
    *,
    trigger: float = 0.0,
    joints: bool = False,
) -> None:
    points = (
        _claw_points(trigger) @ np.asarray(claw.rotation).reshape(3, 3).T
        + np.asarray(claw.position)
        - midpoint
    )
    projected = center + points[:, [0, vertical_axis]] * scale
    for first, second in _CLAW_EDGES:
        draw.line((*projected[first], *projected[second]), fill=color, width=width)
    if joints:
        for x, y in projected:
            draw.ellipse((x - width, y - width, x + width, y + width), fill=color)


def _draw_controller_hud(output: Tensor, state: HandDreamsUIState) -> None:
    event = state.controller_frame
    preview = state.controller_preview
    if (
        event is None
        or preview is None
        or state.status.phase
        in (
            HandDreamsPhase.SELECTING,
            HandDreamsPhase.LOADING,
            HandDreamsPhase.CALIBRATING,
            HandDreamsPhase.ALIGNING,
        )
    ):
        return
    frame_height, frame_width = map(int, output.shape[-2:])
    if frame_width < 320 or frame_height < 200:
        return
    width = min(240, max(135, frame_width * 3 // 16))
    margin = max(8, min(16, frame_width // 80))
    height = min(round(width * 0.6), frame_height - margin - 24)
    panel = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=9, fill=(*_BAR, 190))
    draw.text((9, 7), "XR CONTROL", font=_font(9, bold=True), fill=(*_MUTED, 255))
    geometry = Image.new("RGBA", panel.size)
    claw_draw = ImageDraw.Draw(geometry)

    # Calibrated camera axes are +Y down and +Z forward. Anchor the view to
    # the scene, so common hand motion and separation do not move the camera.
    view = np.array([[1, 0, 0], [0, -0.8, -0.6], [0, 0.6, -0.8]])
    hands = (event.left, event.right)
    claws = preview.observe(event)
    positions = [view @ target.position for target in preview.calibration.targets]
    midpoint = np.mean(positions, axis=0)
    span = np.ptp(positions, axis=0)
    # Leave room for arm movement while keeping the calibrated view fixed.
    scale = 0.7 * min(
        (width - 24) / max(0.8, span[0] + 0.3),
        (height - 42) / max(0.35, span[2] + 0.3),
    )
    center = np.array([width / 2, height / 2 + 9])
    for hand, claw, color in zip(
        hands, claws, ((65, 185, 235), (240, 150, 60)), strict=True
    ):
        if claw is None:
            continue
        assert hand is not None and claw is not None
        rotation = view @ np.asarray(claw.rotation).reshape(3, 3)
        claw = AlignmentClaw(tuple(view @ claw.position), tuple(rotation.reshape(-1)))
        _draw_claw(
            claw_draw,
            claw,
            midpoint,
            center,
            scale,
            2,
            color,
            1,
            trigger=hand.trigger,
            joints=True,
        )

    bounds = (9, 24, width - 9, height - 9)
    panel.alpha_composite(geometry.crop(bounds), bounds[:2])
    pixels = torch.from_numpy(np.array(panel, copy=True)).permute(2, 0, 1)
    pixels = pixels.to(device=output.device, dtype=torch.float32)
    alpha = pixels[3:] / 255.0
    colors = pixels[:3]
    if output.dtype.is_floating_point:
        colors = colors / 127.5 - 1.0
    if output.shape[0] == 1:
        colors = colors.mean(0, keepdim=True)
    left, top = frame_width - width - margin, frame_height - height - margin
    target = output[:3, top : top + height, left : left + width]
    target.copy_((target * (1 - alpha) + colors * alpha).to(output.dtype))


def _format_stopwatch(elapsed_s: float) -> str:
    seconds = max(0, int(elapsed_s))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _draw_stopwatch(output: Tensor, elapsed_s: float) -> None:
    """Blend an unlabeled elapsed-time panel into the video's bottom-left corner."""
    frame_height, frame_width = map(int, output.shape[-2:])
    if frame_width < 320 or frame_height < 200:
        return
    text = _format_stopwatch(elapsed_s)
    font = _font(max(11, min(18, frame_width // 35)), bold=True)
    width = max(
        min(99, max(75, frame_width * 3 // 16)), round(font.getlength(text)) + 18
    )
    height = 40
    margin = max(8, min(16, frame_width // 80))
    panel = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=9, fill=(*_BAR, 156))
    draw.text((width / 2, height / 2), text, font=font, fill=(*_TEXT, 255), anchor="mm")
    pixels = torch.from_numpy(np.array(panel, copy=True)).permute(2, 0, 1)
    pixels = pixels.to(device=output.device, dtype=torch.float32)
    alpha = pixels[3:] / 255.0
    colors = pixels[:3]
    if output.dtype.is_floating_point:
        colors = colors / 127.5 - 1.0
    if output.shape[0] == 1:
        colors = colors.mean(0, keepdim=True)
    top = frame_height - height - margin
    target = output[:3, top : top + height, margin : margin + width]
    target.copy_((target * (1 - alpha) + colors * alpha).to(output.dtype))


@lru_cache(maxsize=12)
def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for directory in (
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
    ):
        for name in names:
            try:
                return ImageFont.truetype(directory + name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _postprocess_button_rect(width: int, height: int) -> tuple[int, int, int, int]:
    button_width = max(168, min(260, width // 4))
    button_height = max(32, min(48, height // 8))
    margin = max(8, width // 80)
    top = max(24, height // 18)
    return width - margin - button_width, top, width - margin, top + button_height


def _postprocess_at(
    state: HandDreamsUIState,
    x: float,
    y: float,
    *,
    canvas_width: int,
    canvas_height: int,
) -> bool:
    if not state.show_postprocess_toggle or state.status.phase in (
        HandDreamsPhase.CALIBRATING,
        HandDreamsPhase.ALIGNING,
    ):
        return False
    left, top, right, bottom = _postprocess_button_rect(
        state.stage_width, state.stage_height
    )
    px, py = float(x) * canvas_width, float(y) * canvas_height
    return left <= px <= right and top <= py <= bottom


@lru_cache(maxsize=8)
def _postprocess_button_image(enabled: bool, width: int, height: int) -> Tensor:
    image = Image.new("RGB", (width, height), (24, 29, 36))
    draw = ImageDraw.Draw(image)
    color = (61, 190, 105) if enabled else (150, 155, 165)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=max(5, height // 5),
        outline=color,
        width=max(2, height // 16),
    )
    draw.text(
        (width // 2, height // 2),
        f"POSTPROCESS {'ON' if enabled else 'OFF'}",
        font=_font(max(10, height // 3), bold=True),
        fill=color,
        anchor="mm",
    )
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def _draw_postprocess_toggle(output: Tensor, state: HandDreamsUIState) -> None:
    if not state.show_postprocess_toggle or state.status.phase in (
        HandDreamsPhase.CALIBRATING,
        HandDreamsPhase.ALIGNING,
    ):
        return
    left, top, right, bottom = _postprocess_button_rect(
        int(output.shape[-1]), int(output.shape[-2])
    )
    pixels = _postprocess_button_image(
        state.postprocess_enabled, right - left, bottom - top
    ).to(device=output.device, dtype=output.dtype)
    if output.dtype.is_floating_point:
        pixels = pixels / 127.5 - 1.0
    if output.shape[0] == 1:
        output[0, top:bottom, left:right] = pixels.float().mean(0).to(output.dtype)
    else:
        output[:3, top:bottom, left:right] = pixels


def _ellipsize(
    text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int
) -> str:
    if font.getlength(text) <= width:
        return text
    for end in range(len(text) - 1, 0, -1):
        candidate = text[:end] + "..."
        if font.getlength(candidate) <= width:
            return candidate
    return "..."


def _scene_card_layout(
    previews: tuple[ScenePreview, ...],
    current_scene: str,
    width: int,
    height: int,
) -> tuple[tuple[ScenePreview, tuple[int, int, int, int]], ...]:
    if not previews:
        return ()
    selected = next(
        (
            index
            for index, preview in enumerate(previews)
            if preview.name == current_scene
        ),
        0,
    )
    page = selected // _SCENE_PAGE_SIZE
    visible = previews[page * _SCENE_PAGE_SIZE : (page + 1) * _SCENE_PAGE_SIZE]
    padding = max(8, width // 80)
    gap = max(6, width // 128)
    top = max(28, round(height * 0.16))
    bottom = height - max(8, height // 24)
    card_width = max(
        1, (width - 2 * padding - gap * (len(visible) - 1)) // len(visible)
    )
    return tuple(
        (
            preview,
            (
                padding + index * (card_width + gap),
                top,
                padding + index * (card_width + gap) + card_width,
                bottom,
            ),
        )
        for index, preview in enumerate(visible)
    )


def _scene_at(
    state: HandDreamsUIState,
    x: float,
    y: float,
    *,
    canvas_width: int,
    canvas_height: int,
) -> str | None:
    strip_height = canvas_height - state.stage_height
    local_x = float(x) * canvas_width
    local_y = float(y) * canvas_height - state.stage_height
    if strip_height <= 0 or local_y < 0:
        return None
    for preview, (left, top, right, bottom) in _scene_card_layout(
        state.scene_previews,
        state.selected_scene,
        canvas_width,
        strip_height,
    ):
        if left <= local_x <= right and top <= local_y <= bottom:
            return preview.name
    return None


def _render_scene_strip(
    state: HandDreamsUIState,
    *,
    width: int,
    height: int,
    hovered_scene: str | None,
) -> Tensor:
    image = Image.new("RGB", (width, height), _BAR)
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, width, 0), fill=_ACCENT, width=max(1, width // 640))
    header_font = _font(max(10, width // 106), bold=True)
    draw.text(
        (max(8, width // 160), max(6, height // 24)),
        "SCENES",
        font=header_font,
        fill=_TEXT,
    )
    pages = max(
        1, (len(state.scene_previews) + _SCENE_PAGE_SIZE - 1) // _SCENE_PAGE_SIZE
    )
    if pages > 1:
        selected = next(
            (
                index
                for index, preview in enumerate(state.scene_previews)
                if preview.name == state.selected_scene
            ),
            0,
        )
        draw.text(
            (width // 2, max(6, height // 24)),
            f"{selected // _SCENE_PAGE_SIZE + 1} / {pages}",
            font=_font(max(9, width // 116), bold=True),
            fill=_MUTED,
            anchor="ma",
        )

    title_font = _font(max(10, width // 106), bold=True)
    title_height = max(22, round(height * 0.16))
    for preview, rect in _scene_card_layout(
        state.scene_previews, state.selected_scene, width, height
    ):
        left, top, right, bottom = rect
        selected = preview.name == state.selected_scene
        hovered = preview.name == hovered_scene
        draw.rounded_rectangle(
            rect,
            radius=max(5, width // 128),
            fill=_PANEL,
            outline=_ACCENT if selected or hovered else _WELL,
            width=3 if selected or hovered else 2,
        )
        image_rect = (left + 5, top + 5, right - 5, bottom - title_height)
        draw.rounded_rectangle(image_rect, radius=4, fill=_WELL)
        if preview.thumbnail is not None:
            thumbnail = Image.fromarray(preview.thumbnail)
            thumbnail.thumbnail(
                (
                    max(1, image_rect[2] - image_rect[0]),
                    max(1, image_rect[3] - image_rect[1]),
                ),
                Image.Resampling.LANCZOS,
            )
            image.paste(
                thumbnail,
                (
                    (image_rect[0] + image_rect[2] - thumbnail.width) // 2,
                    (image_rect[1] + image_rect[3] - thumbnail.height) // 2,
                ),
            )
        else:
            draw.text(
                (
                    (image_rect[0] + image_rect[2]) // 2,
                    (image_rect[1] + image_rect[3]) // 2,
                ),
                "Preview unavailable",
                font=_font(max(8, width // 142)),
                fill=_MUTED,
                anchor="mm",
            )
        draw.text(
            (left + 7, bottom - title_height + 5),
            _ellipsize(preview.title, title_font, max(1, right - left - 14)),
            font=title_font,
            fill=_TEXT,
        )
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def _compose_scene_selector(
    frame: Tensor,
    state: HandDreamsUIState,
    *,
    canvas_width: int,
    canvas_height: int,
) -> Tensor:
    if int(frame.shape[-1]) != canvas_width:
        raise ValueError(
            f"HandDreams stage width {frame.shape[-1]} does not match UI width {canvas_width}"
        )
    strip_height = canvas_height - int(frame.shape[-2])
    if strip_height <= 0 or not state.scene_previews:
        return frame
    hovered = (
        _scene_at(
            state,
            *state.pointer,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        if state.pointer is not None
        else None
    )
    key = (
        state.selected_scene,
        hovered,
        canvas_width,
        strip_height,
        frame.device,
        frame.dtype,
        int(frame.shape[0]),
    )
    if state.scene_strip_cache_key != key:
        strip = _render_scene_strip(
            state, width=canvas_width, height=strip_height, hovered_scene=hovered
        ).to(device=frame.device, dtype=frame.dtype)
        if frame.dtype.is_floating_point:
            strip = strip / 127.5 - 1.0
        if frame.shape[0] == 1:
            strip = strip.float().mean(0, keepdim=True).to(frame.dtype)
        elif frame.shape[0] == 4:
            alpha = torch.full_like(strip[:1], 255)
            if frame.dtype.is_floating_point:
                alpha.fill_(1)
            strip = torch.cat((strip, alpha))
        state.scene_strip_cache_key = key
        state.scene_strip_cache = strip
    assert state.scene_strip_cache is not None
    return torch.cat((frame, state.scene_strip_cache), dim=-2)


class HandDreamsUILoop(IUILoop[HandDreamsUIState]):
    """Present model video, setup guidance, and the V1-style scene selector."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        self.state.observe_events(events)
        requested_scene: str | None = None
        toggle_postprocess = False
        select_scene = False
        for event in events.get_events():
            if (
                isinstance(event, KeyboardUserInputEvent)
                and event.state is KeyboardInputState.PRESSED
            ):
                if event.key == "Escape":
                    select_scene = True
                elif event.key == "ArrowRight":
                    self._highlight_scene(1)
                elif event.key == "ArrowLeft":
                    self._highlight_scene(-1)
                elif event.key == "Enter":
                    requested_scene = self.state.selected_scene or next(
                        iter(self.state.status.scene_catalog), None
                    )
            elif isinstance(event, MouseUserInputEvent):
                self.state.pointer = (event.x, event.y)
                if event.action == "button" and event.button == 0 and event.pressed:
                    if self._postprocess_at(event.x, event.y):
                        toggle_postprocess = True
                    else:
                        requested_scene = self._scene_at(event.x, event.y)
            elif (
                isinstance(event, TouchUserInputEvent)
                and event.action == "start"
                and event.primary
            ):
                if self._postprocess_at(event.x, event.y):
                    toggle_postprocess = True
                else:
                    requested_scene = self._scene_at(event.x, event.y)
        if select_scene:
            self._request_scene_name(None)
        elif toggle_postprocess:
            self.set_postprocess_enabled(not self.state.postprocess_enabled)
        elif requested_scene is not None:
            self._request_scene_name(requested_scene)
        status = self.state.status
        frame = (
            self.presented_model_frame()
            if status.phase is HandDreamsPhase.LIVE
            else None
        )
        waiting_for_video = frame is None and status.phase is HandDreamsPhase.LIVE
        if frame is None:
            frame = torch.zeros(
                (3, self.state.stage_height, self.state.stage_width),
                dtype=torch.uint8,
            )
        tutorial_frame = None
        if status.phase in (HandDreamsPhase.CALIBRATING, HandDreamsPhase.ALIGNING):
            tutorial_frame = self.state.tutorial.frame(
                "press_triggers.mp4"
                if status.phase is HandDreamsPhase.CALIBRATING
                else "align_arms.mp4",
                max(1, int(frame.shape[-1] * 0.18)),
                max(1, int(frame.shape[-2] * 0.40)),
                time.monotonic(),
            )
        else:
            self.state.tutorial.close()
        output = draw_status_overlay(
            frame,
            status,
            waiting_for_video=waiting_for_video,
            controller_frame=self.state.controller_frame,
            tutorial_frame=tutorial_frame,
            connection_status=self.state.connection_status,
        )
        if status.phase is HandDreamsPhase.LIVE and not waiting_for_video:
            if not status.replay:
                _draw_controller_hud(output, self.state)
            now = time.monotonic()
            if self.state.live_started_at is None:
                self.state.live_started_at = now
            _draw_stopwatch(output, now - self.state.live_started_at)
        _draw_postprocess_toggle(output, self.state)
        output = _compose_scene_selector(
            output,
            self.state,
            canvas_width=self.session_desc.video_width,
            canvas_height=self.session_desc.video_height,
        )
        links = ()
        if (
            not status.replay
            and not (status.left_tracked and status.right_tracked)
            and (
                waiting_for_video
                or status.phase
                in (
                    HandDreamsPhase.SELECTING,
                    HandDreamsPhase.LOADING,
                    HandDreamsPhase.WAITING,
                )
            )
        ):
            stage_width, stage_height = int(frame.shape[-1]), int(frame.shape[-2])
            height_scale = stage_height / int(output.shape[-2])
            links = tuple(
                replace(
                    link,
                    bounds=(
                        link.bounds[0],
                        link.bounds[1] * height_scale,
                        link.bounds[2],
                        link.bounds[3] * height_scale,
                    ),
                )
                for link in _connection_links(
                    self.state.connection_status, stage_width, stage_height
                )
            )
        if links != self.state.connection_links:
            self.request_links(links)
            self.state.connection_links = links
        return [
            StepResult(
                step_index=step_index,
                output=output.unsqueeze(0),
                frame_count=1,
                output_layout=self.output_layout,
            )
        ]

    def is_finished(self) -> bool:
        return False

    def close(self) -> None:
        """Release the setup video decoder when the session ends."""
        self.state.tutorial.close()

    def reset(self) -> None:
        status = self.state.status
        self.state.status = HandDreamsUIStatus(
            phase=(
                HandDreamsPhase.SELECTING
                if status.phase is HandDreamsPhase.SELECTING
                else HandDreamsPhase.LIVE
                if status.replay
                else HandDreamsPhase.WAITING
            ),
            current_scene=status.current_scene,
            scene_catalog=status.scene_catalog,
            replay=status.replay,
        )
        self.state.reset_interaction()

    def _highlight_scene(self, direction: int) -> None:
        status = self.state.status
        catalog = status.scene_catalog
        if not catalog or status.phase is HandDreamsPhase.LOADING:
            return
        try:
            current_index = catalog.index(self.state.selected_scene)
        except ValueError:
            current_index = -1 if direction > 0 else 0
        self.state.highlighted_scene = catalog[
            (current_index + direction) % len(catalog)
        ]

    def _scene_at(self, x: float, y: float) -> str | None:
        return _scene_at(
            self.state,
            x,
            y,
            canvas_width=self.session_desc.video_width,
            canvas_height=self.session_desc.video_height,
        )

    def _postprocess_at(self, x: float, y: float) -> bool:
        return _postprocess_at(
            self.state,
            x,
            y,
            canvas_width=self.session_desc.video_width,
            canvas_height=self.session_desc.video_height,
        )

    def set_postprocess_enabled(self, enabled: bool) -> None:
        """Apply a button change without replacing the active session."""
        if not self.state.show_postprocess_toggle:
            return
        enabled = bool(enabled)
        if enabled == self.state.postprocess_enabled:
            return
        self.state.postprocess_enabled = enabled
        invoke_async(
            self._model_loop,
            lambda model_state, enabled=enabled: model_state.set_postprocess_enabled(
                enabled
            ),
        )

    def _request_scene_name(self, sample: str | None) -> None:
        """Request a scene, or return to the picker when ``sample`` is ``None``."""
        status = self.state.status
        catalog = status.scene_catalog
        if status.phase is HandDreamsPhase.LOADING:
            return
        metadata = dict(self.session_desc.metadata)
        if sample is None:
            if status.phase is HandDreamsPhase.SELECTING:
                return
            metadata["select_scene"] = True
        else:
            if sample not in catalog or (
                sample == status.current_scene
                and status.phase is not HandDreamsPhase.SELECTING
            ):
                return
            metadata.pop("select_scene", None)
            metadata["sample"] = sample
        metadata["scene_catalog"] = catalog
        self.state.status = replace(
            status,
            phase=HandDreamsPhase.LOADING,
            current_scene=sample or "",
            alignment=None,
            calibration_progress=0.0,
        )
        self.state.reset_interaction()
        self.request_new_session(replace(self.session_desc, metadata=metadata))


__all__ = [
    "HandDreamsPhase",
    "HandDreamsUILoop",
    "HandDreamsUIState",
    "HandDreamsUIStatus",
    "draw_status_overlay",
    "scene_strip_height",
]
