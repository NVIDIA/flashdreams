# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Steering-wheel and pedal overlays."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

from flashdreams.serving.presentation import (
    DisplayFrame,
    LRUCache,
    PointerEvent,
    Rect,
    measure_text,
    resolve_font,
)
from PIL import Image, ImageDraw

from interactive_drive_app.overlays.theme import (
    ACCENT_AMBER,
    NVIDIA_GREEN,
    TEXT_COLOR,
)

WHEEL_ROTATION_QUANTUM_DEG = 3
"""Rotation bucket size. Full lock is +/-450 degrees, so worst case is 300
cached images -- small at this radius, and it saves a ~2 ms rotate per tick."""

MAX_STEER_DEG = 450.0

_PEDAL_PRESSED_THRESHOLD = 0.05


class DriveState(Protocol):
    """The control values the chrome reads each tick."""

    steering: float
    throttle: float
    brake: float


class WheelWidget:
    """Steering wheel sprite, rotated to the current steering angle.

    Falls back to a drawn circle with an angle indicator when the optional
    control-asset pack is absent, so the chrome stays informative on a plain
    install.
    """

    def __init__(
        self,
        drive_state: Callable[[], DriveState],
        *,
        control_assets: Any | None = None,
        radius: int = 112,
    ) -> None:
        self._drive_state = drive_state
        self._control_assets = control_assets
        self._radius = radius
        self._font = resolve_font(22)
        self._base: Image.Image | None = None
        self._base_radius: int | None = None
        self._rotations: LRUCache = LRUCache(maxsize=480)

    def measure(self, panel_width: int) -> int:
        del panel_width
        return self._radius * 2 + 40

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        rect: Rect,
    ) -> None:
        del frame
        state = self._drive_state()
        centre = ((rect[0] + rect[2]) // 2, rect[1] + self._radius)

        base = self._wheel_base()
        if base is None:
            self._draw_fallback(draw, centre, state.steering)
        else:
            bucket = (
                round(state.steering * MAX_STEER_DEG / WHEEL_ROTATION_QUANTUM_DEG)
                * WHEEL_ROTATION_QUANTUM_DEG
            )
            rotated = self._rotations.get_or_compute(
                bucket,
                lambda b=bucket, image=base: image.rotate(
                    b, resample=Image.Resampling.BILINEAR
                ),
            )
            canvas.alpha_composite(
                rotated,
                (centre[0] - rotated.width // 2, centre[1] - rotated.height // 2),
            )

        angle_text = f"{int(state.steering * MAX_STEER_DEG):+}\u00b0"
        bbox = measure_text(self._font, angle_text)
        draw.text(
            (
                centre[0] - (bbox[2] - bbox[0]) // 2 - bbox[0],
                centre[1] + self._radius + 16 - bbox[1],
            ),
            angle_text,
            fill=ACCENT_AMBER,
            font=self._font,
        )

    def prepare(self, frame: DisplayFrame) -> None:
        del frame

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size

    def on_pointer(self, event: PointerEvent) -> bool:
        del event
        return False

    def close(self) -> None:
        return

    def _wheel_base(self) -> Image.Image | None:
        if self._base_radius == self._radius and self._base is not None:
            return self._base
        source = getattr(self._control_assets, "steering_wheel", None)
        if source is None:
            return None
        diameter = max(2, self._radius * 2)
        scaled = source.copy()
        scaled.thumbnail((diameter, diameter), Image.Resampling.BILINEAR)
        if scaled.mode != "RGBA":
            scaled = scaled.convert("RGBA")
        self._base = scaled
        self._base_radius = self._radius
        self._rotations.clear()
        return scaled

    def _draw_fallback(
        self, draw: ImageDraw.ImageDraw, centre: tuple[int, int], steering: float
    ) -> None:
        cx, cy = centre
        radius = self._radius
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(60, 60, 80, 255),
            width=4,
        )
        angle = -steering * math.radians(MAX_STEER_DEG)
        draw.line(
            (
                cx,
                cy,
                cx + int(math.sin(angle) * (radius - 6)),
                cy - int(math.cos(angle) * (radius - 6)),
            ),
            fill=NVIDIA_GREEN + (255,),
            width=4,
        )


class PedalsWidget:
    """Throttle and brake, as sprites when available and fill bars otherwise.

    Bars fill upward from the bottom, matching how a pedal travels.
    """

    def __init__(
        self,
        drive_state: Callable[[], DriveState],
        *,
        control_assets: Any | None = None,
        pedal_size: tuple[int, int] = (80, 160),
    ) -> None:
        self._drive_state = drive_state
        self._control_assets = control_assets
        self._pedal_width, self._pedal_height = pedal_size
        self._font = resolve_font(14)
        self._sprites: LRUCache = LRUCache(maxsize=16)

    def measure(self, panel_width: int) -> int:
        del panel_width
        return self._pedal_height + 30

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        rect: Rect,
    ) -> None:
        del frame
        state = self._drive_state()
        top = rect[1]
        centre_x = (rect[0] + rect[2]) // 2
        gap = 24
        throttle_x = centre_x + gap
        brake_x = centre_x - gap - self._pedal_width

        self._draw_pedal(
            canvas,
            draw,
            x=throttle_x,
            y=top,
            value=state.throttle,
            sprite=self._sprite("throttle", state.throttle),
            bar_color=NVIDIA_GREEN,
        )
        self._draw_pedal(
            canvas,
            draw,
            x=brake_x,
            y=top,
            value=state.brake,
            sprite=self._sprite("brake", state.brake),
            # Soft red reads as "stop" without competing with the amber
            # steering readout.
            bar_color=(220, 80, 80),
        )

        labels_y = top + self._pedal_height + 8
        for label_x, text in (
            (throttle_x + self._pedal_width // 2, f"Throttle {state.throttle:0.2f}"),
            (brake_x + self._pedal_width // 2, f"Brake {state.brake:0.2f}"),
        ):
            bbox = measure_text(self._font, text)
            draw.text(
                (label_x - (bbox[2] - bbox[0]) // 2 - bbox[0], labels_y - bbox[1]),
                text,
                fill=TEXT_COLOR,
                font=self._font,
            )

    def prepare(self, frame: DisplayFrame) -> None:
        del frame

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size

    def on_pointer(self, event: PointerEvent) -> bool:
        del event
        return False

    def close(self) -> None:
        return

    def _sprite(self, kind: str, value: float) -> Image.Image | None:
        pressed = value > _PEDAL_PRESSED_THRESHOLD
        attribute = f"{kind}_{'pressed' if pressed else 'unpressed'}"
        source = getattr(self._control_assets, attribute, None)
        if source is None:
            return None

        def build() -> Image.Image:
            scaled = source.copy()
            scaled.thumbnail(
                (self._pedal_width, self._pedal_height), Image.Resampling.BILINEAR
            )
            return scaled if scaled.mode == "RGBA" else scaled.convert("RGBA")

        return self._sprites.get_or_compute((attribute, self._pedal_width), build)

    def _draw_pedal(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        x: int,
        y: int,
        value: float,
        sprite: Image.Image | None,
        bar_color: tuple[int, int, int],
    ) -> None:
        if sprite is not None:
            # Aspect-fitting a wide sprite into a tall slot leaves it hugging
            # the top; centre it so both pedals sit at the same height.
            canvas.alpha_composite(
                sprite, (x, y + max(0, (self._pedal_height - sprite.height) // 2))
            )
            return

        width, height = self._pedal_width, self._pedal_height
        draw.rounded_rectangle(
            (x, y, x + width, y + height),
            radius=8,
            fill=(40, 40, 50, 255),
            outline=(80, 80, 90, 255),
            width=2,
        )
        inner_top, inner_bottom = y + 4, y + height - 4
        inner_height = inner_bottom - inner_top
        fraction = max(0.0, min(1.0, float(value)))
        fill_height = round(inner_height * fraction)
        if inner_height <= 0 or fill_height <= 0:
            return
        draw.rounded_rectangle(
            (x + 4, inner_bottom - fill_height, x + width - 4, inner_bottom),
            radius=4,
            fill=bar_color + (255,),
        )


__all__ = [
    "MAX_STEER_DEG",
    "WHEEL_ROTATION_QUANTUM_DEG",
    "DriveState",
    "PedalsWidget",
    "WheelWidget",
]
