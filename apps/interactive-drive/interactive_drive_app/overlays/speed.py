# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Speed readout overlay."""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.serving.presentation import (
    DisplayFrame,
    LRUCache,
    PointerEvent,
    Rect,
    measure_text,
    resolve_font,
)
from PIL import Image, ImageDraw

from interactive_drive_app.overlays.theme import LABEL_COLOR, NVIDIA_GREEN

MPS_TO_MPH = 2.2369362920544

_DIGIT_CACHE_SIZE = 64
"""One rendered chip per integer mph. Small images, and it saves re-rasterising
a 76pt glyph run every tick."""

_UNIT_GAP = 6


class SpeedWidget:
    """Large speed digit in miles per hour, with a ``mph`` label beneath.

    Reads its value through a callable rather than holding engine state, so
    the same overlay works against the simulation, a replay trace, or a test
    fixture.
    """

    def __init__(
        self,
        speed_mps: Callable[[], float],
        *,
        digit_size: int = 76,
        label_size: int = 18,
    ) -> None:
        self._speed_mps = speed_mps
        self._font_digit = resolve_font(digit_size)
        self._font_label = resolve_font(label_size)
        self._chips: LRUCache = LRUCache(maxsize=_DIGIT_CACHE_SIZE)
        self._row_height = digit_size + label_size + _UNIT_GAP + 8

    def measure(self, panel_width: int) -> int:
        del panel_width
        return self._row_height

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        rect: Rect,
    ) -> None:
        del frame
        mph = max(0, int(self._speed_mps() * MPS_TO_MPH))
        chip = self._chips.get_or_compute(mph, lambda: self._render_chip(mph))
        x = (rect[0] + rect[2]) // 2 - chip.width // 2
        y = rect[1]
        canvas.alpha_composite(chip, (x, y))

        label_bbox = measure_text(self._font_label, "mph")
        draw.text(
            (
                x + (chip.width - (label_bbox[2] - label_bbox[0])) // 2 - label_bbox[0],
                y + chip.height + _UNIT_GAP - label_bbox[1],
            ),
            "mph",
            fill=LABEL_COLOR,
            font=self._font_label,
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

    def _render_chip(self, mph: int) -> Image.Image:
        text = f"{mph:d}"
        bbox = measure_text(self._font_digit, text)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        chip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(chip).text(
            (-bbox[0], -bbox[1]), text, fill=NVIDIA_GREEN, font=self._font_digit
        )
        return chip


__all__ = ["MPS_TO_MPH", "SpeedWidget"]
