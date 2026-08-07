# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Panel dimensions and the title row at the top of the driving chrome."""

from __future__ import annotations

from flashdreams.serving.presentation import (
    DisplayFrame,
    PointerEvent,
    Rect,
    measure_text,
    resolve_font,
)
from PIL import Image, ImageDraw

from interactive_drive_app.overlays.theme import LABEL_COLOR

PANEL_WIDTH = 500
"""Matches the legacy HUD so ported widgets keep their proportions."""

MIN_CAMERA_WIDTH = 320
"""Below this the panel is dropped entirely rather than squeezing the camera
into a sliver."""

_HEADER_HEIGHT = 26


class TitleWidget:
    """Static caption identifying the app in the panel's first row."""

    def __init__(self, title: str = "interactive-drive") -> None:
        self._title = title
        self._font = resolve_font(18)

    def measure(self, panel_width: int) -> int:
        del panel_width
        return _HEADER_HEIGHT

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        rect: Rect,
    ) -> None:
        del canvas, frame
        bbox = measure_text(self._font, self._title)
        draw.text(
            (rect[0] + 14 - bbox[0], rect[1] - bbox[1]),
            self._title,
            fill=LABEL_COLOR,
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


__all__ = ["MIN_CAMERA_WIDTH", "PANEL_WIDTH", "TitleWidget"]
