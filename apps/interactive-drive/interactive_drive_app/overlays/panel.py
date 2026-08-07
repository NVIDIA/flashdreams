# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Driving panel column: reserves the chrome area and seeds the row layout."""

from __future__ import annotations

from interactive_drive_app.overlays.theme import (
    LABEL_COLOR,
    NVIDIA_GREEN,
    PANEL_BG,
)
from PIL import Image, ImageDraw

from flashdreams.serving.presentation import (
    DisplayFrame,
    KeyEvent,
    PanelLayout,
    PointerEvent,
    Rect,
    measure_text,
    resolve_font,
)

PANEL_WIDTH = 500
"""Matches the legacy HUD so ported widgets keep their proportions."""

MIN_CAMERA_WIDTH = 320
"""Below this the panel is dropped entirely rather than squeezing the camera
into a sliver."""

_TOP_INSET = 10
_HEADER_HEIGHT = 26


class DrivingPanelOverlay:
    """Reserve the right-hand chrome column and prepare it for its widgets.

    Draws the panel background and title, then opens the shared
    :class:`PanelLayout` that every widget stacked after it positions
    against. Must be the first layer in the composite: later layers reserve
    rows from the cursor it establishes.
    """

    def __init__(
        self, layout: PanelLayout, *, title: str = "interactive-drive"
    ) -> None:
        self._layout = layout
        self._title = title
        self._font = resolve_font(18)

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        width, height = canvas_size
        if width - PANEL_WIDTH < MIN_CAMERA_WIDTH:
            return (0, 0, width, height)
        return (0, 0, width - PANEL_WIDTH, height)

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        del frame
        canvas_width, canvas_height = canvas.size
        panel_left = camera_area[2]
        if panel_left >= canvas_width:
            # Window too narrow for the panel; give widgets an empty layout so
            # they skip cleanly instead of drawing over the camera.
            self._layout.begin((canvas_width, 0, canvas_width, 0))
            return

        panel = (panel_left, 0, canvas_width, canvas_height)
        draw.rectangle(panel, fill=PANEL_BG + (255,))
        draw.line(
            (panel_left, 0, panel_left, canvas_height),
            fill=NVIDIA_GREEN + (255,),
            width=2,
        )

        self._layout.begin(panel, top_inset=_TOP_INSET)
        header = self._layout.reserve(_HEADER_HEIGHT)
        bbox = measure_text(self._font, self._title)
        draw.text(
            (header[0] + 14 - bbox[0], header[1] - bbox[1]),
            self._title,
            fill=LABEL_COLOR,
            font=self._font,
        )

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, camera_area

    def prepare(self, frame: DisplayFrame) -> None:
        del frame

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size

    def on_key(self, event: KeyEvent) -> bool:
        del event
        return False

    def on_pointer(self, event: PointerEvent) -> bool:
        del event
        return False

    def close(self) -> None:
        return


__all__ = ["MIN_CAMERA_WIDTH", "PANEL_WIDTH", "DrivingPanelOverlay"]
