# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scene and variant header bars, plus the post-processing toggle."""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.serving.presentation import (
    DisplayFrame,
    PointerEvent,
    Rect,
    resolve_font,
    truncate_text_to_width,
)
from PIL import Image, ImageDraw

from interactive_drive_app.overlays.theme import (
    ACTIVE_BG,
    HEADER_BG,
    LABEL_COLOR,
    NVIDIA_GREEN,
    PANEL_BG,
    TEXT_COLOR,
)

_BAR_HEIGHT = 32
_BAR_GAP = 4
_MARGIN = 10
_DOT_INSET = 8
_LABEL_INSET = 26
_ARROW_INSET = 24


class SceneHeaderWidget:
    """Scene and variant bars with an optional post-processing toggle.

    The bars render the current selection; they are not a picker. Switching
    scenes needs an outer loop that tears down a rollout and re-enters over
    the same window, which the single-scene path this presenter currently
    runs on does not have. The toggle is live, because it only has to call
    back into the running backend.
    """

    def __init__(
        self,
        *,
        scene_label: Callable[[], str],
        variant_label: Callable[[], str],
        postprocess_preset: str = "",
        postprocess_enabled: bool = False,
        on_postprocess_toggled: Callable[[bool], None] | None = None,
    ) -> None:
        self._scene_label = scene_label
        self._variant_label = variant_label
        self._preset = postprocess_preset
        self._postprocess_enabled = postprocess_enabled
        self._on_toggled = on_postprocess_toggled
        self._font = resolve_font(18)
        self._toggle_rect: Rect | None = None
        self._chrome: Image.Image | None = None
        self._chrome_key: tuple[object, ...] | None = None

    def set_postprocess_control(
        self,
        *,
        preset: str,
        enabled: bool,
        callback: Callable[[bool], None],
    ) -> None:
        """Bind the toggle to the pipeline after the window already exists.

        The presenter captures its overlay at construction, but the pipeline
        this drives is built afterwards, so the binding arrives late. Without
        a preset the row is not drawn at all.
        """
        self._preset = preset
        self._postprocess_enabled = bool(enabled and preset)
        self._on_toggled = callback
        self._chrome = None
        self._chrome_key = None

    def measure(self, panel_width: int) -> int:
        del panel_width
        return self._rows() * (_BAR_HEIGHT + _BAR_GAP)

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        rect: Rect,
    ) -> None:
        del draw, frame
        rows = self._rows()
        left, top, right, bottom = rect
        width = right - left - _MARGIN * 2
        if bottom - top <= 0 or width <= 0:
            self._toggle_rect = None
            return

        chrome = self._chrome_image((width, rows * (_BAR_HEIGHT + _BAR_GAP)))
        canvas.alpha_composite(chrome, (left + _MARGIN, top))

        if self._preset:
            toggle_top = top + 2 * (_BAR_HEIGHT + _BAR_GAP)
            self._toggle_rect = (
                left + _MARGIN,
                toggle_top,
                left + _MARGIN + width,
                toggle_top + _BAR_HEIGHT,
            )
        else:
            self._toggle_rect = None

    def prepare(self, frame: DisplayFrame) -> None:
        del frame

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size
        self._chrome = None
        self._chrome_key = None

    def on_pointer(self, event: PointerEvent) -> bool:
        if event.action != "press" or event.button != "left":
            return False
        rect = self._toggle_rect
        if rect is None or not _contains(rect, event.position):
            return False
        self._postprocess_enabled = not self._postprocess_enabled
        if self._on_toggled is not None:
            self._on_toggled(self._postprocess_enabled)
        return True

    def close(self) -> None:
        return

    def _rows(self) -> int:
        return 3 if self._preset else 2

    def _chrome_image(self, size: tuple[int, int]) -> Image.Image:
        """Render the bars, cached until their contents change.

        Only the label text and toggle state vary, so re-rasterising every
        tick would spend milliseconds redrawing identical pixels.
        """
        key = (
            size,
            self._scene_label(),
            self._variant_label(),
            self._preset,
            self._postprocess_enabled,
        )
        if key == self._chrome_key and self._chrome is not None:
            return self._chrome

        width, height = size
        chrome = Image.new("RGBA", (width, height), PANEL_BG + (255,))
        draw = ImageDraw.Draw(chrome)

        # Scene bar: status dot, truncated label, and a disabled-looking
        # caret so the bar reads as a selector even while it is read-only.
        draw.rounded_rectangle(
            (0, 0, width, _BAR_HEIGHT), radius=6, fill=HEADER_BG + (255,)
        )
        draw.ellipse((_DOT_INSET, 11, _DOT_INSET + 10, 21), fill=NVIDIA_GREEN + (255,))
        draw.text(
            (_LABEL_INSET, 6),
            truncate_text_to_width(
                self._font, self._scene_label(), width - _LABEL_INSET - 30
            ),
            fill=TEXT_COLOR,
            font=self._font,
        )
        draw.text(
            (width - _ARROW_INSET, 6), "\u25bc", fill=LABEL_COLOR, font=self._font
        )

        variant_top = _BAR_HEIGHT + _BAR_GAP
        draw.rounded_rectangle(
            (0, variant_top, width, variant_top + _BAR_HEIGHT),
            radius=6,
            fill=HEADER_BG + (255,),
        )
        draw.text(
            (_MARGIN, variant_top + 6),
            truncate_text_to_width(
                self._font, f"Variant: {self._variant_label()}", width - _MARGIN * 2
            ),
            fill=TEXT_COLOR,
            font=self._font,
        )

        if self._preset:
            toggle_top = 2 * (_BAR_HEIGHT + _BAR_GAP)
            draw.rounded_rectangle(
                (0, toggle_top, width, toggle_top + _BAR_HEIGHT),
                radius=6,
                fill=(ACTIVE_BG if self._postprocess_enabled else HEADER_BG) + (255,),
            )
            state = "on" if self._postprocess_enabled else "off"
            draw.text(
                (_MARGIN, toggle_top + 6),
                truncate_text_to_width(
                    self._font,
                    f"Post-process: {self._preset} ({state})",
                    width - _MARGIN * 2,
                ),
                fill=TEXT_COLOR,
                font=self._font,
            )

        self._chrome = chrome
        self._chrome_key = key
        return chrome


def _contains(rect: Rect, position: tuple[int, int]) -> bool:
    x, y = position
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


__all__ = ["SceneHeaderWidget"]
