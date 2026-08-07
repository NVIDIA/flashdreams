# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Top-down BEV minimap overlay."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable

from loguru import logger
from omnidreams.interactive_drive.overlays.theme import LABEL_COLOR
from PIL import Image, ImageDraw

from flashdreams.serving.presentation import (
    DisplayFrame,
    KeyEvent,
    PanelLayout,
    PointerEvent,
    Rect,
    as_rgb_host_uint8,
    measure_text,
    prefetch_frame,
    resolve_font,
)

BEV_OVERLAY_KEY = "bev"
"""``DisplayFrame.overlay_data`` key carrying the lazy BEV frame."""

_MIN_HEIGHT = 100
_SIDE_MARGIN = 14
_BOTTOM_MARGIN = 12
_TOP_GAP = 12
_INNER_INSET = 4

_PanelKey = tuple[int, int, int, int]


class BevOverlay:
    """Minimap with an ego chevron, resized and recoloured off the render thread.

    Materializing and filtering the BEV frame costs several milliseconds, so a
    worker does it and the presenter draws whatever the worker last finished.
    A frame or two of staleness on a minimap is invisible; a stall on the
    render thread is not.
    """

    def __init__(
        self,
        layout: PanelLayout,
        *,
        marker_y_fraction: Callable[[], float] = lambda: 0.5,
        recolor: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self._layout = layout
        self._marker_y_fraction = marker_y_fraction
        self._recolor = recolor or (lambda image: image)
        self._font = resolve_font(14)
        self._source: object | None = None
        self._source_generation = 0
        self._prepared_source_key: object | None = None
        self._epoch = 0
        self._cache: Image.Image | None = None
        self._cache_key: _PanelKey | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="interactive-drive-bev-panel"
        )
        self._pending: (
            concurrent.futures.Future[tuple[_PanelKey, Image.Image]] | None
        ) = None

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        return (0, 0, canvas_size[0], canvas_size[1])

    def prepare(self, frame: DisplayFrame) -> None:
        """Stage the incoming BEV frame off the presentation thread."""
        source = frame.overlay_data.get(BEV_OVERLAY_KEY)
        if source is None:
            return
        group_key = getattr(source, "source_group_key", None)
        key = group_key() if callable(group_key) else id(source)
        if key != self._prepared_source_key:
            prefetch_frame(source)
            self._prepared_source_key = key
        if source is not self._source:
            self._source = source
            self._source_generation += 1

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        del frame, camera_area
        row = self._layout.reserve(
            self._layout.remaining - _BOTTOM_MARGIN, gap=_TOP_GAP
        )
        left, top, right, bottom = row
        if bottom - top < _MIN_HEIGHT:
            return
        panel = (left + _SIDE_MARGIN, top, right - _SIDE_MARGIN, bottom)
        inner = (
            panel[0] + _INNER_INSET,
            panel[1] + _INNER_INSET,
            panel[2] - _INNER_INSET,
            panel[3] - _INNER_INSET,
        )
        inner_width = inner[2] - inner[0]
        inner_height = inner[3] - inner[1]
        if inner_width <= 0 or inner_height <= 0:
            return

        if self._source is None:
            self._draw_waiting(draw, panel)
            return

        image = self._panel_image((inner_width, inner_height))
        if image is not None:
            canvas.paste(image, (inner[0], inner[1]))

        marker_y = inner[1] + int(inner_height * self._marker_y_fraction())
        _draw_marker(
            draw,
            inner[0] + inner_width // 2,
            marker_y,
            max(10, min(inner_width, inner_height) // 14),
        )

    def draw_placeholder(
        self, canvas: Image.Image, draw: ImageDraw.ImageDraw, *, camera_area: Rect
    ) -> None:
        del canvas, draw, camera_area

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        del canvas_size
        self._cache = None
        self._cache_key = None

    def on_key(self, event: KeyEvent) -> bool:
        del event
        return False

    def on_pointer(self, event: PointerEvent) -> bool:
        del event
        return False

    def reset(self) -> None:
        """Drop the current scene's BEV so the next one does not ghost it."""
        self._epoch += 1
        self._source = None
        self._prepared_source_key = None
        self._source_generation = 0
        self._cache = None
        self._cache_key = None
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _draw_waiting(self, draw: ImageDraw.ImageDraw, panel: Rect) -> None:
        text = "WAITING FOR BEV..."
        bbox = measure_text(self._font, text)
        centre_x = (panel[0] + panel[2]) // 2
        centre_y = (panel[1] + panel[3]) // 2
        draw.text(
            (
                centre_x - (bbox[2] - bbox[0]) // 2 - bbox[0],
                centre_y - (bbox[3] - bbox[1]) // 2 - bbox[1],
            ),
            text,
            fill=LABEL_COLOR,
            font=self._font,
        )

    def _panel_image(self, target_size: tuple[int, int]) -> Image.Image | None:
        key = (self._epoch, self._source_generation, *target_size)
        pending = self._pending
        if pending is not None and pending.done():
            try:
                self._cache_key, self._cache = pending.result()
            except Exception as exc:  # noqa: BLE001 -- a stale minimap is survivable
                logger.warning(f"[presenter] BEV panel processing failed: {exc}")
            self._pending = None

        if (
            key != self._cache_key
            and self._pending is None
            and self._source is not None
        ):
            self._pending = self._executor.submit(
                _build_panel, key, self._source, target_size, self._recolor
            )

        cached_key = self._cache_key
        if (
            cached_key is not None
            and cached_key[0] == self._epoch
            and cached_key[2:] == target_size
        ):
            return self._cache
        return None


def _build_panel(
    key: _PanelKey,
    source: object,
    target_size: tuple[int, int],
    recolor: Callable[[Image.Image], Image.Image],
) -> tuple[_PanelKey, Image.Image]:
    """Materialize, cover-fit, and recolour a BEV frame away from presentation."""
    image = Image.fromarray(as_rgb_host_uint8(source), mode="RGB")
    target_width, target_height = target_size
    scale = max(target_width / image.width, target_height / image.height)
    scaled = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    crop_left = (scaled.width - target_width) // 2
    crop_top = (scaled.height - target_height) // 2
    cropped = scaled.crop(
        (crop_left, crop_top, crop_left + target_width, crop_top + target_height)
    )
    return key, recolor(cropped)


def _draw_marker(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    """Ego chevron: drop shadow, white disc, forward-pointing arrow."""
    shadow = size + 4
    draw.ellipse(
        (cx - shadow, cy - shadow + 2, cx + shadow, cy + shadow + 2), fill=(0, 0, 0, 60)
    )
    draw.ellipse(
        (cx - size, cy - size, cx + size, cy + size), fill=(255, 255, 255, 255)
    )
    chevron = size - 4
    draw.polygon(
        [
            (cx, cy - chevron),
            (cx - int(chevron * 0.7), cy + int(chevron * 0.55)),
            (cx, cy + int(chevron * 0.18)),
            (cx + int(chevron * 0.7), cy + int(chevron * 0.55)),
        ],
        fill=(66, 133, 244, 255),
    )


__all__ = ["BEV_OVERLAY_KEY", "BevOverlay"]
