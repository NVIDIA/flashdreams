# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canvas allocation, font/text measurement, and fit geometry for PIL chrome."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from flashdreams.serving.presentation.base import Rect

_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/gnu-free/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
"""Host TrueType paths probed in order; PIL ships no system-font resolver.

Deliberately broad: minimal server images often carry none of DejaVu,
Liberation, or Noto, and the bitmap fallback renders digits as empty boxes.
"""

_fallback_warned = False


def allocate_canvas(
    width: int, height: int, *, background: tuple[int, int, int]
) -> tuple[np.ndarray, Image.Image]:
    """Allocate a chrome buffer and a PIL Image view sharing its memory.

    ``Image.frombuffer`` (RGBA "raw", Pillow >= 9) aliases ``buf``, so PIL
    draws write into it directly and the buffer can go straight to a GPU
    upload with no PIL-to-numpy memcpy. ``readonly = 0`` is required or
    ``ImageDraw`` rejects the image as a draw target.

    Returns:
        The ``[height, width, 4]`` uint8 buffer and the aliasing image.
    """
    buf = np.empty((height, width, 4), dtype=np.uint8)
    buf[..., :3] = background
    buf[..., 3] = 255
    img = Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", 0, 1)
    img.readonly = 0
    return buf, img


class LRUCache(OrderedDict):
    """Ordered-dict-backed LRU for per-bucket render artefacts.

    Keeps rotation / digit / sprite caches from growing without bound; the
    move-to-end on every hit is what makes the eviction order correct.
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = int(maxsize)

    def get_or_compute(self, key: Any, build: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, building and storing it on a miss."""
        existing = self.get(key)
        if existing is not None:
            self.move_to_end(key)
            return existing
        value = build()
        self[key] = value
        if len(self) > self._maxsize:
            self.popitem(last=False)
        return value


def resolve_font(size: int) -> Any:
    """Load a host TrueType font at ``size``, falling back to PIL's default.

    Warns once when no TrueType font is found: the bitmap fallback has patchy
    glyph coverage and draws digits as boxes, which reads as a corrupt overlay
    rather than a missing font.
    """
    global _fallback_warned

    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue

    if not _fallback_warned:
        logger.warning(
            "[presentation] no TrueType font found; overlay text falls back to "
            "PIL's bitmap font and will render digits as boxes. Install one, "
            "e.g. `apt-get install fonts-dejavu-core`.",
        )
        _fallback_warned = True
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def measure_text(font: Any, text: str) -> Rect:
    """Return ``text``'s bounding box, tolerating the legacy bitmap fallback."""
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    # The 9.x-era bitmap fallback only has ``getsize``.
    width, height = font.getsize(text)  # type: ignore[attr-defined]
    return (0, 0, int(width), int(height))


def truncate_text_to_width(
    font: Any, text: str, max_width: int, ellipsis: str = "\u2026"
) -> str:
    """Shrink ``text`` (with a trailing ellipsis) until it fits ``max_width`` pixels.

    PIL doesn't auto-clip ``ImageDraw.text``, so an over-long label would
    bleed out of its panel; progressively shorter prefixes are measured
    until one fits.
    """
    bbox = measure_text(font, text)
    if bbox[2] - bbox[0] <= max_width:
        return text
    # Greedy shrink. Labels are short, so re-measuring per truncation is fine.
    for end in range(len(text), 0, -1):
        candidate = text[:end] + ellipsis
        candidate_bbox = measure_text(font, candidate)
        if candidate_bbox[2] - candidate_bbox[0] <= max_width:
            return candidate
    return ellipsis


def fit_rect(*, source_size: tuple[int, int], area: Rect) -> Rect | None:
    """Centre ``source_size`` inside ``area``, downscaling only when it overflows.

    Frames smaller than the area keep their native pixels rather than being
    upscaled, so the surrounding gap stays as letterbox background.

    Args:
        source_size: Source ``(width, height)`` in pixels.
        area: Destination rectangle in canvas pixels.

    Returns:
        The centred destination rectangle, or ``None`` when either the
        source or the area is degenerate.
    """
    source_width, source_height = source_size
    left, top, right, bottom = area
    area_width = right - left
    area_height = bottom - top
    if min(source_width, source_height, area_width, area_height) <= 0:
        return None
    scale = min(1.0, area_width / source_width, area_height / source_height)
    fit_width = max(1, int(source_width * scale))
    fit_height = max(1, int(source_height * scale))
    offset_x = left + (area_width - fit_width) // 2
    offset_y = top + (area_height - fit_height) // 2
    return (offset_x, offset_y, offset_x + fit_width, offset_y + fit_height)


def draw_status_overlay(
    draw: ImageDraw.ImageDraw,
    *,
    area: Rect,
    message: str,
    font: Any,
    text_color: tuple[int, int, int],
    padding: int = 24,
) -> None:
    """Draw ``message`` in a centred callout box over ``area``."""
    left, top, right, bottom = area
    centre_x, centre_y = (left + right) // 2, (top + bottom) // 2
    bbox = measure_text(font, message)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # PIL's rectangle writes the alpha channel straight through the
    # alpha-composited canvas, so this stays a translucent callout.
    draw.rectangle(
        (
            centre_x - text_width // 2 - padding,
            centre_y - text_height // 2 - padding,
            centre_x + text_width // 2 + padding,
            centre_y + text_height // 2 + padding,
        ),
        fill=(20, 20, 20, 230),
        outline=(240, 240, 240, 255),
        width=2,
    )
    draw.text(
        (centre_x - text_width // 2 - bbox[0], centre_y - text_height // 2 - bbox[1]),
        message,
        fill=text_color,
        font=font,
    )


__all__ = [
    "LRUCache",
    "allocate_canvas",
    "draw_status_overlay",
    "fit_rect",
    "measure_text",
    "resolve_font",
    "truncate_text_to_width",
]
