# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Camera and overlay chrome composited onto one canvas, independent of transport."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from flashdreams.serving.presentation.base import HudOverlay, Rect
from flashdreams.serving.presentation.canvas import (
    allocate_canvas,
    draw_status_overlay,
    fit_rect,
    resolve_font,
)
from flashdreams.serving.presentation.frame import DisplayFrame, as_rgb_host_uint8

CameraMode = Literal["composite", "deferred", "transparent"]
"""How the camera region should be filled.

``composite`` draws the image on the CPU and is what a transport without its
own GPU compositor wants. ``deferred`` paints only the letterbox bars because
the caller stamps the image in afterwards on the GPU. ``transparent`` leaves a
hole for a caller that alpha-blends the chrome over the image instead.
"""


class FrameCompositor:
    """Own the canvas, the retained camera image, and the chrome draw order.

    Split out of the local-window presenter so any transport can produce the
    same pixels: a windowed presenter uploads :attr:`canvas_buffer` to a
    swapchain, a streaming one hands it to an encoder. Nothing here knows
    which.
    """

    def __init__(
        self,
        *,
        overlay: HudOverlay,
        background: tuple[int, int, int],
        text_color: tuple[int, int, int],
        size: tuple[int, int],
        status_font_size: int = 44,
    ) -> None:
        self._overlay = overlay
        self._background = background
        self._text_color = text_color
        self._status_font = resolve_font(status_font_size)
        self._buffer, self._canvas = allocate_canvas(*size, background=background)
        self._camera_image: Image.Image | None = None
        self._camera_source_size: tuple[int, int] | None = None
        self._camera_generation = 0
        self._resize_cache: Image.Image | None = None
        self._resize_cache_key: tuple[int, int, int] | None = None

    ## Canvas

    @property
    def canvas(self) -> Image.Image:
        """PIL view of the canvas; chrome draws into this."""
        return self._canvas

    @property
    def canvas_buffer(self) -> np.ndarray:
        """``[H, W, 4]`` uint8 aliasing :attr:`canvas`, ready to upload or encode."""
        return self._buffer

    @property
    def size(self) -> tuple[int, int]:
        return self._canvas.size

    def resize(self, width: int, height: int) -> None:
        """Reallocate the canvas and drop caches keyed on the old size."""
        self._buffer, self._canvas = allocate_canvas(
            width, height, background=self._background
        )
        self._resize_cache = None
        self._resize_cache_key = None
        self._overlay.on_canvas_resized((width, height))

    ## Camera

    @property
    def camera_image(self) -> Image.Image | None:
        return self._camera_image

    @property
    def camera_source_size(self) -> tuple[int, int] | None:
        """Source ``(width, height)``, or ``None`` before the first frame."""
        return self._camera_source_size

    @property
    def camera_generation(self) -> int:
        """Counter bumped whenever the retained frame changes.

        Producers reuse their scratch buffers, so object identity is stable
        across frames with different pixels and cannot be used to detect a new
        one. Callers holding derived copies -- a GPU texture, an encoded
        surface -- compare this instead of the image itself.
        """
        return self._camera_generation

    def camera_area(self) -> Rect:
        return self._overlay.camera_area(self._canvas.size)

    def set_camera(self, image: Any) -> None:
        """Retain ``image`` as the frame to display, materializing it to host."""
        rgb = as_rgb_host_uint8(image)
        # ``Image.fromarray`` over a contiguous buffer is zero-copy at the C
        # level; this image is only ever a paste source, which does not copy
        # either.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        self._camera_image = Image.fromarray(rgb, mode="RGB")
        height, width = rgb.shape[:2]
        self._camera_source_size = (width, height)
        self._camera_generation += 1
        self._resize_cache = None
        self._resize_cache_key = None

    def reset_camera(self) -> None:
        """Forget the retained frame so a new producer does not ghost the old."""
        self._camera_image = None
        self._camera_source_size = None
        self._camera_generation += 1
        self._resize_cache = None
        self._resize_cache_key = None

    ## Render

    def render(self, frame: DisplayFrame, *, camera_mode: CameraMode) -> Rect:
        """Composite one tick into the canvas and return the camera area.

        No full-canvas clear: the overlay and camera paths cover their own
        regions every frame and the letterbox bars stay at the background
        colour, which saves a 2 MP RGBA fill per tick at 1080p.
        """
        canvas = self._canvas
        camera_area = self.camera_area()
        draw = ImageDraw.Draw(canvas)
        background = self._background

        if camera_mode == "transparent":
            draw.rectangle(camera_area, fill=(0, 0, 0, 0))
        elif self._camera_image is None:
            # Wipe first so the previous tick does not ghost behind the
            # placeholder; placeholder ticks only, so cheaper than an
            # always-on full-canvas clear.
            draw.rectangle(camera_area, fill=background + (255,))
            self._overlay.draw_placeholder(canvas, draw, camera_area=camera_area)
        elif camera_mode == "deferred":
            # The caller stamps the image into the centred fit rect after
            # this canvas is uploaded; repaint only the letterbox bars so
            # they do not show the previous frame when the rect resizes.
            draw.rectangle(camera_area, fill=background + (255,))
        else:
            self._draw_camera(canvas, camera_area)

        self._overlay.draw(canvas, draw, frame=frame, camera_area=camera_area)

        if frame.status_message:
            draw_status_overlay(
                draw,
                area=camera_area,
                message=frame.status_message,
                font=self._status_font,
                text_color=self._text_color,
            )
        return camera_area

    def _draw_camera(self, canvas: Image.Image, area: Rect) -> None:
        camera = self._camera_image
        if camera is None:
            return
        target = fit_rect(source_size=camera.size, area=area)
        if target is None:
            return
        left, top, right, bottom = target
        target_w, target_h = right - left, bottom - top
        cache_key = (id(camera), target_w, target_h)
        if cache_key != self._resize_cache_key or self._resize_cache is None:
            if (target_w, target_h) == camera.size:
                resized = camera
            else:
                resized = camera.resize(
                    (target_w, target_h),
                    Image.Resampling.LANCZOS
                    if target_w < camera.size[0]
                    else Image.Resampling.BILINEAR,
                )
            self._resize_cache = resized
            self._resize_cache_key = cache_key
        else:
            resized = self._resize_cache
        if resized.mode == "RGBA":
            canvas.alpha_composite(resized, (left, top))
        else:
            canvas.paste(resized, (left, top))


__all__ = ["CameraMode", "FrameCompositor"]
