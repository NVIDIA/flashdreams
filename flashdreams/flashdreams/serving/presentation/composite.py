# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stack several overlays over one presenter."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from flashdreams.serving.presentation.base import (
    HudOverlay,
    KeyEvent,
    PointerEvent,
    Rect,
)
from flashdreams.serving.presentation.frame import DisplayFrame


@dataclass(frozen=True, kw_only=True, slots=True)
class CompositeOverlay:
    """Draw several overlays in order over one presenter.

    Lets a demo assemble its chrome from independent pieces -- a speed
    readout, a minimap, a scene picker -- rather than one class that owns
    every widget. Each piece is separately testable, and a different demo can
    take a different subset without inheriting the rest.

    Layers draw back to front, so later layers paint over earlier ones. Input
    is offered in reverse, front to back, so the layer visually on top gets
    first refusal on a click.
    """

    layers: tuple[HudOverlay, ...]

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        """Intersect every layer's requested camera area.

        A layer reserving screen space returns a smaller rectangle; one with
        no spatial claim returns the whole canvas. Intersecting means each
        layer's reservation is honoured without any of them needing to know
        what the others asked for.
        """
        width, height = canvas_size
        left, top, right, bottom = 0, 0, width, height
        for layer in self.layers:
            layer_left, layer_top, layer_right, layer_bottom = layer.camera_area(
                canvas_size
            )
            left = max(left, layer_left)
            top = max(top, layer_top)
            right = min(right, layer_right)
            bottom = min(bottom, layer_bottom)
        # Collapsed intersections would break fit geometry downstream; a
        # degenerate camera area means the layers disagree, so fall back to
        # the full canvas rather than presenting nothing.
        if right <= left or bottom <= top:
            return (0, 0, width, height)
        return (left, top, right, bottom)

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        for layer in self.layers:
            layer.draw(canvas, draw, frame=frame, camera_area=camera_area)

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        for layer in self.layers:
            layer.draw_placeholder(canvas, draw, camera_area=camera_area)

    def prepare(self, frame: DisplayFrame) -> None:
        for layer in self.layers:
            layer.prepare(frame)

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        for layer in self.layers:
            layer.on_canvas_resized(canvas_size)

    def on_key(self, event: KeyEvent) -> bool:
        return any(layer.on_key(event) for layer in reversed(self.layers))

    def on_pointer(self, event: PointerEvent) -> bool:
        return any(layer.on_pointer(event) for layer in reversed(self.layers))

    def close(self) -> None:
        errors: list[BaseException] = []
        for layer in self.layers:
            try:
                layer.close()
            except BaseException as exc:  # noqa: BLE001 -- close every layer first
                errors.append(exc)
        if errors:
            raise errors[0]


__all__ = ["CompositeOverlay"]
