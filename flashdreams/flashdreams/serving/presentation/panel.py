# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A panel column that stacks widgets, as one overlay owning its children."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from PIL import Image, ImageDraw

from flashdreams.serving.presentation.base import KeyEvent, PointerEvent, Rect
from flashdreams.serving.presentation.frame import DisplayFrame


@runtime_checkable
class PanelWidget(Protocol):
    """One stacked element inside a :class:`PanelOverlay`.

    Widgets are told where to draw rather than negotiating position, so a
    panel can be reordered or a widget dropped without any of the others
    noticing.
    """

    def measure(self, panel_width: int) -> int | None:
        """Height this widget needs, or ``None`` to fill what is left.

        At most one widget should return ``None``; a later one that does gets
        nothing, because the space is already claimed.
        """
        ...

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        rect: Rect,
    ) -> None:
        """Paint into ``rect``, the row the panel assigned this widget."""
        ...

    def prepare(self, frame: DisplayFrame) -> None:
        """Stage anything in ``frame`` needed at draw time."""
        ...

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        """Discard anything cached against the previous canvas size."""
        ...

    def on_pointer(self, event: PointerEvent) -> bool:
        """Handle a pointer event; ``True`` stops it reaching other widgets."""
        ...

    def close(self) -> None:
        """Release widget-owned resources."""
        ...


@dataclass(kw_only=True)
class PanelOverlay:
    """Reserve a side column and stack widgets down it.

    The panel measures its children, assigns each a rectangle, and draws them
    in order, so position is computed in one pass by the only object that
    knows the whole column. Widgets therefore never coordinate with their
    siblings, and reordering the column means reordering ``children``.
    """

    children: tuple[PanelWidget, ...]

    width: int = 500
    """Column width in pixels."""

    min_camera_width: int = 320
    """Below this the column is dropped rather than crushing the camera."""

    side: Literal["left", "right"] = "right"
    """Which edge of the canvas to reserve."""

    background: tuple[int, int, int] = (25, 25, 35)
    accent: tuple[int, int, int] = (118, 185, 0)

    top_inset: int = 10
    gap: int = 12

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        width, height = canvas_size
        if width - self.width < self.min_camera_width:
            return (0, 0, width, height)
        if self.side == "left":
            return (self.width, 0, width, height)
        return (0, 0, width - self.width, height)

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        canvas_width, canvas_height = canvas.size
        if self.side == "left":
            column = (0, 0, camera_area[0], canvas_height)
        else:
            column = (camera_area[2], 0, canvas_width, canvas_height)
        if column[2] - column[0] <= 0:
            return

        draw.rectangle(column, fill=self.background + (255,))
        edge = column[0] if self.side == "right" else column[2]
        draw.line((edge, 0, edge, canvas_height), fill=self.accent + (255,), width=2)

        for widget, rect in self._assign_rows(column):
            widget.draw(canvas, draw, frame=frame, rect=rect)

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, camera_area

    def prepare(self, frame: DisplayFrame) -> None:
        for widget in self.children:
            widget.prepare(frame)

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        for widget in self.children:
            widget.on_canvas_resized(canvas_size)

    def on_key(self, event: KeyEvent) -> bool:
        del event
        return False

    def on_pointer(self, event: PointerEvent) -> bool:
        # Front to back: a widget drawn later sits on top, so it gets the
        # click first.
        return any(widget.on_pointer(event) for widget in reversed(self.children))

    def close(self) -> None:
        errors: list[BaseException] = []
        for widget in self.children:
            try:
                widget.close()
            except BaseException as exc:  # noqa: BLE001 -- close every widget first
                errors.append(exc)
        if errors:
            raise errors[0]

    def _assign_rows(self, column: Rect) -> list[tuple[PanelWidget, Rect]]:
        """Measure children once, then hand each the row it gets.

        A widget measuring ``None`` takes whatever is unclaimed after the
        fixed-height ones, which is how a minimap fills the space below the
        controls without knowing what sits above it.
        """
        left, top, right, bottom = column
        panel_width = right - left
        heights = [widget.measure(panel_width) for widget in self.children]
        fixed = sum(height for height in heights if height is not None)
        gaps = self.gap * max(0, len(self.children) - 1)
        flexible = max(0, (bottom - top - self.top_inset) - fixed - gaps)

        rows: list[tuple[PanelWidget, Rect]] = []
        cursor = top + self.top_inset
        for widget, height in zip(self.children, heights, strict=True):
            if cursor >= bottom:
                break
            resolved = flexible if height is None else height
            if height is None:
                flexible = 0
            row_bottom = min(cursor + resolved, bottom)
            if row_bottom > cursor:
                rows.append((widget, (left, cursor, right, row_bottom)))
            cursor = row_bottom + self.gap
        return rows


__all__ = ["PanelOverlay", "PanelWidget"]
