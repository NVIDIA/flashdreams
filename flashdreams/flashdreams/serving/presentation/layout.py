# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared vertical layout for widgets stacked inside a reserved panel."""

from __future__ import annotations

from flashdreams.serving.presentation.base import Rect


class PanelLayout:
    """Hands out stacked rows inside a panel column.

    Overlays in a :class:`~flashdreams.serving.presentation.CompositeOverlay`
    draw in order but cannot see each other, so widgets sharing a panel need
    somewhere to agree on vertical position. A layout object passed to each of
    them turns that into a running cursor: the panel layer calls :meth:`begin`
    once per frame, then each widget reserves the height it needs in draw
    order.

    Reserving past the bottom of the panel yields a zero-height row rather
    than raising, so a window too short for every widget drops the ones that
    no longer fit instead of drawing them over each other.
    """

    def __init__(self) -> None:
        self._rect: Rect = (0, 0, 0, 0)
        self._cursor = 0

    def begin(self, rect: Rect, *, top_inset: int = 0) -> None:
        """Start a frame, resetting the cursor to the top of ``rect``."""
        self._rect = rect
        self._cursor = rect[1] + top_inset

    def reserve(self, height: int, *, gap: int = 0) -> Rect:
        """Claim the next ``height`` pixels, after ``gap`` of spacing."""
        left, _top, right, bottom = self._rect
        row_top = min(self._cursor + gap, bottom)
        row_bottom = min(row_top + max(0, height), bottom)
        self._cursor = row_bottom
        return (left, row_top, right, row_bottom)

    @property
    def rect(self) -> Rect:
        """The full panel rectangle for this frame."""
        return self._rect

    @property
    def width(self) -> int:
        return self._rect[2] - self._rect[0]

    @property
    def center_x(self) -> int:
        return (self._rect[0] + self._rect[2]) // 2

    @property
    def remaining(self) -> int:
        """Unreserved height left below the cursor."""
        return max(0, self._rect[3] - self._cursor)


__all__ = ["PanelLayout"]
