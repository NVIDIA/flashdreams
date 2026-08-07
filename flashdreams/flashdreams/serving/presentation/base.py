# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Presenter, overlay, and input-sink contracts shared by presentation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from PIL import Image, ImageDraw

from flashdreams.serving.presentation.frame import DisplayFrame

Rect = tuple[int, int, int, int]
"""Axis-aligned ``(left, top, right, bottom)`` rectangle in canvas pixels."""

KeyAction = Literal["press", "release", "repeat"]
PointerAction = Literal["move", "press", "release"]


@dataclass(frozen=True, kw_only=True, slots=True)
class KeyEvent:
    """A key transition normalized away from any windowing toolkit."""

    key: str
    """Lower-case key name: ``w``, ``escape``, ``space``, ``1``, or ``up``.

    Arrow keys use the bare cardinal names and digit rows use the digit
    character, so overlays never see toolkit-specific spellings such as
    ``arrow_up`` or ``key1``.
    """

    action: KeyAction
    """Whether the key went down, came up, or auto-repeated while held.

    ``repeat`` is reported separately rather than folded into ``press``
    because some backends interleave a spurious release around each OS
    repeat, and only the consumer knows whether to debounce that.
    """

    timestamp_s: float
    """Monotonic arrival time, stamped when the backend observed the event."""


@dataclass(frozen=True, kw_only=True, slots=True)
class PointerEvent:
    """A pointer transition normalized away from any windowing toolkit."""

    action: PointerAction
    """Whether the pointer moved or a button went down or up."""

    position: tuple[int, int]
    """Pointer position in canvas pixels, rounded for integer hit-testing."""

    timestamp_s: float
    """Monotonic arrival time, stamped when the backend observed the event."""

    button: str | None = None
    """``left``, ``middle``, ``right``, or ``None`` for moves."""


@runtime_checkable
class PresenterBackend(Protocol):
    """One display target driven by an application's render loop.

    Pull-shaped rather than push-shaped: a presenter owns its own event
    pump and decides when the run is over, because a native window has to
    service OS events to stay responsive and a user closing that window is
    a termination signal the loop must observe.
    """

    @property
    def should_close(self) -> bool:
        """Whether the presenter wants the driving loop to stop."""
        ...

    def process_events(self) -> None:
        """Pump pending window/transport events once."""
        ...

    def present_frame(self, frame: DisplayFrame) -> None:
        """Display ``frame``, compositing overlay chrome over it."""
        ...

    def close(self) -> None:
        """Release window, GPU, and transport resources."""
        ...


@runtime_checkable
class SupportsPrepareFrame(Protocol):
    """Optional presenter capability for off-critical-path frame staging.

    Loops should feature-detect this so presenters that cannot usefully
    prefetch stay minimal.
    """

    def prepare_frame(self, frame: DisplayFrame) -> None:
        """Start any host materialization ``frame`` will need at present time."""
        ...


@runtime_checkable
class HudOverlay(Protocol):
    """Demo-owned chrome and interaction for a presenter's canvas.

    The presenter owns the window, swapchain, and camera composite; the demo
    owns layout, chrome pixels, and UI-local interactions. Model controls that
    the demo does not consume should flow into a ``UserInputSource`` rather
    than mutate a model integration directly.
    """

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        """Return the rectangle the camera image should occupy.

        Reserving less than the full canvas is how an overlay makes room
        for a side panel; the presenter letterboxes the image inside
        whatever rectangle it gets back.
        """
        ...

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        """Paint chrome onto ``canvas`` after the camera area is composited.

        Called every tick even when the overlay renders nothing, so
        per-tick bookkeeping such as control polling stays live while
        chrome is hidden.
        """
        ...

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        """Paint the camera area for a tick with no image yet."""
        ...

    def prepare(self, frame: DisplayFrame) -> None:
        """Stage anything in ``frame.overlay_data`` needed at draw time."""
        ...

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        """Discard chrome cached against the previous canvas size.

        Presenters reallocate the canvas on resize, so any sprite or panel an
        overlay cached for the old dimensions is stale. Overlays that cache
        nothing size-dependent can leave this empty.
        """
        ...

    def on_key(self, event: KeyEvent) -> bool:
        """Handle a key event.

        Returns:
            ``True`` when the overlay consumed the event, so a composite
            stops offering it to the layers underneath.
        """
        ...

    def on_pointer(self, event: PointerEvent) -> bool:
        """Handle a pointer event.

        Returns:
            ``True`` when the overlay consumed the event.
        """
        ...

    def close(self) -> None:
        """Release overlay-owned resources."""
        ...


class NullOverlay:
    """No-op chrome for a full-canvas camera view."""

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        return (0, 0, canvas_size[0], canvas_size[1])

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, frame, camera_area

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


__all__ = [
    "HudOverlay",
    "KeyAction",
    "KeyEvent",
    "NullOverlay",
    "PointerAction",
    "PointerEvent",
    "PresenterBackend",
    "Rect",
    "SupportsPrepareFrame",
]
