# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the model-agnostic presentation contracts, canvas, and geometry."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from omnidreams.presentation import (
    DisplayFrame,
    HudOverlay,
    InputSink,
    KeyEvent,
    LRUCache,
    PointerEvent,
    Rect,
    allocate_canvas,
    as_rgb_host_uint8,
    fit_rect,
    has_cuda_tensor,
    rgb_source_size,
    truncate_text_to_width,
)
from omnidreams.presentation.canvas import resolve_font
from PIL import Image, ImageDraw

pytestmark = pytest.mark.ci_cpu

BLACK = (0, 0, 0)


class _RecordingOverlay:
    """Minimal overlay that records the calls a presenter would make."""

    def __init__(self, *, reserved_width: int = 0, consume: bool = False) -> None:
        self._reserved_width = reserved_width
        self._consume = consume
        self.drawn: list[DisplayFrame] = []
        self.placeholders = 0
        self.prepared: list[DisplayFrame] = []
        self.keys: list[KeyEvent] = []
        self.pointers: list[PointerEvent] = []
        self.closed = False

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        width, height = canvas_size
        return (0, 0, max(1, width - self._reserved_width), height)

    def draw(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        frame: DisplayFrame,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, camera_area
        self.drawn.append(frame)

    def draw_placeholder(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        camera_area: Rect,
    ) -> None:
        del canvas, draw, camera_area
        self.placeholders += 1

    def prepare(self, frame: DisplayFrame) -> None:
        self.prepared.append(frame)

    def on_key(self, event: KeyEvent) -> bool:
        self.keys.append(event)
        return self._consume

    def on_pointer(self, event: PointerEvent) -> bool:
        self.pointers.append(event)
        return self._consume

    def close(self) -> None:
        self.closed = True


class _RecordingSink:
    def __init__(self) -> None:
        self.keys: list[KeyEvent] = []
        self.pointers: list[PointerEvent] = []

    def key_event(self, event: KeyEvent) -> None:
        self.keys.append(event)

    def pointer_event(self, event: PointerEvent) -> None:
        self.pointers.append(event)


class _LazyCudaFrame:
    """Stand-in for a producer handle that keeps its pixels off the host."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def to_cuda_tensor(self) -> np.ndarray:
        return self._array

    def to_numpy(self) -> np.ndarray:
        return self._array


## Protocol conformance


def test_recording_overlay_satisfies_the_overlay_protocol() -> None:
    assert isinstance(_RecordingOverlay(), HudOverlay)


def test_recording_sink_satisfies_the_input_sink_protocol() -> None:
    assert isinstance(_RecordingSink(), InputSink)


## DisplayFrame


def test_display_frame_defaults_to_an_empty_presentable_frame() -> None:
    frame = DisplayFrame()

    assert frame.image is None
    assert frame.status_message is None
    assert dict(frame.overlay_data) == {}


def test_display_frame_supports_replacing_only_the_status_message() -> None:
    """Loops overlay transient text without disturbing the rest of the frame."""
    frame = DisplayFrame(image=object(), timestamp_us=17, overlay_data={"bev": 1})

    overlaid = dataclasses.replace(frame, status_message="Respawning...")

    assert overlaid.status_message == "Respawning..."
    assert overlaid.image is frame.image
    assert overlaid.timestamp_us == 17
    assert overlaid.overlay_data["bev"] == 1


## Fit geometry


def test_fit_rect_centres_a_smaller_source_without_upscaling() -> None:
    assert fit_rect(source_size=(100, 50), area=(0, 0, 200, 200)) == (50, 75, 150, 125)


def test_fit_rect_downscales_a_larger_source_preserving_aspect() -> None:
    assert fit_rect(source_size=(400, 200), area=(0, 0, 200, 200)) == (0, 50, 200, 150)


def test_fit_rect_honours_a_non_zero_area_origin() -> None:
    assert fit_rect(source_size=(100, 100), area=(100, 0, 300, 200)) == (
        150,
        50,
        250,
        150,
    )


@pytest.mark.parametrize(
    ("source_size", "area"),
    [
        ((0, 100), (0, 0, 200, 200)),
        ((100, 0), (0, 0, 200, 200)),
        ((100, 100), (0, 0, 0, 200)),
        ((100, 100), (0, 0, 200, 0)),
    ],
)
def test_fit_rect_returns_none_for_degenerate_inputs(
    source_size: tuple[int, int], area: Rect
) -> None:
    assert fit_rect(source_size=source_size, area=area) is None


## Canvas


def test_allocate_canvas_aliases_the_upload_buffer() -> None:
    """PIL must draw into the same memory the GPU upload reads."""
    buffer, canvas = allocate_canvas(8, 4, background=BLACK)

    ImageDraw.Draw(canvas).rectangle((0, 0, 7, 3), fill=(10, 20, 30, 255))

    assert buffer.shape == (4, 8, 4)
    assert np.array_equal(buffer[0, 0], np.array([10, 20, 30, 255], dtype=np.uint8))


def test_allocate_canvas_starts_opaque_at_the_background_colour() -> None:
    buffer, _canvas = allocate_canvas(4, 2, background=(1, 2, 3))

    assert np.array_equal(buffer[..., :3], np.broadcast_to((1, 2, 3), (2, 4, 3)))
    assert np.all(buffer[..., 3] == 255)


def test_truncate_text_to_width_leaves_a_fitting_label_untouched() -> None:
    font = resolve_font(14)

    assert truncate_text_to_width(font, "ok", 10_000) == "ok"


def test_truncate_text_to_width_appends_an_ellipsis_when_clipping() -> None:
    font = resolve_font(14)

    truncated = truncate_text_to_width(font, "a very long scene label", 40)

    assert truncated.endswith("\u2026")
    assert len(truncated) < len("a very long scene label")


## LRU cache


def test_lru_cache_computes_once_per_key() -> None:
    cache: LRUCache = LRUCache(maxsize=4)
    calls = 0

    def build() -> str:
        nonlocal calls
        calls += 1
        return "value"

    assert cache.get_or_compute("k", build) == "value"
    assert cache.get_or_compute("k", build) == "value"
    assert calls == 1


def test_lru_cache_evicts_the_least_recently_used_entry() -> None:
    cache: LRUCache = LRUCache(maxsize=2)
    cache.get_or_compute("a", lambda: 1)
    cache.get_or_compute("b", lambda: 2)

    # Touch "a" so "b" becomes the eviction candidate.
    cache.get_or_compute("a", lambda: 1)
    cache.get_or_compute("c", lambda: 3)

    assert set(cache) == {"a", "c"}


## Lazy frame helpers


def test_has_cuda_tensor_detects_a_lazy_producer_handle() -> None:
    assert has_cuda_tensor(_LazyCudaFrame(np.zeros((2, 3, 3), dtype=np.uint8)))
    assert not has_cuda_tensor(np.zeros((2, 3, 3), dtype=np.uint8))


def test_rgb_source_size_reports_width_and_height() -> None:
    frame = _LazyCudaFrame(np.zeros((4, 6, 3), dtype=np.uint8))

    assert rgb_source_size(frame) == (6, 4)


@pytest.mark.parametrize("shape", [(4, 6), (2, 4, 6, 3)])
def test_rgb_source_size_rejects_non_hwc_sources(shape: tuple[int, ...]) -> None:
    assert rgb_source_size(np.zeros(shape, dtype=np.uint8)) is None


def test_as_rgb_host_uint8_drops_alpha_and_returns_contiguous_rgb() -> None:
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255

    rgb = as_rgb_host_uint8(rgba)

    assert rgb.shape == (2, 3, 3)
    assert rgb.flags["C_CONTIGUOUS"]
