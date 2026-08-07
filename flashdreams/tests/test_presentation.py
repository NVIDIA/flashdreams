# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the model-agnostic presentation contracts, canvas, and geometry."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from flashdreams.serving.presentation import (
    CompositeOverlay,
    DisplayFrame,
    FrameCompositor,
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
from flashdreams.serving.presentation.canvas import resolve_font
from PIL import Image, ImageDraw

pytestmark = pytest.mark.ci_cpu

BLACK = (0, 0, 0)


class _RecordingOverlay:
    """Minimal overlay that records the calls a presenter would make."""

    def __init__(
        self,
        *,
        reserved_width: int = 0,
        consume: bool = False,
        name: str = "overlay",
        draw_log: list[str] | None = None,
        key_log: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._reserved_width = reserved_width
        self._consume = consume
        self._name = name
        self._draw_log = draw_log
        self._key_log = key_log
        self._close_error = close_error
        self.drawn: list[DisplayFrame] = []
        self.placeholders = 0
        self.prepared: list[DisplayFrame] = []
        self.resizes: list[tuple[int, int]] = []
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
        if self._draw_log is not None:
            self._draw_log.append(self._name)

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

    def on_canvas_resized(self, canvas_size: tuple[int, int]) -> None:
        self.resizes.append(canvas_size)

    def on_key(self, event: KeyEvent) -> bool:
        self.keys.append(event)
        if self._key_log is not None:
            self._key_log.append(self._name)
        return self._consume

    def on_pointer(self, event: PointerEvent) -> bool:
        self.pointers.append(event)
        return self._consume

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _FixedAreaOverlay(_RecordingOverlay):
    """Claims one specific rectangle, whatever the canvas size."""

    def __init__(self, area: Rect) -> None:
        super().__init__()
        self._area = area

    def camera_area(self, canvas_size: tuple[int, int]) -> Rect:
        del canvas_size
        return self._area


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


## Overlay composition


def test_composite_intersects_each_layer_camera_area() -> None:
    """A layer reserving panel space shrinks the camera without coordination."""
    composite = CompositeOverlay(
        layers=(_RecordingOverlay(), _RecordingOverlay(reserved_width=300))
    )

    assert composite.camera_area((1000, 500)) == (0, 0, 700, 500)


def test_composite_intersects_layers_reserving_from_the_same_side() -> None:
    """Two panels on the same edge yield the narrower camera, not the sum."""
    composite = CompositeOverlay(
        layers=(
            _RecordingOverlay(reserved_width=600),
            _RecordingOverlay(reserved_width=400),
        )
    )

    assert composite.camera_area((1000, 500)) == (0, 0, 400, 500)


def test_composite_falls_back_to_the_full_canvas_when_layers_collapse_it() -> None:
    """Layers claiming opposite halves leave no camera, so the claim is ignored."""
    composite = CompositeOverlay(
        layers=(
            _FixedAreaOverlay((0, 0, 400, 500)),
            _FixedAreaOverlay((600, 0, 1000, 500)),
        )
    )

    assert composite.camera_area((1000, 500)) == (0, 0, 1000, 500)


def test_composite_draws_layers_back_to_front() -> None:
    order: list[str] = []
    first = _RecordingOverlay(draw_log=order, name="first")
    second = _RecordingOverlay(draw_log=order, name="second")

    CompositeOverlay(layers=(first, second)).draw(
        Image.new("RGBA", (8, 8)),
        ImageDraw.Draw(Image.new("RGBA", (8, 8))),
        frame=DisplayFrame(),
        camera_area=(0, 0, 8, 8),
    )

    assert order == ["first", "second"]


def test_composite_offers_input_front_to_back_and_stops_at_the_consumer() -> None:
    """The layer drawn on top gets first refusal on a click."""
    order: list[str] = []
    back = _RecordingOverlay(key_log=order, name="back")
    front = _RecordingOverlay(key_log=order, name="front", consume=True)

    handled = CompositeOverlay(layers=(back, front)).on_key(
        KeyEvent(key="w", action="press", timestamp_s=0.0)
    )

    assert handled
    assert order == ["front"]


def test_composite_forwards_resize_and_prepare_to_every_layer() -> None:
    layers = (_RecordingOverlay(), _RecordingOverlay())
    composite = CompositeOverlay(layers=layers)

    composite.on_canvas_resized((640, 480))
    composite.prepare(DisplayFrame())

    assert all(layer.resizes == [(640, 480)] for layer in layers)
    assert all(len(layer.prepared) == 1 for layer in layers)


def test_composite_closes_every_layer_even_when_one_raises() -> None:
    failing = _RecordingOverlay(close_error=RuntimeError("layer down"))
    healthy = _RecordingOverlay()

    with pytest.raises(RuntimeError, match="layer down"):
        CompositeOverlay(layers=(failing, healthy)).close()

    assert healthy.closed


## Frame compositor


def _compositor(overlay: _RecordingOverlay) -> FrameCompositor:
    return FrameCompositor(
        overlay=overlay,
        background=BLACK,
        text_color=(255, 255, 255),
        size=(64, 32),
    )


def test_compositor_buffer_aliases_the_canvas() -> None:
    """Transports upload the buffer, so chrome must land in it directly."""
    compositor = _compositor(_RecordingOverlay())

    ImageDraw.Draw(compositor.canvas).rectangle((0, 0, 63, 31), fill=(9, 8, 7, 255))

    assert np.array_equal(
        compositor.canvas_buffer[0, 0], np.array([9, 8, 7, 255], dtype=np.uint8)
    )


def test_compositor_draws_the_placeholder_before_any_camera_frame() -> None:
    overlay = _RecordingOverlay()
    compositor = _compositor(overlay)

    compositor.render(DisplayFrame(), camera_mode="composite")

    assert overlay.placeholders == 1
    assert len(overlay.drawn) == 1


def test_compositor_composites_the_camera_and_reports_the_area() -> None:
    overlay = _RecordingOverlay(reserved_width=24)
    compositor = _compositor(overlay)
    compositor.set_camera(np.full((32, 40, 3), 200, dtype=np.uint8))

    area = compositor.render(DisplayFrame(image=object()), camera_mode="composite")

    assert area == (0, 0, 40, 32)
    assert overlay.placeholders == 0
    assert compositor.camera_source_size == (40, 32)


def test_compositor_leaves_a_hole_in_transparent_mode() -> None:
    """The GPU path alpha-blends chrome over the image, so the region must clear."""
    compositor = _compositor(_RecordingOverlay())
    compositor.set_camera(np.full((32, 64, 3), 200, dtype=np.uint8))

    compositor.render(DisplayFrame(image=object()), camera_mode="transparent")

    assert compositor.canvas_buffer[0, 0, 3] == 0


def test_compositor_bumps_the_generation_for_each_new_frame() -> None:
    """Producers reuse buffers, so identity cannot signal a new frame."""
    compositor = _compositor(_RecordingOverlay())
    scratch = np.zeros((32, 64, 3), dtype=np.uint8)

    compositor.set_camera(scratch)
    first = compositor.camera_generation
    scratch[:] = 255
    compositor.set_camera(scratch)

    assert compositor.camera_generation != first


def test_compositor_reset_camera_restores_the_placeholder() -> None:
    overlay = _RecordingOverlay()
    compositor = _compositor(overlay)
    compositor.set_camera(np.full((32, 64, 3), 200, dtype=np.uint8))

    compositor.reset_camera()
    compositor.render(DisplayFrame(), camera_mode="composite")

    assert compositor.camera_source_size is None
    assert overlay.placeholders == 1


def test_compositor_resize_reallocates_and_notifies_the_overlay() -> None:
    overlay = _RecordingOverlay()
    compositor = _compositor(overlay)

    compositor.resize(20, 10)

    assert compositor.size == (20, 10)
    assert compositor.canvas_buffer.shape == (10, 20, 4)
    assert overlay.resizes == [(20, 10)]


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
    assert frame.allow_window_resize
    assert dict(frame.overlay_data) == {}


def test_display_frame_can_opt_out_of_source_driven_window_resize() -> None:
    """Images already rendered at window resolution must not resize the window."""
    frame = DisplayFrame(image=object(), allow_window_resize=False)

    assert not frame.allow_window_resize


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
