# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``LocalWindowPresenter`` driven against a fake window, device, and swapchain.

Covers the parts that only misbehave at runtime and are otherwise validated by
launching the demo and looking at it: deferred resize, swapchain recovery,
source-driven window growth, and camera re-upload on change.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from fake_slangpy import (
    FakeKeyboardEvent,
    FakeMouseEvent,
    FakeSlangpy,
    FakeSurfaceInfo,
    Format,
    KeyCode,
    MouseButton,
    MouseEventType,
)
from flashdreams.serving.presentation import (
    DisplayFrame,
    KeyEvent,
    PointerEvent,
    Rect,
    WindowConfig,
)
from flashdreams.serving.presentation.local_window import LocalWindowPresenter
from PIL import Image, ImageDraw

pytestmark = pytest.mark.ci_cpu


class _StubOverlay:
    """Records what the presenter asked of it, and can reserve a panel."""

    def __init__(self, *, reserved_width: int = 0, consume_input: bool = False) -> None:
        self._reserved_width = reserved_width
        self._consume_input = consume_input
        self.draws = 0
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
        del canvas, draw, frame, camera_area
        self.draws += 1

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
        return self._consume_input

    def on_pointer(self, event: PointerEvent) -> bool:
        self.pointers.append(event)
        return self._consume_input

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeSlangpy) -> FakeSlangpy:
    """Stand in for ``slangpy`` when the presenter imports it at construction."""
    monkeypatch.setitem(sys.modules, "slangpy", fake)
    return fake


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> FakeSlangpy:
    return _install(monkeypatch, FakeSlangpy())


def _presenter(
    spy: FakeSlangpy,
    overlay: _StubOverlay | None = None,
    **config: Any,
) -> tuple[LocalWindowPresenter, _StubOverlay]:
    resolved = overlay if overlay is not None else _StubOverlay()
    settings: dict[str, Any] = {"width": 800, "height": 600} | config
    presenter = LocalWindowPresenter(
        overlay=resolved,
        config=WindowConfig(**settings),
        cuda_interop_disabled=True,
    )
    del spy
    return presenter, resolved


def _camera(width: int = 64, height: int = 32, *, fill: int = 128) -> np.ndarray:
    return np.full((height, width, 3), fill, dtype=np.uint8)


## Construction


def test_construction_configures_the_surface_at_the_live_window_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window manager may clamp or scale the requested size, and configuring
    the surface at the wrong one makes the first acquire fail with a generic
    error."""
    spy = _install(monkeypatch, FakeSlangpy(clamp_window_to=(1280, 720)))

    presenter, _ = _presenter(spy, width=3840, height=2160)

    assert presenter.window_size == (1280, 720)
    assert spy.surface.configure_calls[0] == (1280, 720)
    presenter.close()


def test_construction_requires_a_linear_surface_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched gamma between the display texture and an sRGB swapchain
    silently washes the colours out, so refuse rather than present wrong."""
    spy = _install(
        monkeypatch,
        FakeSlangpy(
            surface_info=FakeSurfaceInfo(
                preferred_format=Format.bgrx8_unorm_srgb,
                formats=(Format.rgba8_unorm_srgb,),
            )
        ),
    )

    with pytest.raises(RuntimeError, match="linear swapchain"):
        _presenter(spy)


def test_missing_slangpy_names_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ``None`` entry in ``sys.modules`` makes ``import`` raise ImportError.
    monkeypatch.setitem(sys.modules, "slangpy", None)

    with pytest.raises(RuntimeError, match="local-window"):
        LocalWindowPresenter(overlay=_StubOverlay())


## Presenting


def test_presenting_a_frame_uploads_and_presents_once(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)

    presenter.present_frame(DisplayFrame(image=_camera()))

    assert overlay.draws == 1
    assert spy.device.submits == 1
    presenter.close()


def test_a_frameless_tick_draws_the_overlay_placeholder(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)

    presenter.present_frame(DisplayFrame())

    # Chrome still renders; only the camera region falls back to a placeholder.
    assert overlay.placeholders == 1
    assert overlay.draws == 1
    presenter.close()


def test_a_status_message_composites_on_the_cpu(spy: FakeSlangpy) -> None:
    """Status text has to land over the camera, which the GPU stamp-in path
    cannot express, so that tick falls back to a full CPU composite."""
    presenter, _ = _presenter(spy)

    presenter.present_frame(
        DisplayFrame(image=_camera(), status_message="Loading world model...")
    )

    assert "copy_texture" not in spy.device.encoder_log
    presenter.close()


def test_present_status_paints_without_a_frame(spy: FakeSlangpy) -> None:
    presenter, _ = _presenter(spy)

    presenter.present_status("Optimizing...")

    assert spy.device.submits == 1
    assert spy.window.process_event_calls == 1
    presenter.close()


def test_prepare_frame_forwards_to_the_overlay(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)
    frame = DisplayFrame(image=_camera())

    presenter.prepare_frame(frame)

    assert overlay.prepared == [frame]
    presenter.close()


## Resize


def test_a_resize_callback_is_applied_on_the_next_present(spy: FakeSlangpy) -> None:
    """Vulkan resources must only be rebuilt on the presentation thread, so the
    windowing-thread callback may only record the request."""
    presenter, overlay = _presenter(spy)

    spy.window.notify_resize(1024, 768)
    assert presenter.window_size == (800, 600)

    presenter.present_frame(DisplayFrame(image=_camera()))

    assert presenter.window_size == (1024, 768)
    assert overlay.resizes[-1] == (1024, 768)
    assert spy.surface.configure_calls[-1] == (1024, 768)
    presenter.close()


def test_a_compositor_side_resize_is_noticed_without_a_callback(
    spy: FakeSlangpy,
) -> None:
    """SDL3 misses resize callbacks for window-manager fitting and HiDPI
    scaling, so the live size is compared every tick."""
    presenter, _ = _presenter(spy)

    spy.window.set_size(640, 480)
    presenter.present_frame(DisplayFrame(image=_camera()))

    assert presenter.window_size == (640, 480)
    presenter.close()


def test_a_persistently_failing_resize_keeps_presenting_at_the_previous_size(
    spy: FakeSlangpy,
) -> None:
    """Losing the ability to allocate must degrade to a stale-size window
    rather than take down the run."""
    presenter, _ = _presenter(spy)
    spy.device.texture_error = RuntimeError("out of memory")

    spy.window.notify_resize(1024, 768)
    presenter.present_frame(DisplayFrame(image=_camera()))
    presenter.present_frame(DisplayFrame(image=_camera()))

    assert presenter.window_size == (800, 600)
    presenter.close()


def test_a_transient_resize_failure_recovers_on_the_next_tick(
    spy: FakeSlangpy,
) -> None:
    """The live window size is re-read every tick, so a one-off allocation
    failure is retried rather than leaving the window permanently mismatched."""
    presenter, _ = _presenter(spy)
    spy.device.texture_error = RuntimeError("transient")

    spy.window.notify_resize(1024, 768)
    presenter.present_frame(DisplayFrame(image=_camera()))
    assert presenter.window_size == (800, 600)

    spy.device.texture_error = None
    presenter.present_frame(DisplayFrame(image=_camera()))

    assert presenter.window_size == (1024, 768)
    presenter.close()


def test_a_degenerate_resize_is_clamped_to_one_pixel(spy: FakeSlangpy) -> None:
    """Minimising a window can report a zero dimension."""
    presenter, _ = _presenter(spy)

    spy.window.notify_resize(0, 0)
    presenter.present_frame(DisplayFrame(image=_camera()))

    assert presenter.window_size == (1, 1)
    presenter.close()


## Source-driven window growth


def test_a_larger_source_grows_the_window_past_the_reserved_panel(
    spy: FakeSlangpy,
) -> None:
    overlay = _StubOverlay(reserved_width=500)
    presenter, _ = _presenter(spy, overlay, width=800, height=600)

    presenter.present_frame(DisplayFrame(image=_camera(1280, 704)))

    assert spy.window.resize_requests == [(1780, 704)]
    presenter.close()


def test_the_growth_frame_is_dropped_rather_than_presented_at_the_old_size(
    spy: FakeSlangpy,
) -> None:
    presenter, _ = _presenter(spy, width=800, height=600)

    presenter.present_frame(DisplayFrame(image=_camera(1280, 704)))

    assert spy.device.submits == 0
    presenter.close()


def test_a_smaller_source_leaves_the_window_alone(spy: FakeSlangpy) -> None:
    """Small frames stay centred at native resolution instead of upscaling."""
    presenter, _ = _presenter(spy, width=800, height=600)

    presenter.present_frame(DisplayFrame(image=_camera(320, 240)))

    assert spy.window.resize_requests == []
    presenter.close()


def test_the_window_grows_once_per_source_resolution(spy: FakeSlangpy) -> None:
    presenter, _ = _presenter(spy, width=800, height=600)

    for _ in range(3):
        presenter.present_frame(DisplayFrame(image=_camera(1280, 704)))

    assert len(spy.window.resize_requests) == 1
    presenter.close()


def test_shrinking_the_window_is_not_undone_by_the_same_source(
    spy: FakeSlangpy,
) -> None:
    """Once a resolution has been fitted the user owns the window. Re-growing
    on every frame would make the window impossible to shrink."""
    presenter, _ = _presenter(spy, width=800, height=600)
    frame = DisplayFrame(image=_camera(1280, 704))
    presenter.present_frame(frame)
    grew = len(spy.window.resize_requests)

    spy.window.notify_resize(640, 480)
    presenter.present_frame(frame)
    presenter.present_frame(frame)

    assert len(spy.window.resize_requests) == grew
    presenter.close()


def test_a_frame_can_opt_out_of_driving_the_window_size(spy: FakeSlangpy) -> None:
    presenter, _ = _presenter(spy, width=800, height=600)

    presenter.present_frame(
        DisplayFrame(image=_camera(1280, 704), allow_window_resize=False)
    )

    assert spy.window.resize_requests == []
    presenter.close()


def test_auto_resize_can_be_disabled_by_config(spy: FakeSlangpy) -> None:
    presenter, _ = _presenter(spy, width=800, height=600, auto_resize_to_source=False)

    presenter.present_frame(DisplayFrame(image=_camera(1280, 704)))

    assert spy.window.resize_requests == []
    presenter.close()


## Swapchain recovery


def test_a_stale_swapchain_is_reconfigured_and_the_frame_skipped(
    spy: FakeSlangpy,
) -> None:
    """NVIDIA's driver reports VK_ERROR_OUT_OF_DATE_KHR as a generic failure;
    reconfiguring at the live size recovers and the next tick retries."""
    presenter, _ = _presenter(spy)
    frame = DisplayFrame(image=_camera())
    presenter.present_frame(frame)
    submits = spy.device.submits
    configures = len(spy.surface.configure_calls)

    spy.surface.acquire_error = RuntimeError("SLANG_FAIL")
    presenter.present_frame(frame)

    # Skipping alone is not recovery: without reconfiguring at the live size
    # every later acquire would fail the same way.
    assert spy.device.submits == submits
    assert len(spy.surface.configure_calls) > configures

    presenter.present_frame(frame)
    assert spy.device.submits == submits + 1
    presenter.close()


def test_a_failed_present_reconfigures_the_surface(spy: FakeSlangpy) -> None:
    presenter, _ = _presenter(spy)
    surface = spy.surface
    before = len(surface.configure_calls)

    surface.present_error = RuntimeError("SLANG_FAIL")
    presenter.present_frame(DisplayFrame(image=_camera()))

    assert len(surface.configure_calls) > before
    presenter.close()


def test_an_unavailable_swapchain_image_is_skipped_without_error(
    spy: FakeSlangpy,
) -> None:
    presenter, _ = _presenter(spy)
    spy.surface.acquire_returns_none = True

    presenter.present_frame(DisplayFrame(image=_camera()))

    assert spy.device.submits == 0
    presenter.close()


## Camera upload


def test_a_changed_camera_frame_reaches_the_gpu(spy: FakeSlangpy) -> None:
    """A stale-generation check here once froze the video on its first frame.

    The upload itself is unconditional, so the bytes are what matter: counting
    calls would pass even if every tick re-sent the first frame.
    """
    presenter, _ = _presenter(spy)

    presenter.present_frame(DisplayFrame(image=_camera(fill=10)))
    presenter.present_frame(DisplayFrame(image=_camera(fill=200)))

    uploaded = _camera_uploads(spy)
    assert len(uploaded) >= 2
    assert uploaded[0][..., 0].max() == 10
    assert uploaded[-1][..., 0].max() == 200
    presenter.close()


def test_an_unchanged_camera_frame_is_not_re_staged(spy: FakeSlangpy) -> None:
    """Re-staging costs a full host memcpy of the frame every tick."""
    presenter, _ = _presenter(spy)
    frame = DisplayFrame(image=_camera(fill=42))

    presenter.present_frame(frame)
    presenter.present_frame(frame)

    uploaded = _camera_uploads(spy)
    assert all(upload[..., 0].max() == 42 for upload in uploaded)
    presenter.close()


def test_reset_camera_drops_the_retained_frame(spy: FakeSlangpy) -> None:
    """A new scene must not ghost the previous rollout's last frame."""
    presenter, overlay = _presenter(spy)
    presenter.present_frame(DisplayFrame(image=_camera()))

    presenter.reset_camera()
    presenter.present_frame(DisplayFrame())

    assert overlay.placeholders == 1
    presenter.close()


## Input


def test_key_events_reach_the_overlay_normalized(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)

    spy.window.on_keyboard_event(FakeKeyboardEvent(key=KeyCode.w, action="press"))
    spy.window.on_keyboard_event(FakeKeyboardEvent(key=KeyCode.w, action="release"))

    assert [(event.key, event.action) for event in overlay.keys] == [
        ("w", "press"),
        ("w", "release"),
    ]
    presenter.close()


def test_escape_closes_the_window_without_reaching_the_overlay(
    spy: FakeSlangpy,
) -> None:
    presenter, overlay = _presenter(spy)

    spy.window.on_keyboard_event(FakeKeyboardEvent(key=KeyCode.escape, action="press"))

    assert presenter.should_close
    assert overlay.keys == []
    presenter.close()


def test_an_unmapped_key_is_dropped(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)

    spy.window.on_keyboard_event(FakeKeyboardEvent(key="f24", action="press"))

    assert overlay.keys == []
    presenter.close()


def test_pointer_events_reach_the_overlay_with_integer_positions(
    spy: FakeSlangpy,
) -> None:
    presenter, overlay = _presenter(spy)

    spy.window.on_mouse_event(
        FakeMouseEvent(
            type=MouseEventType.button_down,
            pos=(12.7, 34.2),
            button=MouseButton.left,
        )
    )

    assert overlay.pointers[0].position == (12, 34)
    assert overlay.pointers[0].button == "left"
    assert overlay.pointers[0].action == "press"
    presenter.close()


def test_a_scroll_event_is_ignored(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)

    spy.window.on_mouse_event(FakeMouseEvent(type=MouseEventType.scroll, pos=(0, 0)))

    assert overlay.pointers == []
    presenter.close()


## Lifecycle


def test_should_close_follows_the_window(spy: FakeSlangpy) -> None:
    presenter, _ = _presenter(spy)
    assert not presenter.should_close

    spy.window.request_close()

    assert presenter.should_close
    presenter.close()


def test_close_releases_the_overlay_and_the_window(spy: FakeSlangpy) -> None:
    presenter, overlay = _presenter(spy)

    presenter.close()

    assert overlay.closed
    assert spy.window.closed


def test_close_still_shuts_the_window_when_the_overlay_raises(
    spy: FakeSlangpy,
) -> None:
    """A leaky overlay must not leave a window on screen with no way to quit."""

    class _FailingOverlay(_StubOverlay):
        def close(self) -> None:
            raise RuntimeError("overlay close failed")

    presenter, _ = _presenter(spy, _FailingOverlay())

    presenter.close()

    assert spy.window.closed


def _camera_uploads(spy: FakeSlangpy) -> list[Any]:
    """Pixels uploaded to the source-sized camera texture, oldest first."""
    return [
        upload
        for texture in spy.device.textures
        if texture.label == "local_window_camera_src"
        for upload in texture.uploads
    ]
