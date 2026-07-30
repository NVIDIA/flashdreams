# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import concurrent.futures
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omnidreams.interactive_drive.presenter import (
    SlangPyPresenter,
    _CudaRGBFrame,
    _CudaRGBInterop,
    _NonBlockingCudaStream,
)
from omnidreams.interactive_drive.slangpy_hud_presenter import SlangPyHudPresenter
from omnidreams.interactive_drive.types import PresentedFrame
from PIL import Image


class _LazyFrame:
    def __init__(self) -> None:
        self.numpy_calls = 0
        self.prefetch_calls = 0

    def prefetch_to_numpy(self) -> None:
        self.prefetch_calls += 1

    def to_numpy(self) -> np.ndarray:
        self.numpy_calls += 1
        return np.full((4, 4, 3), 127, dtype=np.uint8)


def _presenter_without_window() -> SlangPyPresenter:
    return SlangPyPresenter.__new__(SlangPyPresenter)


def _hud_presenter_without_window() -> SlangPyHudPresenter:
    return SlangPyHudPresenter.__new__(SlangPyHudPresenter)


def test_cuda_existing_device_handles_uses_current_context_by_default(
    monkeypatch,
) -> None:
    presenter = _presenter_without_window()
    handles = [object()]

    class _Spy:
        @staticmethod
        def get_cuda_current_context_native_handles() -> list[object]:
            return handles

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            current_stream=lambda: object(),
        )
    )
    monkeypatch.delenv("INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    assert presenter._cuda_existing_device_handles() == handles


@pytest.mark.parametrize(
    "presenter_factory",
    [_presenter_without_window, _hud_presenter_without_window],
)
def test_cuda_existing_device_handles_initializes_lazy_cuda_context(
    monkeypatch, presenter_factory
) -> None:
    presenter = presenter_factory()
    handles = [object()]
    initialized = False
    calls: list[str] = []

    class _Spy:
        @staticmethod
        def get_cuda_current_context_native_handles() -> list[object]:
            assert initialized
            assert calls == ["init", "current_stream"]
            return handles

    def init() -> None:
        nonlocal initialized
        calls.append("init")
        initialized = True

    def current_stream() -> object:
        calls.append("current_stream")
        return object()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: initialized,
            init=init,
            current_stream=current_stream,
        )
    )
    monkeypatch.delenv("INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    assert presenter._cuda_existing_device_handles() == handles
    assert calls == ["init", "current_stream"]


def test_cuda_existing_device_handles_can_be_disabled(monkeypatch) -> None:
    presenter = _presenter_without_window()

    class _Spy:
        @staticmethod
        def get_cuda_current_context_native_handles() -> list[object]:
            raise AssertionError("native handle query should be disabled")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            current_stream=lambda: object(),
        )
    )
    monkeypatch.setenv("INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP", "1")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    assert presenter._cuda_existing_device_handles() == []


def test_create_device_enables_cuda_interop_with_current_context_by_default(
    monkeypatch,
) -> None:
    presenter = _presenter_without_window()
    created_kwargs: list[dict[str, object]] = []

    class _DeviceType:
        vulkan = object()

    class _Spy:
        DeviceType = _DeviceType

        @staticmethod
        def get_cuda_current_context_native_handles() -> list[object]:
            return ["cuda-context"]

        @staticmethod
        def Device(**kwargs):
            created_kwargs.append(kwargs)
            return SimpleNamespace(info=SimpleNamespace(adapter_name="fake"))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            current_stream=lambda: object(),
        )
    )
    monkeypatch.delenv("INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    presenter._create_device()

    assert created_kwargs[0]["enable_cuda_interop"] is True
    assert created_kwargs[0]["existing_device_handles"] == ["cuda-context"]


@pytest.mark.parametrize(
    "presenter_factory",
    [_presenter_without_window, _hud_presenter_without_window],
)
def test_create_device_disables_cuda_interop_without_a_cuda_context(
    monkeypatch, presenter_factory
) -> None:
    presenter = presenter_factory()
    presenter._cuda_interop_unavailable_reason = None
    created_kwargs: list[dict[str, object]] = []

    class _DeviceType:
        vulkan = object()

    class _Spy:
        DeviceType = _DeviceType

        @staticmethod
        def Device(**kwargs):
            created_kwargs.append(kwargs)
            return SimpleNamespace(info=SimpleNamespace(adapter_name="fake"))

    def fail_cuda_init() -> None:
        raise RuntimeError("CUDA unavailable")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: False,
            init=fail_cuda_init,
        )
    )
    monkeypatch.delenv("INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    presenter._create_device()

    assert created_kwargs[0]["enable_cuda_interop"] is False
    assert "existing_device_handles" not in created_kwargs[0]
    assert presenter._cuda_interop_unavailable_reason == "CUDA context unavailable"


def test_create_device_disables_cuda_interop_when_cuda_interop_is_disabled(
    monkeypatch,
) -> None:
    presenter = _presenter_without_window()
    created_kwargs: list[dict[str, object]] = []

    class _DeviceType:
        vulkan = object()

    class _Spy:
        DeviceType = _DeviceType

        @staticmethod
        def Device(**kwargs):
            created_kwargs.append(kwargs)
            return SimpleNamespace(info=SimpleNamespace(adapter_name="fake"))

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setenv("INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP", "1")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    presenter._create_device()

    assert created_kwargs[0]["enable_cuda_interop"] is False
    assert "existing_device_handles" not in created_kwargs[0]
    assert "INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP" in (
        presenter._cuda_interop_unavailable_reason or ""
    )


def test_non_blocking_cuda_stream_uses_pytorch_native_stream_handle() -> None:
    device = SimpleNamespace(index=0)
    synchronize_calls = 0

    class _Stream:
        cuda_stream = 12345

        def synchronize(self) -> None:
            nonlocal synchronize_calls
            synchronize_calls += 1

    pytorch_stream = _Stream()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=lambda *, device: pytorch_stream,
        )
    )

    stream = _NonBlockingCudaStream(fake_torch, device)

    assert stream.stream is pytorch_stream
    assert stream.cuda_stream == 12345

    stream.close()
    stream.close()

    assert synchronize_calls == 1
    assert stream.stream is None
    assert stream.cuda_stream == 0


def test_prepare_frame_prefetches_host_fallback_model_rgb() -> None:
    presenter = _presenter_without_window()
    lazy = _LazyFrame()
    presenter._cuda_rgb_interop = None

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.prepare_frame(frame, view_mode="model_rgb")

    assert lazy.prefetch_calls == 1
    assert lazy.numpy_calls == 0


def test_model_rgb_uses_cuda_path_without_materializing_host_frame() -> None:
    presenter = _presenter_without_window()
    lazy = _LazyFrame()
    cuda_calls: list[tuple[object, str | None]] = []

    def present_cuda_rgb(rgb_frame: object, *, status_message: str | None) -> bool:
        cuda_calls.append((rgb_frame, status_message))
        return True

    def present_array(rgb_host_uint8: np.ndarray) -> None:
        del rgb_host_uint8
        raise AssertionError("host presenter path should not run")

    presenter._present_cuda_rgb = present_cuda_rgb
    presenter._present_array = present_array

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert cuda_calls == [(lazy, None)]
    assert lazy.numpy_calls == 0


def test_model_rgb_falls_back_to_host_when_cuda_path_declines() -> None:
    presenter = _presenter_without_window()
    lazy = _LazyFrame()
    presented: list[np.ndarray] = []

    presenter._present_cuda_rgb = lambda rgb_frame, *, status_message: False

    def present_array(rgb_host_uint8: np.ndarray) -> None:
        presented.append(rgb_host_uint8)

    presenter._present_array = present_array

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert lazy.numpy_calls == 1
    assert len(presented) == 1
    assert np.all(presented[0] == 127)


def test_model_rgb_does_not_materialize_host_frame_when_cuda_source_is_pending() -> (
    None
):
    presenter = _presenter_without_window()
    lazy = _LazyFrame()

    class _PendingInterop:
        def as_cuda_rgb_frame(self, rgb_frame: object) -> _CudaRGBFrame | None:
            assert rgb_frame is lazy
            return _CudaRGBFrame(tensor=object(), source_event=object(), ready=False)

        def ready_rgba_buffer(self) -> None:
            return None

    def present_array(rgb_host_uint8: np.ndarray) -> None:
        del rgb_host_uint8
        raise AssertionError("host presenter path should not run")

    presenter._cuda_rgb_interop = _PendingInterop()
    presenter._present_array = present_array

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
        status_message="pending",
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert lazy.numpy_calls == 0


def test_hud_recreates_cuda_interop_after_resize() -> None:
    presenter = _hud_presenter_without_window()
    old_interop = object()
    new_interop = object()
    created_sizes: list[tuple[int, int]] = []
    presenter._cuda_hud_interop = old_interop
    presenter._retired_cuda_hud_interops = []
    presenter._cuda_hud_resize_logged = True

    def create_interop(width: int, height: int) -> object:
        created_sizes.append((width, height))
        return new_interop

    presenter._create_cuda_hud_interop = create_interop

    presenter._recreate_cuda_hud_interop_after_resize(123, 456)

    assert presenter._retired_cuda_hud_interops == [old_interop]
    assert presenter._cuda_hud_interop is new_interop
    assert created_sizes == [(123, 456)]
    assert presenter._cuda_hud_resize_logged is False


def test_model_rgb_does_not_fallback_to_host_when_interop_buffers_are_busy() -> None:
    presenter = _presenter_without_window()
    lazy = _LazyFrame()

    class _BusyInterop:
        enqueue_calls = 0

        def as_cuda_rgb_frame(self, rgb_frame: object) -> _CudaRGBFrame | None:
            assert rgb_frame is lazy
            return _CudaRGBFrame(tensor=object(), source_event=None, ready=True)

        def ready_rgba_buffer(self) -> None:
            return None

        def enqueue_rgb_to_shared_rgba(self, rgb_frame: _CudaRGBFrame) -> bool:
            assert rgb_frame.ready
            self.enqueue_calls += 1
            return False

    busy_interop = _BusyInterop()

    def present_array(rgb_host_uint8: np.ndarray) -> None:
        del rgb_host_uint8
        raise AssertionError("host presenter path should not run")

    presenter._cuda_rgb_interop = busy_interop
    presenter._present_array = present_array

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert lazy.numpy_calls == 0
    assert busy_interop.enqueue_calls == 1


def test_cuda_hud_alpha_composite_uses_supported_tensor_math() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this regression test")

    interop = _CudaRGBInterop.__new__(_CudaRGBInterop)
    interop._torch = torch
    base = torch.zeros((2, 2, 4), device="cuda", dtype=torch.uint8)
    base[..., :3] = 10
    base[..., 3] = 255
    overlay = torch.zeros((2, 2, 4), device="cuda", dtype=torch.uint8)
    overlay[..., 0] = 110
    overlay[..., 3] = 128

    interop._alpha_composite_rgba(base, overlay)
    torch.cuda.synchronize()

    assert base[..., 3].eq(255).all()
    assert base[..., 0].eq(60).all()
    assert base[..., 1].eq(5).all()
    assert base[..., 2].eq(5).all()


def test_hud_prepare_frame_keeps_cuda_model_rgb_lazy() -> None:
    presenter = _hud_presenter_without_window()
    model = _LazyFrame()
    bev = _LazyFrame()
    presenter._cuda_hud_interop = object()

    model.to_cuda_tensor = lambda: object()  # type: ignore[attr-defined]

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=model,
        bev_host_uint8=bev,
    )

    presenter.prepare_frame(frame, view_mode="model_rgb")

    assert model.prefetch_calls == 0
    assert model.numpy_calls == 0
    assert bev.prefetch_calls == 1


def test_hud_prepare_frame_prefetches_one_bev_per_raster_batch() -> None:
    presenter = _hud_presenter_without_window()
    first = _LazyFrame()
    second = _LazyFrame()
    batch_key = object()
    first.source_group_key = lambda: batch_key  # type: ignore[attr-defined]
    second.source_group_key = lambda: batch_key  # type: ignore[attr-defined]
    presenter._cuda_hud_interop = object()

    for bev in (first, second):
        presenter.prepare_frame(
            PresentedFrame(
                timestamp_us=0,
                rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
                depth_host_f32=None,
                bev_host_uint8=bev,
            ),
            view_mode="rgb",
        )

    assert first.prefetch_calls == 1
    assert second.prefetch_calls == 0


def test_hud_bev_panel_reuses_completed_image_while_refresh_is_in_flight() -> None:
    presenter = _hud_presenter_without_window()
    pending: concurrent.futures.Future[object] = concurrent.futures.Future()
    cached = Image.new("RGB", (4, 3), (1, 2, 3))
    presenter._latest_bev_source = np.full((8, 8, 3), 5, dtype=np.uint8)
    presenter._bev_source_generation = 4
    presenter._bev_panel_epoch = 2
    presenter._bev_panel_future = pending
    presenter._bev_panel_cache_key = (2, 123, 4, 3)
    presenter._bev_panel_cache = cached

    assert presenter._get_bev_panel_image((4, 3)) is cached
    assert not pending.done()


def test_hud_bev_panel_build_runs_outside_draw_path() -> None:
    presenter = _hud_presenter_without_window()
    presenter._latest_bev_source = np.zeros((8, 8, 3), dtype=np.uint8)
    presenter._bev_source_generation = 1
    presenter._bev_panel_epoch = 0
    presenter._bev_panel_future = None
    presenter._bev_panel_cache_key = None
    presenter._bev_panel_cache = None
    presenter._bev_panel_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        assert presenter._get_bev_panel_image((4, 3)) is None
        assert presenter._bev_panel_future is not None
        presenter._bev_panel_future.result(timeout=2.0)

        panel = presenter._get_bev_panel_image((4, 3))

        assert panel is not None
        assert panel.size == (4, 3)
    finally:
        presenter._bev_panel_exec.shutdown(wait=True, cancel_futures=True)


def test_hud_bev_update_keeps_lazy_source_unmaterialized() -> None:
    presenter = _hud_presenter_without_window()
    lazy = _LazyFrame()
    presenter._latest_bev_source = None
    presenter._bev_source_generation = 0

    presenter._update_bev_pil(lazy)

    assert presenter._latest_bev_source is lazy
    assert presenter._bev_source_generation == 1
    assert lazy.numpy_calls == 0


def test_hud_model_rgb_uses_cuda_path_without_materializing_host_frame() -> None:
    presenter = _hud_presenter_without_window()
    lazy = _LazyFrame()
    cuda_calls: list[tuple[PresentedFrame, object]] = []

    def present_cuda_hud_frame(frame: PresentedFrame, rgb: object) -> bool:
        cuda_calls.append((frame, rgb))
        return True

    def update_camera_pil(rgb: object) -> None:
        del rgb
        raise AssertionError("host HUD camera path should not run")

    presenter._pending_resize = None
    presenter._present_cuda_hud_frame = present_cuda_hud_frame
    presenter._update_camera_pil = update_camera_pil

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert cuda_calls == [(frame, lazy)]
    assert lazy.numpy_calls == 0


def test_hud_world_model_loading_pumps_events_and_presents_placeholder() -> None:
    presenter = _hud_presenter_without_window()
    calls: list[tuple[str, object]] = []

    presenter.process_events = lambda: calls.append(("events", None))
    presenter.set_engine_active = lambda active: calls.append(("active", active))
    presenter._render_canvas = lambda status_message: calls.append(
        ("render", status_message)
    )
    presenter._present_canvas = lambda **kwargs: calls.append(("present", kwargs))

    presenter.present_world_model_loading()

    assert calls == [
        ("events", None),
        ("active", True),
        ("render", "Loading World Model"),
        ("present", {"use_gpu_camera": False}),
    ]


def test_hud_resize_updates_presenter_texture_and_recreates_cuda_interop() -> None:
    presenter = _hud_presenter_without_window()
    configured: list[tuple[int, int]] = []
    created: list[tuple[int, int]] = []

    class _Interop:
        def close(self) -> None:
            raise AssertionError("resize should not destroy CUDA interop in place")

    interop = _Interop()
    new_interop = _Interop()
    presenter._configured_size = (1920, 1080)
    presenter._surface_format = object()
    presenter._display_texture = "old-texture"
    presenter._cuda_hud_interop = interop
    presenter._retired_cuda_hud_interops = []
    presenter._cuda_hud_resize_logged = False
    presenter._panel_chrome_cache_key = object()
    presenter._panel_chrome_cache = object()
    presenter._bev_panel_cache_key = object()
    presenter._bev_panel_cache = object()
    presenter._wheel_rotation_cache = {}
    presenter._pedal_cache = {}
    presenter._camera_fit_texture = object()
    presenter._camera_fit_size = (1, 1)
    presenter._configure_surface = lambda width, height: configured.append(
        (width, height)
    )
    presenter._build_display_texture = lambda width, height: (
        "texture",
        width,
        height,
    )
    presenter._create_cuda_hud_interop = lambda width, height: (
        created.append((width, height)) or new_interop
    )

    assert presenter._apply_resize(1000, 700)

    assert configured == [(1000, 700)]
    assert presenter._configured_size == (1000, 700)
    assert presenter._display_texture == ("texture", 1000, 700)
    assert presenter._canvas.size == (1000, 700)
    assert created == [(1000, 700)]
    assert presenter._cuda_hud_interop is new_interop
    assert presenter._retired_cuda_hud_interops == [interop]


def test_hud_postprocess_control_toggles_configured_preset() -> None:
    presenter = _hud_presenter_without_window()
    calls: list[bool] = []
    presenter._postprocess_rect = (10, 20, 110, 52)
    presenter._panel_chrome_cache_key = object()
    presenter._panel_chrome_cache = object()
    presenter._scene_dropdown_open = False
    presenter._variant_dropdown_open = False
    presenter.set_postprocess_control(
        preset="rtx-super-resolution",
        enabled=True,
        callback=calls.append,
    )

    presenter._handle_click((20, 30))
    presenter._handle_click((20, 30))

    assert calls == [False, True]
    assert presenter._postprocess_enabled is True


def test_hud_scene_dropdown_blocks_underlying_upsample_toggle() -> None:
    presenter = _hud_presenter_without_window()
    calls: list[bool] = []
    presenter._postprocess_rect = (10, 20, 110, 52)
    presenter._postprocess_preset = "rtx-super-resolution-ultra"
    presenter._postprocess_enabled = True
    presenter._postprocess_callback = calls.append
    presenter._scene_dropdown_open = True
    presenter._variant_dropdown_open = False
    presenter._scene_item_rects = []
    presenter._scene_header_rect = None
    presenter._scene_selection_locked_probe = lambda: False

    presenter._handle_click((20, 30))

    assert calls == []
    assert presenter._postprocess_enabled is True


def test_hud_resize_uses_actual_window_size_without_model_resolution_clamp() -> None:
    presenter = _hud_presenter_without_window()
    presenter._pending_resize = None

    presenter._on_resize(320, 200)

    assert presenter._pending_resize == (320, 200)


def test_hud_auto_sizes_window_to_native_model_frame_resolution() -> None:
    presenter = _hud_presenter_without_window()
    resize_calls: list[tuple[int, int]] = []
    presenter._auto_sized_camera_src_size = None
    presenter._pending_resize = None
    presenter._window = SimpleNamespace(
        size=SimpleNamespace(x=1000, y=600),
        resize=lambda width, height: resize_calls.append((width, height)),
    )

    resized = presenter._resize_window_for_native_model_frame(
        np.zeros((704, 1280, 3), dtype=np.uint8)
    )

    assert resized is True
    assert resize_calls == [(1780, 704)]
    assert presenter._pending_resize == (1780, 704)


def test_hud_keeps_larger_canvas_when_model_resolution_shrinks() -> None:
    presenter = _hud_presenter_without_window()
    resize_calls: list[tuple[int, int]] = []
    presenter._auto_sized_camera_src_size = (1280, 704)
    presenter._pending_resize = None
    presenter._window = SimpleNamespace(
        size=SimpleNamespace(x=1500, y=800),
        resize=lambda width, height: resize_calls.append((width, height)),
    )

    assert not presenter._resize_window_for_native_model_frame(
        np.zeros((704, 1280, 3), dtype=np.uint8)
    )
    assert not presenter._resize_window_for_native_model_frame(
        np.zeros((352, 640, 3), dtype=np.uint8)
    )

    assert resize_calls == []


def test_hud_centers_smaller_camera_at_native_resolution() -> None:
    presenter = _hud_presenter_without_window()
    presenter._latest_camera_src_size = (640, 352)
    presenter._configured_size = (1920, 1080)

    fit = presenter._compute_camera_fit()

    assert fit == (640, 352, 390, 364)


def test_hud_cuda_submit_abandons_ready_buffer_if_resize_retires_interop() -> None:
    presenter = _hud_presenter_without_window()
    mark_calls = 0

    class _Interop:
        def ready_rgba_buffer(self):
            return object(), object()

        def mark_submitted(self, *args: object) -> None:
            nonlocal mark_calls
            mark_calls += 1

    interop = _Interop()
    presenter._cuda_hud_interop = interop

    def sync_window_size() -> None:
        presenter._cuda_hud_interop = None

    presenter._sync_window_size = sync_window_size

    assert not presenter._submit_ready_cuda_hud()
    assert mark_calls == 0


def test_hud_cuda_submit_does_not_forward_copy_stream() -> None:
    submitted_kwargs: list[dict[str, object]] = []
    marked: list[tuple[object, int]] = []
    surface_events: list[str] = []
    buffer = SimpleNamespace(
        buffer=object(),
        size_bytes=16,
        row_pitch=8,
    )

    class _Interop:
        def ready_rgba_buffer(self) -> tuple[object, object]:
            return buffer, object()

        def mark_submitted(self, submitted_buffer: object, submit_id: int) -> None:
            marked.append((submitted_buffer, submit_id))

    class _Encoder:
        def copy_buffer_to_texture(self, *args: object) -> None:
            del args

        def blit(self, *args: object) -> None:
            del args

        def finish(self) -> object:
            return object()

    class _Device:
        def create_command_encoder(self) -> _Encoder:
            return _Encoder()

        def submit_command_buffer(self, command: object, **kwargs: object) -> int:
            del command
            submitted_kwargs.append(kwargs)
            return 7

    class _Surface:
        config = object()

        def acquire_next_image(self) -> object:
            class _SurfaceTexture:
                def __del__(self) -> None:
                    surface_events.append("release")

            return _SurfaceTexture()

        def present(self) -> None:
            surface_events.append("present")

    presenter = _hud_presenter_without_window()
    presenter._cuda_hud_interop = _Interop()
    presenter._sync_window_size = lambda: None
    presenter._configured_size = (2, 2)
    presenter._surface = _Surface()
    presenter._device = _Device()
    presenter._display_texture = object()

    assert presenter._submit_ready_cuda_hud()
    assert submitted_kwargs == [{}]
    assert marked == [(buffer, 7)]
    assert surface_events == ["present", "release"]


def test_hud_model_rgb_falls_back_to_host_when_cuda_path_declines() -> None:
    presenter = _hud_presenter_without_window()
    lazy = _LazyFrame()
    presented: list[object] = []

    presenter._pending_resize = None
    presenter._present_cuda_hud_frame = lambda frame, rgb: False
    presenter._update_camera_pil = lambda rgb: presented.append(rgb)
    presenter._render_canvas = lambda status_message: None
    presenter._present_canvas = lambda *args, **kwargs: None

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert presented == [lazy]


def test_hud_model_rgb_falls_back_to_host_when_cuda_path_raises() -> None:
    presenter = _hud_presenter_without_window()
    lazy = _LazyFrame()
    presented: list[object] = []
    close_calls = 0

    class _Interop:
        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    def raise_cuda(frame: PresentedFrame, rgb: object) -> bool:
        del frame, rgb
        raise RuntimeError("cuda blend failed")

    presenter._pending_resize = None
    presenter._cuda_hud_interop = _Interop()
    presenter._cuda_hud_error_logged = False
    presenter._present_cuda_hud_frame = raise_cuda
    presenter._update_camera_pil = lambda rgb: presented.append(rgb)
    presenter._render_canvas = lambda status_message: None
    presenter._present_canvas = lambda *args, **kwargs: None

    frame = PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
        model_rgb_host_uint8=lazy,
    )

    presenter.present_frame(frame, view_mode="model_rgb")

    assert presented == [lazy]
    assert presenter._cuda_hud_interop is None
    assert close_calls == 1


class _ExitSceneKeyboard:
    def __init__(self) -> None:
        self.cleared = 0

    def clear_telemetry(self) -> None:
        self.cleared += 1


def _hud_presenter_for_exit(selected_variant: str) -> SlangPyHudPresenter:
    """Window-less HUD presenter wired with just the state exit-to-selector touches."""
    from pathlib import Path

    from omnidreams.interactive_drive.demo import SceneOption

    presenter = _hud_presenter_without_window()
    scene_path = Path("clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz")
    option = SceneOption(
        label="Quiet Suburban Boulevard",
        path=scene_path,
        variants=("default", "rain", "snow"),
        variant_paths={"default": scene_path},
    )
    presenter._scene_options = (option,)
    presenter._current_scene = scene_path
    presenter._selected_variant = selected_variant
    presenter._pending_exit_scene = True
    presenter._should_close_flag = True
    presenter._keyboard = _ExitSceneKeyboard()
    # State cleared by _reset_scene_view_state.
    presenter._scene_dropdown_open = True
    presenter._variant_dropdown_open = True
    presenter._camera_resize_cache_key = object()
    presenter._camera_resize_cache = object()
    presenter._latest_camera_pil = object()
    presenter._latest_bev_source = object()
    presenter._bev_source_generation = 1
    presenter._bev_panel_cache_key = object()
    presenter._bev_panel_cache = object()
    presenter._panel_chrome_cache_key = object()
    presenter._panel_chrome_cache = object()
    presenter._has_camera_frame = True
    presenter._speed_mph = 42.0
    presenter._pending_drive_releases = {"w": 1.0}
    return presenter


def test_acknowledge_exit_scene_resets_variant_to_scene_default() -> None:
    # Exiting a rain/snow rollout must not leave the selector header stuck on
    # the exited variant.
    presenter = _hud_presenter_for_exit(selected_variant="rain")

    presenter.acknowledge_exit_scene()

    assert presenter._selected_variant == "default"
    assert presenter._pending_exit_scene is False
    assert presenter._should_close_flag is False
    assert presenter._has_camera_frame is False


def test_acknowledge_exit_scene_falls_back_to_default_when_scene_unknown() -> None:
    presenter = _hud_presenter_for_exit(selected_variant="snow")
    # Current scene no longer matches any discovered option.
    presenter._scene_options = ()

    presenter.acknowledge_exit_scene()

    assert presenter._selected_variant == "default"


def _hud_presenter_for_preload() -> SlangPyHudPresenter:
    presenter = _hud_presenter_without_window()
    presenter._engine_active = True
    presenter._should_close_flag = False
    presenter._window = SimpleNamespace(should_close=lambda: False)
    presenter.process_events = lambda: None
    presenter._present_canvas = lambda *a, **k: None
    return presenter


def test_wait_while_preloading_pumps_until_in_progress_clears() -> None:
    presenter = _hud_presenter_for_preload()
    renders = 0

    def render(_status: object) -> None:
        nonlocal renders
        renders += 1

    presenter._render_canvas = render

    states = iter([True, True, False])
    presenter.wait_while_preloading(lambda: next(states))

    assert renders == 2
    assert presenter._engine_active is True  # restored to its prior value


def test_wait_while_preloading_stops_when_window_closes() -> None:
    presenter = _hud_presenter_for_preload()
    presenter._should_close_flag = True
    renders = 0

    def render(_status: object) -> None:
        nonlocal renders
        renders += 1

    presenter._render_canvas = render

    # Still "in progress", but a closed window must short-circuit the wait.
    presenter.wait_while_preloading(lambda: True)

    assert renders == 0
