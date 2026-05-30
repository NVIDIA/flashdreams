# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omnidreams.interactive_drive.presenter import (
    SlangPyPresenter,
    _CudaRGBFrame,
    _CudaRGBInterop,
)
from omnidreams.interactive_drive.slangpy_hud_presenter import SlangPyHudPresenter
from omnidreams.interactive_drive.types import PresentedFrame


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


def test_cuda_existing_device_handles_skips_by_default(monkeypatch) -> None:
    presenter = _presenter_without_window()

    class _Spy:
        @staticmethod
        def get_cuda_current_context_native_handles() -> list[object]:
            raise AssertionError("native handle query should be opt-in")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    assert presenter._cuda_existing_device_handles() == []


def test_create_device_disables_cuda_interop_when_torch_cuda_is_initialized(
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
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    presenter._spy = _Spy()

    presenter._create_device()

    assert created_kwargs[0]["enable_cuda_interop"] is False
    assert (
        presenter._cuda_interop_unavailable_reason
        == "disabled after torch CUDA initialization"
    )


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
