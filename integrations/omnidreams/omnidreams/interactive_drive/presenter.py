# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
from typing import Any

import numpy as np
from loguru import logger
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.cuda_env import (
    DISABLE_CUDA_INTEROP_ENV,
    env_truthy,
)
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.loading_overlay import render_loading_overlay
from omnidreams.interactive_drive.types import PresentedFrame

from flashdreams.infra.acceleration.frame_prefetch import prefetch_to_numpy


class SlangPyPresenter:
    def __init__(self, raster: RasterConfig, keyboard: KeyboardState) -> None:
        try:
            import slangpy as spy
        except ImportError as exc:
            raise RuntimeError(
                "SlangPy is required for the presenter. Install with"
                " `uv sync --package flashdreams-omnidreams --extra interactive-drive`."
            ) from exc

        self._spy = spy
        self._raster = raster
        self._keyboard = keyboard
        self._cuda_interop_unavailable_reason: str | None = None
        self._window = spy.Window(
            width=raster.width,
            height=raster.height,
            title="interactive_drive",
            resizable=False,
        )
        self._device = self._create_device()
        logger.info(f"[presenter] device={self._device.info.adapter_name}")
        self._surface = self._device.create_surface(self._window)
        self._surface_format = self._choose_surface_format()
        self._display_format = spy.Format.rgba8_unorm
        logger.info(
            f"[presenter] surface preferred={self._surface.info.preferred_format} chosen={self._surface_format} display={self._display_format}",
        )
        self._surface.configure(
            width=raster.width, height=raster.height, format=self._surface_format
        )
        self._display_texture = self._device.create_texture(
            format=self._display_format,
            width=raster.width,
            height=raster.height,
            usage=(
                spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access
                | spy.TextureUsage.copy_destination
            ),
            label="display_texture",
        )
        self._cuda_rgb_interop = self._create_cuda_rgb_interop()
        self._key_codes = self._build_key_codes()
        self._window.on_keyboard_event = self._on_keyboard_event

    @property
    def should_close(self) -> bool:
        return self._window.should_close()

    def close(self) -> None:
        if self._cuda_rgb_interop is not None:
            self._cuda_rgb_interop.close()
            self._cuda_rgb_interop = None
        self._window.close()

    def process_events(self) -> None:
        self._window.process_events()

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if (
            view_mode == "model_rgb"
            and frame.model_rgb_host_uint8 is not None
            and self._cuda_rgb_interop is None
        ):
            _prefetch_to_numpy(frame.model_rgb_host_uint8)
            return
        if view_mode != "model_rgb":
            _prefetch_to_numpy(frame.rgb_host_uint8)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        if view_mode == "model_rgb" and frame.model_rgb_host_uint8 is not None:
            if self._present_cuda_rgb(
                frame.model_rgb_host_uint8,
                status_message=frame.status_message,
            ):
                return
            rgb = _with_status_overlay(frame.model_rgb_host_uint8, frame.status_message)
            self._present_array(rgb)
            return
        self._present_array(
            _with_status_overlay(frame.rgb_host_uint8, frame.status_message)
        )

    def _create_device(self):
        existing_device_handles = self._cuda_existing_device_handles()
        enable_cuda_interop = not env_truthy(DISABLE_CUDA_INTEROP_ENV)
        if not enable_cuda_interop:
            self._cuda_interop_unavailable_reason = (
                f"disabled by {DISABLE_CUDA_INTEROP_ENV}"
            )
        device_kwargs = {
            "type": self._spy.DeviceType.vulkan,
            "enable_debug_layers": False,
            "enable_cuda_interop": enable_cuda_interop,
            "enable_cuda_launch_from_gfx": False,
            "enable_ray_tracing": False,
        }
        if existing_device_handles:
            device_kwargs["existing_device_handles"] = existing_device_handles
        try:
            return self._spy.Device(**device_kwargs)
        except RuntimeError as exc:
            logger.info(
                "[presenter] CUDA interop device creation failed; retrying Vulkan without "
                f"interop ({exc})",
            )
            self._cuda_interop_unavailable_reason = "device creation failed"
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
            )

    def _cuda_existing_device_handles(self) -> list[Any]:
        if env_truthy(DISABLE_CUDA_INTEROP_ENV):
            return []
        try:
            import torch
        except ImportError:
            return []
        try:
            if not torch.cuda.is_initialized():
                return []
        except Exception:
            return []

        get_handles = getattr(
            self._spy, "get_cuda_current_context_native_handles", None
        )
        if not callable(get_handles):
            return []
        try:
            handles: Any = get_handles()
            return list(handles)
        except Exception:
            return []

    def _create_cuda_rgb_interop(self):
        if env_truthy(DISABLE_CUDA_INTEROP_ENV):
            logger.info(
                f"[presenter] cuda_interop=disabled by {DISABLE_CUDA_INTEROP_ENV}; "
                "using host RGB upload",
            )
            return None
        if not self._device.supports_cuda_interop:
            reason = self._cuda_interop_unavailable_reason or "unsupported"
            logger.info(f"[presenter] cuda_interop={reason}; using host RGB upload")
            return None
        try:
            interop = _CudaRGBInterop(
                spy=self._spy,
                device=self._device,
                width=self._raster.width,
                height=self._raster.height,
            )
        except Exception as exc:
            logger.info(
                f"[presenter] cuda_interop=unavailable; using host RGB upload ({exc})",
            )
            return None
        logger.info("[presenter] cuda_interop=enabled")
        return interop

    def _present_cuda_rgb(
        self, rgb_frame: object, *, status_message: str | None
    ) -> bool:
        if self._cuda_rgb_interop is None:
            return False

        cuda_rgb_frame = self._cuda_rgb_interop.as_cuda_rgb_frame(rgb_frame)
        if cuda_rgb_frame is None:
            return False

        if not cuda_rgb_frame.ready:
            self._submit_ready_cuda_rgb()
            return True

        # The loading/status overlay is rendered on the CPU today. Fall back
        # only after the CUDA producer is done so this path does not become an
        # accidental long UI-thread synchronization.
        if status_message is not None:
            return False

        submitted = self._submit_ready_cuda_rgb()
        self._cuda_rgb_interop.enqueue_rgb_to_shared_rgba(cuda_rgb_frame)
        if not submitted:
            submitted = self._submit_ready_cuda_rgb()
        return True

    def _submit_ready_cuda_rgb(self) -> bool:
        if self._cuda_rgb_interop is None:
            return False
        interop_frame = self._cuda_rgb_interop.ready_rgba_buffer()
        if interop_frame is None:
            return False
        rgba_buffer, cuda_stream = interop_frame
        if not self._surface.config:
            return False
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            time.sleep(0.001)
            return False

        command_encoder = self._device.create_command_encoder()
        command_encoder.copy_buffer_to_texture(
            self._display_texture,
            0,
            0,
            [0, 0, 0],
            rgba_buffer.buffer,
            0,
            rgba_buffer.size_bytes,
            rgba_buffer.row_pitch,
            [self._raster.width, self._raster.height, 1],
        )
        command_encoder.blit(surface_texture, self._display_texture)
        submit_id = self._device.submit_command_buffer(
            command_encoder.finish(),
            cuda_stream=cuda_stream,
        )
        self._cuda_rgb_interop.mark_submitted(rgba_buffer, submit_id)
        del surface_texture
        self._surface.present()
        return True

    def _present_array(self, rgb_host_uint8: np.ndarray) -> None:
        if not self._surface.config:
            return
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            time.sleep(0.001)
            return

        upload = self._pack_surface_pixels(rgb_host_uint8)
        self._display_texture.copy_from_numpy(upload)

        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(surface_texture, self._display_texture)
        self._device.submit_command_buffer(command_encoder.finish())
        del surface_texture
        self._surface.present()

    def _choose_surface_format(self):
        linear_pairs = {
            self._spy.Format.rgba8_unorm_srgb: self._spy.Format.rgba8_unorm,
            self._spy.Format.bgra8_unorm_srgb: self._spy.Format.bgra8_unorm,
            self._spy.Format.bgrx8_unorm_srgb: self._spy.Format.bgrx8_unorm,
        }
        preferred = self._surface.info.preferred_format
        supported = list(self._surface.info.formats)

        for candidate in (
            self._spy.Format.rgba8_unorm,
            self._spy.Format.bgra8_unorm,
            self._spy.Format.bgrx8_unorm,
        ):
            if candidate in supported:
                return candidate

        preferred_linear = linear_pairs.get(preferred, preferred)
        if preferred_linear in supported:
            return preferred_linear

        raise RuntimeError(
            f"Presenter requires a linear swapchain, but the surface only supports: {supported}"
        )

    def _pack_surface_pixels(self, rgb_host_uint8: np.ndarray) -> np.ndarray:
        upload = np.zeros((self._raster.height, self._raster.width, 4), dtype=np.uint8)
        upload[..., :3] = rgb_host_uint8
        upload[..., 3] = 255
        return upload

    def _on_keyboard_event(self, event) -> None:
        is_press = event.is_key_press() if hasattr(event, "is_key_press") else False
        is_release = (
            event.is_key_release() if hasattr(event, "is_key_release") else False
        )
        if not (is_press or is_release):
            return

        if self._matches_key(event.key, "escape") and is_press:
            self.close()
            return

        key_map = {
            self._key_codes["w"]: "w",
            self._key_codes["a"]: "a",
            self._key_codes["s"]: "s",
            self._key_codes["d"]: "d",
            self._key_codes["up"]: "up",
            self._key_codes["left"]: "left",
            self._key_codes["down"]: "down",
            self._key_codes["right"]: "right",
        }
        key_map = {
            key_code: name for key_code, name in key_map.items() if key_code is not None
        }
        if event.key in key_map:
            self._keyboard.set_key(key_map[event.key], is_press)
            return

        if is_press and self._matches_key(event.key, "key1"):
            self._keyboard.set_view_mode("model_rgb")
        elif is_press and self._matches_key(event.key, "key2"):
            self._keyboard.set_view_mode("rgb")
        elif is_press and self._matches_key(event.key, "r"):
            self._keyboard.request_reset()

    def _build_key_codes(self) -> dict[str, object | None]:
        return {
            "escape": self._lookup_key_code("escape"),
            "w": self._lookup_key_code("w"),
            "a": self._lookup_key_code("a"),
            "s": self._lookup_key_code("s"),
            "d": self._lookup_key_code("d"),
            "r": self._lookup_key_code("r"),
            "up": self._lookup_key_code("up", "arrow_up"),
            "left": self._lookup_key_code("left", "arrow_left"),
            "down": self._lookup_key_code("down", "arrow_down"),
            "right": self._lookup_key_code("right", "arrow_right"),
            "key1": self._lookup_key_code("key1", "digit1", "num_1"),
            "key2": self._lookup_key_code("key2", "digit2", "num_2"),
        }

    def _lookup_key_code(self, *names: str) -> object | None:
        for name in names:
            value = getattr(self._spy.KeyCode, name, None)
            if value is not None:
                return value
        return None

    def _matches_key(self, event_key: object, name: str) -> bool:
        key_code = self._key_codes.get(name)
        return key_code is not None and event_key == key_code


class _CudaRGBInterop:
    def __init__(self, *, spy: Any, device: Any, width: int, height: int) -> None:
        import torch

        self._spy = spy
        self._device = device
        self._torch = torch
        self._width = int(width)
        self._height = int(height)
        self._row_pitch = self._width * 4
        self._size_bytes = self._row_pitch * self._height
        self._buffers = [
            _SharedRGBABuffer(
                buffer=device.create_buffer(
                    size=self._size_bytes,
                    usage=spy.BufferUsage.shared | spy.BufferUsage.copy_source,
                    label=f"display_cuda_rgba_buffer_{index}",
                ),
                row_pitch=self._row_pitch,
                size_bytes=self._size_bytes,
                rgba_tensor=None,
                copy_done_event=None,
                pending_submit_id=None,
            )
            for index in range(3)
        ]
        for shared_buffer in self._buffers:
            shared_buffer.rgba_tensor = shared_buffer.buffer.to_torch(
                type=spy.DataType.uint8,
                shape=[self._height, self._width, 4],
            )
        self._next_buffer_index = 0
        first_tensor = self._buffers[0].rgba_tensor
        if first_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        self._cuda_device = first_tensor.device
        self._copy_stream = _NonBlockingCudaStream(self._torch, self._cuda_device)
        self._device_mismatch_logged = False

    def as_cuda_rgb_frame(self, rgb_frame: object) -> "_CudaRGBFrame | None":
        cuda_frame = self.as_cuda_rgb_source(rgb_frame)
        if cuda_frame is None:
            return None
        if tuple(cuda_frame.tensor.shape) != (self._height, self._width, 3):
            return None
        return cuda_frame

    def as_cuda_rgb_source(self, rgb_frame: object) -> "_CudaRGBFrame | None":
        to_cuda_tensor = getattr(rgb_frame, "to_cuda_tensor", None)
        try:
            tensor = to_cuda_tensor() if callable(to_cuda_tensor) else rgb_frame
        except RuntimeError:
            return None
        if not self._torch.is_tensor(tensor):
            return None
        if not tensor.is_cuda or tensor.dtype != self._torch.uint8:
            return None
        if self._cuda_device_index(tensor.device) != self._cuda_device_index(
            self._cuda_device
        ):
            if not self._device_mismatch_logged:
                logger.info(
                    "[presenter] cuda_interop skipped: model RGB tensor is on "
                    f"{tensor.device}, presenter shared buffer is on {self._cuda_device}",
                )
                self._device_mismatch_logged = True
            return None
        if tensor.ndim != 3 or tensor.shape[-1] < 3:
            return None
        to_cuda_event = getattr(rgb_frame, "to_cuda_event", None)
        source_event = to_cuda_event() if callable(to_cuda_event) else None
        return _CudaRGBFrame(
            tensor=tensor[..., :3].detach(),
            source_event=source_event,
            ready=_cuda_event_ready(source_event),
        )

    def enqueue_rgb_to_shared_rgba(self, rgb_frame: "_CudaRGBFrame") -> bool:
        shared_buffer = self._acquire_buffer()
        if shared_buffer is None:
            return False
        rgba_tensor = shared_buffer.rgba_tensor
        if rgba_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        rgb_tensor = rgb_frame.tensor
        if rgb_frame.source_event is not None:
            self._copy_stream.stream.wait_event(rgb_frame.source_event)
        with self._torch.cuda.stream(self._copy_stream.stream):
            if not rgb_tensor.is_contiguous():
                rgb_tensor = rgb_tensor.contiguous()
            rgba_tensor[..., :3].copy_(rgb_tensor, non_blocking=True)
            rgba_tensor[..., 3].fill_(255)
            rgb_tensor.record_stream(self._copy_stream.stream)
            rgba_tensor.record_stream(self._copy_stream.stream)
            copy_done_event = self._torch.cuda.Event()
            copy_done_event.record(self._copy_stream.stream)
        shared_buffer.copy_done_event = copy_done_event
        return True

    def enqueue_camera_to_shared_rgba(
        self,
        rgb_frame: "_CudaRGBFrame",
        *,
        overlay_rgba: np.ndarray,
        camera_area: tuple[int, int, int, int],
        bg_rgb: tuple[int, int, int],
    ) -> bool:
        shared_buffer = self._acquire_buffer()
        if shared_buffer is None:
            return False
        rgba_tensor = shared_buffer.rgba_tensor
        if rgba_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")

        overlay = np.ascontiguousarray(overlay_rgba, dtype=np.uint8)
        if tuple(overlay.shape) != (self._height, self._width, 4):
            raise ValueError(
                "HUD overlay shape does not match shared display buffer: "
                f"{tuple(overlay.shape)} vs {(self._height, self._width, 4)}"
            )

        rgb_tensor = rgb_frame.tensor
        if rgb_frame.source_event is not None:
            self._copy_stream.stream.wait_event(rgb_frame.source_event)

        ax, ay, ar, ab = camera_area
        area_w = max(1, int(ar) - int(ax))
        area_h = max(1, int(ab) - int(ay))
        src_h = int(rgb_tensor.shape[0])
        src_w = int(rgb_tensor.shape[1])
        if src_h <= 0 or src_w <= 0:
            return False
        scale = min(area_w / src_w, area_h / src_h)
        target_w = max(1, int(src_w * scale))
        target_h = max(1, int(src_h * scale))
        target_x = int(ax) + (area_w - target_w) // 2
        target_y = int(ay) + (area_h - target_h) // 2

        with self._torch.cuda.stream(self._copy_stream.stream):
            rgba_tensor[..., 0].fill_(int(bg_rgb[0]))
            rgba_tensor[..., 1].fill_(int(bg_rgb[1]))
            rgba_tensor[..., 2].fill_(int(bg_rgb[2]))
            rgba_tensor[..., 3].fill_(255)

            if not rgb_tensor.is_contiguous():
                rgb_tensor = rgb_tensor.contiguous()
            resized = self._resize_rgb_tensor(rgb_tensor, target_h, target_w)
            rgba_tensor[
                target_y : target_y + target_h,
                target_x : target_x + target_w,
                :3,
            ].copy_(resized, non_blocking=True)

            overlay_tensor = self._torch.from_numpy(overlay).to(
                device=self._cuda_device,
                non_blocking=True,
            )
            self._alpha_composite_rgba(rgba_tensor, overlay_tensor)

            rgb_tensor.record_stream(self._copy_stream.stream)
            resized.record_stream(self._copy_stream.stream)
            overlay_tensor.record_stream(self._copy_stream.stream)
            rgba_tensor.record_stream(self._copy_stream.stream)
            copy_done_event = self._torch.cuda.Event()
            copy_done_event.record(self._copy_stream.stream)
        shared_buffer.copy_done_event = copy_done_event
        return True

    def ready_rgba_buffer(self) -> tuple["_SharedRGBABuffer", Any] | None:
        for offset in range(len(self._buffers)):
            index = (self._next_buffer_index + offset) % len(self._buffers)
            shared_buffer = self._buffers[index]
            copy_done_event = shared_buffer.copy_done_event
            if copy_done_event is None or not _cuda_event_ready(copy_done_event):
                continue
            stream = int(self._copy_stream.cuda_stream)
            cuda_stream = self._spy.NativeHandle(
                self._spy.NativeHandleType.CUstream, stream
            )
            return shared_buffer, cuda_stream
        return None

    def close(self) -> None:
        self._copy_stream.close()

    def mark_submitted(
        self, shared_buffer: "_SharedRGBABuffer", submit_id: int
    ) -> None:
        shared_buffer.copy_done_event = None
        shared_buffer.pending_submit_id = int(submit_id)

    def _acquire_buffer(self) -> "_SharedRGBABuffer | None":
        for offset in range(len(self._buffers)):
            index = (self._next_buffer_index + offset) % len(self._buffers)
            shared_buffer = self._buffers[index]
            if shared_buffer.copy_done_event is not None:
                continue
            if shared_buffer.pending_submit_id is None:
                self._next_buffer_index = (index + 1) % len(self._buffers)
                return shared_buffer
            if self._device.is_submit_finished(shared_buffer.pending_submit_id):
                shared_buffer.pending_submit_id = None
                self._next_buffer_index = (index + 1) % len(self._buffers)
                return shared_buffer
        return None

    def _cuda_device_index(self, device: Any) -> int:
        index = device.index
        return 0 if index is None else int(index)

    def _resize_rgb_tensor(self, rgb_tensor: Any, target_h: int, target_w: int) -> Any:
        if tuple(rgb_tensor.shape[:2]) == (target_h, target_w):
            return rgb_tensor if rgb_tensor.is_contiguous() else rgb_tensor.contiguous()
        nchw = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self._torch.float32)
        resized = self._torch.nn.functional.interpolate(
            nchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return (
            resized[0]
            .permute(1, 2, 0)
            .round()
            .clamp_(0, 255)
            .to(self._torch.uint8)
            .contiguous()
        )

    def _alpha_composite_rgba(self, base_rgba: Any, overlay_rgba: Any) -> None:
        alpha = overlay_rgba[..., 3:4].to(self._torch.float32) * (1.0 / 255.0)
        blended = (
            overlay_rgba[..., :3].to(self._torch.float32) * alpha
            + base_rgba[..., :3].to(self._torch.float32) * (1.0 - alpha)
        ).round()
        base_rgba[..., :3].copy_(blended.to(self._torch.uint8), non_blocking=True)
        base_rgba[..., 3].fill_(255)


class _CudaRGBFrame:
    def __init__(self, *, tensor: Any, source_event: Any | None, ready: bool) -> None:
        self.tensor = tensor
        self.source_event = source_event
        self.ready = ready


class _NonBlockingCudaStream:
    def __init__(self, torch_module: Any, device: Any) -> None:
        import ctypes
        import ctypes.util

        self._runtime = None
        self._stream_ptr = 0
        self._stream = None

        library_name = ctypes.util.find_library("cudart") or "libcudart.so"
        runtime = ctypes.CDLL(library_name)
        cuda_set_device = runtime.cudaSetDevice
        cuda_set_device.argtypes = [ctypes.c_int]
        cuda_set_device.restype = ctypes.c_int
        cuda_stream_create = runtime.cudaStreamCreateWithFlags
        cuda_stream_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        cuda_stream_create.restype = ctypes.c_int
        cuda_get_error_string = runtime.cudaGetErrorString
        cuda_get_error_string.argtypes = [ctypes.c_int]
        cuda_get_error_string.restype = ctypes.c_char_p

        device_index = 0 if device.index is None else int(device.index)
        _check_cuda_runtime_result(cuda_set_device(device_index), cuda_get_error_string)
        stream = ctypes.c_void_p()
        cuda_stream_non_blocking = 1
        _check_cuda_runtime_result(
            cuda_stream_create(ctypes.byref(stream), cuda_stream_non_blocking),
            cuda_get_error_string,
        )

        self._runtime = runtime
        self._stream_ptr = int(stream.value or 0)
        self._stream = torch_module.cuda.ExternalStream(self._stream_ptr, device=device)

    @property
    def stream(self) -> Any:
        return self._stream

    @property
    def cuda_stream(self) -> int:
        return self._stream_ptr

    def close(self) -> None:
        if self._runtime is None or self._stream_ptr == 0:
            return
        import ctypes

        stream = self._stream
        if stream is not None:
            stream.synchronize()
        cuda_stream_destroy = self._runtime.cudaStreamDestroy
        cuda_stream_destroy.argtypes = [ctypes.c_void_p]
        cuda_stream_destroy.restype = ctypes.c_int
        cuda_stream_destroy(ctypes.c_void_p(self._stream_ptr))
        self._runtime = None
        self._stream_ptr = 0
        self._stream = None


class _SharedRGBABuffer:
    def __init__(
        self,
        *,
        buffer: Any,
        row_pitch: int,
        size_bytes: int,
        rgba_tensor: Any | None,
        copy_done_event: Any | None,
        pending_submit_id: int | None,
    ) -> None:
        self.buffer = buffer
        self.row_pitch = row_pitch
        self.size_bytes = size_bytes
        self.rgba_tensor = rgba_tensor
        self.copy_done_event = copy_done_event
        self.pending_submit_id = pending_submit_id


def _check_cuda_runtime_result(result: int, get_error_string: Any) -> None:
    if result == 0:
        return
    raw = get_error_string(int(result))
    message = (
        raw.decode("utf-8", errors="replace")
        if raw is not None
        else f"CUDA error {result}"
    )
    raise RuntimeError(message)


def _cuda_event_ready(event: Any | None) -> bool:
    if event is None:
        return True
    query = getattr(event, "query", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except RuntimeError:
        return False


_env_truthy = env_truthy


def _prefetch_to_numpy(frame: object) -> None:
    prefetch_to_numpy(frame)


def _with_status_overlay(rgb_host_uint8: object, message: str | None) -> np.ndarray:
    rgb_host_uint8 = _as_rgb_host_uint8(rgb_host_uint8)
    if message is None:
        return rgb_host_uint8
    return render_loading_overlay(rgb_host_uint8, message=message)


def _as_rgb_host_uint8(frame: object) -> np.ndarray:
    to_numpy = getattr(frame, "to_numpy", None)
    if callable(to_numpy):
        frame = to_numpy()
    return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])
