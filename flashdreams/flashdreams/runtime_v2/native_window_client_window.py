# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy native client window with GPU-resident presentation."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy import uint64
from torch import Tensor

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor

if TYPE_CHECKING:
    import slangpy as spy

_LOGGER = logging.getLogger(__name__)

_CUDA_COPY_POLL_SECONDS = 0.001
"""Delay between checks for an already-enqueued CUDA presentation copy."""

_PRINTABLE_KEY_NAMES = {
    "space": " ",
    "apostrophe": "'",
    "comma": ",",
    "minus": "-",
    "period": ".",
    "slash": "/",
    "semicolon": ";",
    "equal": "=",
    "left_bracket": "[",
    "backslash": "\\",
    "right_bracket": "]",
    "grave_accent": "`",
}
"""Map SlangPy printable key names to browser-style key values."""

_MODIFIER_KEY_NAMES = {
    "left_alt": "Alt",
    "right_alt": "Alt",
    "left_control": "Control",
    "right_control": "Control",
    "left_shift": "Shift",
    "right_shift": "Shift",
    "left_super": "Meta",
    "right_super": "Meta",
}
"""Map physical SlangPy modifiers to browser ``KeyboardEvent.key`` values."""


@dataclass(frozen=True, slots=True)
class _CudaPresentationFrame:
    """CUDA RGB tensor and the event that makes it safe for SlangPy to read."""

    tensor: Tensor
    """Contiguous ``[H, W, 3]`` uint8 CUDA tensor."""

    ready_event: torch.cuda.Event
    """CUDA event recorded after conversion to presentation format."""


class NativeWindowClientWindow(IClientWindow):
    """Present UI output through a main-thread GLFW window."""

    def __init__(
        self,
        *,
        title: str = "FlashDreams",
        presenter_factory: Callable[..., _SlangPyNativeWindowPresenter] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        """Configure the native client window.

        Args:
            title: Native window title.
            presenter_factory: Optional SlangPy-compatible presenter factory.
            clock_ns: Monotonic clock used for input timestamps.

        Raises:
            ValueError: ``title`` is empty.
        """
        if not title.strip():
            raise ValueError("Native-window title must be non-empty.")
        self.title = title
        self._presenter_factory = presenter_factory
        self._clock_ns = clock_ns
        self._session_started_ns: int | None = None
        self._session_desc: SessionDesc | None = None
        self._input_events: queue.SimpleQueue[UserInputEvent] = queue.SimpleQueue()
        self._close_event_enqueued = False
        self._presenter: _SlangPyNativeWindowPresenter | None = None
        self._poll_input_events: list[UserInputEvent] | None = None
        self._pending_printable_keys: deque[tuple[str, int]] = deque()
        self._pressed_key_values: dict[str, str] = {}

    def open(self, session_desc: SessionDesc) -> None:
        """Create the GLFW window on the runtime's UI thread.

        Args:
            session_desc: Resolved output dimensions and tensor layout.

        Raises:
            RuntimeError: The window is already open or initialization fails.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "NativeWindowClientWindow.open() must run on the process main thread for event polling."
            )
        if self._presenter is not None:
            raise RuntimeError("NativeWindowClientWindow is already open.")
        presenter_factory = self._presenter_factory or _SlangPyNativeWindowPresenter

        presenter = presenter_factory(
            width=session_desc.video_width,
            height=session_desc.video_height,
            title=self.title,
        )
        try:
            presenter.set_input_callbacks(
                on_keyboard_event=self._on_keyboard_event,
                on_mouse_event=self._on_mouse_event,
            )
        except BaseException:
            presenter.close()
            raise

        self._session_started_ns = self._clock_ns()
        self._session_desc = session_desc
        self._input_events = queue.SimpleQueue()
        self._close_event_enqueued = False
        self._poll_input_events = None
        self._pending_printable_keys.clear()
        self._pressed_key_values.clear()
        self._presenter = presenter

    def get_user_input_events(self) -> UserInputEvents:
        """Pump GLFW and return native input events not yet read."""
        presenter = self._presenter
        if presenter is not None:
            self._poll_input_events = []
            try:
                presenter.process_events()
                if presenter.should_close:
                    self._on_window_closed()
                polled_input_events = self._poll_input_events
            finally:
                self._poll_input_events = None
                self._pending_printable_keys.clear()
            for event in polled_input_events:
                self._put_input(event)

        events = []
        while True:
            try:
                events.append(self._input_events.get_nowait())
            except queue.Empty:
                return UserInputEvents(events)

    def write(self, result: StepResult) -> None:
        """Convert and submit one result from the runtime's UI thread.

        CUDA output remains GPU-resident. The runtime synchronizes the producer
        stream before handing ``result`` to this method.

        Args:
            result: UI output to present.

        Raises:
            RuntimeError: The window is not open.
        """
        presenter = self._presenter
        if presenter is None:
            raise RuntimeError(
                "NativeWindowClientWindow.open() must run before write()."
            )
        if self._close_event_enqueued:
            return
        frames = result_to_rgb24_tensor(result, self._session_desc_or_raise())
        for frame in frames:
            presentation_frame: Tensor | _CudaPresentationFrame = frame
            if frame.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(frame.device))
                presentation_frame = _CudaPresentationFrame(frame, ready_event)
            if self._close_event_enqueued:
                return
            if not presenter.present_frame(presentation_frame):
                self._on_window_closed()
                return

    def close(self) -> None:
        """Release SlangPy and the GLFW window on the runtime's UI thread."""
        presenter = self._presenter
        self._presenter = None
        self._session_started_ns = None
        self._session_desc = None
        self._input_events = queue.SimpleQueue()
        self._poll_input_events = None
        self._pending_printable_keys.clear()
        self._pressed_key_values.clear()
        if presenter is not None:
            presenter.close()

    def _session_desc_or_raise(self) -> SessionDesc:
        session_desc = self._session_desc
        if session_desc is None:
            raise RuntimeError("Native window has no active session description.")
        return session_desc

    def _put_input(self, event: UserInputEvent) -> None:
        if self._poll_input_events is not None:
            self._poll_input_events.append(event)
            return
        started_ns = self._session_started_ns
        elapsed_ns = 0 if started_ns is None else max(0, self._clock_ns() - started_ns)
        self._input_events.put(
            replace(event, timestamp=uint64(elapsed_ns // 1_000))
        )

    def _on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if _is_keyboard_input(event):
            text = _keyboard_input_text(event)
            if text is not None:
                poll_input_events = self._poll_input_events
                if self._pending_printable_keys and poll_input_events is not None:
                    physical_key, event_index = self._pending_printable_keys.pop()
                    self._pressed_key_values[physical_key] = text
                    poll_input_events[event_index] = KeyboardUserInputEvent(
                        timestamp=uint64(0),
                        key=text,
                        state=KeyboardInputState.PRESSED,
                    )
                else:
                    self._put_input(
                        KeyboardUserInputEvent(
                            timestamp=uint64(0),
                            key=text,
                            state=KeyboardInputState.PRESSED,
                        )
                    )
            return

        keyboard_event = _keyboard_event(event)
        if keyboard_event is None:
            return
        if (
            keyboard_event.state is KeyboardInputState.PRESSED
            and len(keyboard_event.key) == 1
        ):
            poll_input_events = self._poll_input_events
            if poll_input_events is not None:
                event_index = len(poll_input_events)
                self._put_input(keyboard_event)
                self._pending_printable_keys.append((keyboard_event.key, event_index))
            else:
                self._put_input(keyboard_event)
            return
        if (
            keyboard_event.state is KeyboardInputState.RELEASED
            and len(keyboard_event.key) == 1
        ):
            pending_key = next(
                (
                    pending
                    for pending in self._pending_printable_keys
                    if pending[0] == keyboard_event.key
                ),
                None,
            )
            if pending_key is not None:
                self._pending_printable_keys.remove(pending_key)
            key = self._pressed_key_values.pop(keyboard_event.key, keyboard_event.key)
            self._put_input(
                KeyboardUserInputEvent(
                    timestamp=uint64(0), key=key, state=keyboard_event.state
                )
            )
            return
        self._put_input(keyboard_event)

    def _on_mouse_event(self, event: spy.MouseEvent) -> None:
        session_desc = self._session_desc
        if session_desc is None:
            return
        mouse_event = _mouse_event(
            event,
            width=session_desc.video_width,
            height=session_desc.video_height,
        )
        if mouse_event is not None:
            self._put_input(mouse_event)

    def _on_window_closed(self) -> None:
        if self._close_event_enqueued:
            return
        self._close_event_enqueued = True
        self._put_input(CloseUserInputEvent(timestamp=uint64(0)))


class _SlangPyNativeWindowPresenter:
    """Own the SlangPy window and its CUDA/Vulkan presentation resources."""

    def __init__(self, *, width: int, height: int, title: str) -> None:
        """Create a fixed-size SlangPy window and Vulkan surface.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            title: Native window title.

        Raises:
            RuntimeError: SlangPy is unavailable or cannot create the window.
        """
        try:
            import slangpy as spy
        except ImportError as exc:
            raise RuntimeError(
                "Native-window output requires SlangPy. Install "
                "``flashdreams[local-window]``."
            ) from exc

        self._spy = spy
        self._width = int(width)
        self._height = int(height)
        self._closed = False
        self._window = spy.Window(
            width=self._width,
            height=self._height,
            title=title,
            resizable=False,
        )
        self._keyboard_event_callback: Callable[[spy.KeyboardEvent], None] | None = None
        self._mouse_event_callback: Callable[[spy.MouseEvent], None] | None = None
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event
        self._cuda_interop_unavailable_reason: str | None = None
        self._device: spy.Device | None = None
        self._surface: spy.Surface | None = None
        self._display_texture: spy.Texture | None = None
        self._cuda_rgb_interop: _CudaRGBInterop | None = None
        self._host_upload = np.empty(
            (self._height, self._width, 4),
            dtype=np.uint8,
        )
        self._host_upload[..., 3] = 255

    def set_input_callbacks(
        self,
        *,
        on_keyboard_event: Callable[[spy.KeyboardEvent], None] | None = None,
        on_mouse_event: Callable[[spy.MouseEvent], None] | None = None,
    ) -> None:
        """Bind runtime input callbacks to the SlangPy window."""
        self._keyboard_event_callback = on_keyboard_event
        self._mouse_event_callback = on_mouse_event

    @property
    def should_close(self) -> bool:
        """Return whether the user or runtime requested window closure."""
        return self._closed or self._window.should_close()

    def process_events(self) -> None:
        """Pump pending events with SlangPy's standard window API."""
        if not self._closed:
            self._window.process_events()

    def present_frame(self, frame: Tensor | _CudaPresentationFrame) -> bool:
        """Present one RGB frame without pumping window events."""
        if self._closed:
            return False

        cuda_device = _cuda_frame_device(frame)
        cuda_resident = cuda_device is not None
        self._ensure_render_resources(cuda_device)
        if self._cuda_rgb_interop is not None:
            cuda_frame = self._cuda_rgb_interop.as_cuda_rgb_frame(frame)
            if cuda_frame is not None:
                if not self._cuda_rgb_interop.enqueue_rgb_to_shared_rgba(cuda_frame):
                    return True
                while not self._submit_ready_cuda_rgb():
                    if not self._wait_for_presentation_progress():
                        return False
                return True
            if cuda_resident:
                raise ValueError(
                    "CUDA native-window frames must be uint8 RGB tensors with "
                    f"shape {(self._height, self._width, 3)} on the interop device."
                )

        if cuda_resident:
            raise RuntimeError(
                "CUDA native-window presentation requires CUDA/Vulkan interop; "
                "refusing to copy the frame through CPU memory."
            )

        tensor = frame.tensor if isinstance(frame, _CudaPresentationFrame) else frame
        rgb = self._as_host_rgb(tensor)
        if tuple(rgb.shape) != (self._height, self._width, 3):
            raise ValueError(
                "Native-window frame shape does not match the configured surface: "
                f"{tuple(rgb.shape)} != {(self._height, self._width, 3)}."
            )
        self._host_upload[..., :3] = rgb
        self._present_host_upload()
        return True

    def close(self) -> None:
        """Release CUDA interoperability and native window resources."""
        if self._closed:
            return
        self._closed = True
        if self._device is not None:
            self._device.wait_for_idle()
        if self._cuda_rgb_interop is not None:
            self._cuda_rgb_interop.close()
            self._cuda_rgb_interop = None
        self._display_texture = None
        self._surface = None
        self._device = None
        self._window.close()

    def _on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        """Forward one keyboard event to the runtime callback."""
        if self._keyboard_event_callback is not None:
            self._keyboard_event_callback(event)

    def _on_mouse_event(self, event: spy.MouseEvent) -> None:
        """Forward one mouse event to the runtime callback."""
        if self._mouse_event_callback is not None:
            self._mouse_event_callback(event)

    def _wait_for_presentation_progress(self) -> bool:
        """Yield briefly while waiting for CUDA or Vulkan progress."""
        if self._closed:
            return False
        time.sleep(_CUDA_COPY_POLL_SECONDS)
        return True

    def _ensure_render_resources(self, cuda_device: torch.device | None) -> None:
        """Create presentation resources on the first frame's device."""
        if self._device is not None:
            return
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)
        self._initialize_render_resources(enable_cuda_interop=cuda_device is not None)

    def _initialize_render_resources(self, *, enable_cuda_interop: bool) -> None:
        """Create the Vulkan surface and optional CUDA interop resources."""
        self._device = self._create_device(enable_cuda_interop=enable_cuda_interop)
        self._surface = self._device.create_surface(self._window)
        self._surface.configure(
            width=self._width,
            height=self._height,
            format=self._choose_surface_format(),
        )
        self._display_texture = self._device.create_texture(
            format=self._spy.Format.rgba8_unorm,
            width=self._width,
            height=self._height,
            usage=(
                self._spy.TextureUsage.shader_resource
                | self._spy.TextureUsage.unordered_access
                | self._spy.TextureUsage.copy_destination
            ),
            label="flashdreams_v2_native_window_texture",
        )
        self._cuda_rgb_interop = self._create_cuda_rgb_interop()

    def _create_device(self, *, enable_cuda_interop: bool) -> spy.Device:
        """Create the Vulkan device, sharing the active CUDA context if possible."""
        existing_device_handles = (
            self._cuda_existing_device_handles() if enable_cuda_interop else []
        )
        try:
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_interop=bool(existing_device_handles),
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
                existing_device_handles=existing_device_handles or None,
            )
        except RuntimeError as exc:
            _LOGGER.info(
                "Native-window CUDA interop device creation failed; using Vulkan "
                "host upload: %s",
                exc,
            )
            self._cuda_interop_unavailable_reason = "device creation failed"
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
            )

    def _cuda_existing_device_handles(self) -> list[spy.NativeHandle]:
        """Return SlangPy handles for the CUDA context current on this thread."""
        try:
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            torch.cuda.set_device(torch.cuda.current_device())
            torch.cuda.current_stream()
        except Exception:
            self._cuda_interop_unavailable_reason = "CUDA context unavailable"
            return []

        try:
            return list(self._spy.get_cuda_current_context_native_handles())
        except Exception as exc:
            self._cuda_interop_unavailable_reason = (
                f"native handle query failed ({type(exc).__name__}: {exc})"
            )
            return []

    def _create_cuda_rgb_interop(self) -> _CudaRGBInterop | None:
        """Create the GPU-resident RGB upload path when SlangPy supports it."""
        device = self._device
        assert device is not None
        if not device.supports_cuda_interop:
            reason = self._cuda_interop_unavailable_reason or "unsupported"
            _LOGGER.info(
                "Native-window CUDA interop unavailable (%s); using host upload",
                reason,
            )
            return None
        try:
            interop = _CudaRGBInterop(
                spy=self._spy,
                device=device,
                width=self._width,
                height=self._height,
            )
        except Exception as exc:
            _LOGGER.info(
                "Native-window CUDA interop unavailable; using host upload: %s",
                exc,
            )
            return None
        _LOGGER.info("Native-window CUDA interop enabled")
        return interop

    def _submit_ready_cuda_rgb(self) -> bool:
        """Submit one copy-complete shared RGBA buffer to the swapchain."""
        assert self._cuda_rgb_interop is not None
        assert self._surface is not None
        assert self._device is not None
        assert self._display_texture is not None
        ready = self._cuda_rgb_interop.ready_rgba_buffer()
        if ready is None:
            return False
        rgba_buffer, cuda_stream = ready
        if not self._surface.config:
            self._cuda_rgb_interop.discard_ready(rgba_buffer)
            return True
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            self._cuda_rgb_interop.discard_ready(rgba_buffer)
            return True

        encoder = self._device.create_command_encoder()
        encoder.copy_buffer_to_texture(
            self._display_texture,
            0,
            0,
            [0, 0, 0],
            rgba_buffer.buffer,
            0,
            rgba_buffer.size_bytes,
            rgba_buffer.row_pitch,
            [self._width, self._height, 1],
        )
        encoder.blit(surface_texture, self._display_texture)
        submit_id = self._device.submit_command_buffer(
            encoder.finish(),
            cuda_stream=cuda_stream,
        )
        self._cuda_rgb_interop.mark_submitted(rgba_buffer, submit_id)
        self._surface.present()
        del surface_texture
        return True

    def _present_host_upload(self) -> None:
        """Upload and present one CPU-resident RGBA frame."""
        assert self._surface is not None
        assert self._device is not None
        assert self._display_texture is not None
        if not self._surface.config:
            return
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            return
        self._display_texture.copy_from_numpy(self._host_upload)
        encoder = self._device.create_command_encoder()
        encoder.blit(surface_texture, self._display_texture)
        self._device.submit_command_buffer(encoder.finish())
        self._surface.present()
        del surface_texture

    def _choose_surface_format(self) -> spy.Format:
        """Select a linear surface format for byte-exact RGB presentation."""
        assert self._surface is not None
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
            "Native-window output requires a linear swapchain; "
            f"supported formats: {supported}."
        )

    @staticmethod
    def _as_host_rgb(frame: Tensor) -> np.ndarray:
        """Return one CPU frame as contiguous uint8 RGB data."""
        return np.ascontiguousarray(frame.numpy(), dtype=np.uint8)


class _CudaRGBInterop:
    """Map triple-buffered SlangPy shared storage into CUDA tensors."""

    def __init__(
        self, *, spy: Any, device: spy.Device, width: int, height: int
    ) -> None:
        """Allocate and CUDA-map the shared RGBA presentation buffers."""
        self._spy = spy
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._row_pitch = self._width * 4
        self._size_bytes = self._row_pitch * self._height
        self._buffers = [
            _SharedRGBABuffer(
                buffer=device.create_buffer(
                    size=self._size_bytes,
                    usage=spy.BufferUsage.shared | spy.BufferUsage.copy_source,
                    label=f"flashdreams_v2_native_cuda_rgba_{index}",
                ),
                row_pitch=self._row_pitch,
                size_bytes=self._size_bytes,
            )
            for index in range(3)
        ]
        for shared_buffer in self._buffers:
            shared_buffer.rgba_tensor = cast(
                Tensor,
                shared_buffer.buffer.to_torch(
                    type=spy.DataType.uint8,
                    shape=[self._height, self._width, 4],
                ),
            )
        first_tensor = self._buffers[0].rgba_tensor
        if first_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        self._cuda_device = first_tensor.device
        self._copy_stream = torch.cuda.Stream(device=self._cuda_device)
        self._next_buffer_index = 0

    def as_cuda_rgb_frame(
        self, frame: Tensor | _CudaPresentationFrame
    ) -> _CudaRGBFrame | None:
        """Return a device-compatible CUDA RGB view when available."""
        tensor = frame.tensor if isinstance(frame, _CudaPresentationFrame) else frame
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.uint8
            or tensor.ndim != 3
            or tuple(tensor.shape) != (self._height, self._width, 3)
        ):
            return None
        if self._device_index(tensor.device) != self._device_index(self._cuda_device):
            return None
        source_event = (
            frame.ready_event if isinstance(frame, _CudaPresentationFrame) else None
        )
        return _CudaRGBFrame(tensor=tensor.detach(), source_event=source_event)

    def enqueue_rgb_to_shared_rgba(self, frame: _CudaRGBFrame) -> bool:
        """Enqueue one RGB-to-RGBA copy without synchronizing the host."""
        shared_buffer = self._acquire_buffer()
        if shared_buffer is None:
            return False
        rgba = shared_buffer.rgba_tensor
        if rgba is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        if frame.source_event is not None:
            self._copy_stream.wait_event(frame.source_event)
        with torch.cuda.stream(self._copy_stream):
            rgb = frame.tensor
            if not rgb.is_contiguous():
                rgb = rgb.contiguous()
            rgba[..., :3].copy_(rgb, non_blocking=True)
            rgba[..., 3].fill_(255)
            rgb.record_stream(self._copy_stream)
            rgba.record_stream(self._copy_stream)
            done = torch.cuda.Event()
            done.record(self._copy_stream)
        shared_buffer.copy_done_event = done
        return True

    def ready_rgba_buffer(
        self,
    ) -> tuple[_SharedRGBABuffer, spy.NativeHandle] | None:
        """Return the next copy-complete buffer and CUDA stream handle."""
        for shared_buffer in self._buffers:
            event = shared_buffer.copy_done_event
            if event is None or not _cuda_event_ready(event):
                continue
            cuda_stream = self._spy.NativeHandle(
                self._spy.NativeHandleType.CUstream,
                int(self._copy_stream.cuda_stream),
            )
            return shared_buffer, cuda_stream
        return None

    def mark_submitted(self, shared_buffer: _SharedRGBABuffer, submit_id: int) -> None:
        """Associate a shared buffer with its Vulkan submission."""
        shared_buffer.copy_done_event = None
        shared_buffer.pending_submit_id = int(submit_id)

    def discard_ready(self, shared_buffer: _SharedRGBABuffer) -> None:
        """Release a copy-complete buffer when the swapchain is unavailable."""
        shared_buffer.copy_done_event = None
        shared_buffer.pending_submit_id = None

    def close(self) -> None:
        """Synchronize and release the CUDA copy stream."""
        self._copy_stream.synchronize()

    def _acquire_buffer(self) -> _SharedRGBABuffer | None:
        """Return a shared buffer no longer owned by CUDA or Vulkan."""
        for offset in range(len(self._buffers)):
            index = (self._next_buffer_index + offset) % len(self._buffers)
            shared_buffer = self._buffers[index]
            if shared_buffer.copy_done_event is not None:
                continue
            submit_id = shared_buffer.pending_submit_id
            if submit_id is not None and not self._device.is_submit_finished(submit_id):
                continue
            shared_buffer.pending_submit_id = None
            self._next_buffer_index = (index + 1) % len(self._buffers)
            return shared_buffer
        return None

    @staticmethod
    def _device_index(device: torch.device) -> int:
        """Return a concrete CUDA device index."""
        index = device.index
        return 0 if index is None else int(index)


@dataclass(frozen=True, slots=True)
class _CudaRGBFrame:
    """CUDA RGB tensor plus its producer-completion event."""

    tensor: Tensor
    """CUDA uint8 RGB tensor to copy."""

    source_event: torch.cuda.Event | None
    """Producer event the copy stream waits for."""


@dataclass(slots=True)
class _SharedRGBABuffer:
    """Track one shared buffer's CUDA-copy and Vulkan-submit ownership."""

    buffer: spy.Buffer
    """SlangPy shared buffer."""

    row_pitch: int
    """Bytes occupied by each row."""

    size_bytes: int
    """Total shared-buffer size in bytes."""

    rgba_tensor: Tensor | None = None
    """CUDA tensor mapped over ``buffer``."""

    copy_done_event: torch.cuda.Event | None = None
    """CUDA event completing the pending RGB-to-RGBA copy."""

    pending_submit_id: int | None = None
    """Vulkan submission currently reading the buffer."""


def _cuda_frame_device(
    frame: Tensor | _CudaPresentationFrame,
) -> torch.device | None:
    """Return the device of a wrapped CUDA tensor without synchronizing it."""
    tensor = frame.tensor if isinstance(frame, _CudaPresentationFrame) else frame
    if not tensor.is_cuda:
        return None
    return tensor.device


def _cuda_event_ready(event: torch.cuda.Event | None) -> bool:
    """Return whether a CUDA event has completed without blocking the host."""
    if event is None:
        return True
    try:
        return bool(event.query())
    except RuntimeError:
        return False


def _keyboard_event(event: spy.KeyboardEvent) -> KeyboardUserInputEvent | None:
    if event.is_key_press():
        state = KeyboardInputState.PRESSED
    elif event.is_key_release():
        state = KeyboardInputState.RELEASED
    else:
        return None
    key = _runtime_key_name(event.key)
    if not key:
        return None
    return KeyboardUserInputEvent(timestamp=uint64(0), key=key, state=state)


def _runtime_key_name(value: spy.KeyCode) -> str:
    """Return the browser-style key value used by runtime input events."""
    name = value.name
    normalized = name.lower()
    modifier = _MODIFIER_KEY_NAMES.get(normalized)
    if modifier is not None:
        return modifier
    printable = _PRINTABLE_KEY_NAMES.get(normalized)
    if printable is not None:
        return printable
    if len(normalized) == 4 and normalized.startswith("key"):
        digit = normalized[-1]
        if digit.isdigit():
            return digit
    return name


def _is_keyboard_input(event: spy.KeyboardEvent) -> bool:
    """Return whether SlangPy resolved the event to a text codepoint."""
    return event.is_input()


def _keyboard_input_text(event: spy.KeyboardEvent) -> str | None:
    """Return the Unicode character carried by a SlangPy input event."""
    try:
        codepoint = int(event.codepoint)
        if codepoint <= 0:
            return None
        return chr(codepoint)
    except (AttributeError, OverflowError, TypeError, ValueError):
        return None


def _mouse_event(
    event: spy.MouseEvent, *, width: int, height: int
) -> MouseUserInputEvent | None:
    x, y = float(event.pos.x), float(event.pos.y)
    normalized_x = min(1.0, max(0.0, x / width))
    normalized_y = min(1.0, max(0.0, y / height))
    if event.is_move():
        return MouseUserInputEvent(
            timestamp=uint64(0), x=normalized_x, y=normalized_y
        )
    if event.is_button_down() or event.is_button_up():
        button_name = event.button.name.lower()
        button = {"left": 0, "middle": 1, "right": 2}.get(button_name)
        if button is None:
            return None
        return MouseUserInputEvent(
            timestamp=uint64(0),
            action="button",
            x=normalized_x,
            y=normalized_y,
            button=button,
            pressed=event.is_button_down(),
        )
    if event.is_scroll():
        wheel_x, wheel_y = float(event.scroll.x), float(event.scroll.y)
        return MouseUserInputEvent(
            timestamp=uint64(0),
            action="wheel",
            x=normalized_x,
            y=normalized_y,
            wheel_x=wheel_x,
            wheel_y=wheel_y,
        )
    return None


__all__ = ["NativeWindowClientWindow"]
