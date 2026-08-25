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
from typing import Any

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

    ready_event: Any
    """CUDA event recorded after conversion to presentation format."""

    def to_cuda_tensor(self) -> Tensor:
        """Return the CUDA tensor consumed by the SlangPy presenter."""
        return self.tensor

    def to_cuda_event(self) -> Any:
        """Return the event the presenter's copy stream must wait for."""
        return self.ready_event


class NativeWindowClientWindow(IClientWindow):
    """Present UI output through a main-thread GLFW window.

    ``run_session`` owns threading. It calls :meth:`open`,
    :meth:`get_user_input_events`, and :meth:`close` on its I/O thread, while
    :meth:`write` runs on the runtime's presentation thread.
    """

    def __init__(
        self,
        *,
        title: str = "FlashDreams",
        presenter_factory: Callable[..., Any] | None = None,
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
        self._close_event_enqueued = threading.Event()
        self._presenter: Any | None = None
        self._cuda_device_index: int | None = None
        self._presentation_thread_id: int | None = None
        self._poll_input_events: list[UserInputEvent] | None = None
        self._poll_input_thread_id: int | None = None
        self._pending_printable_keys: deque[tuple[str, int]] = deque()
        self._pressed_key_values: dict[str, str] = {}

    def open(self, session_desc: SessionDesc) -> None:
        """Create the GLFW window on the runtime's I/O thread.

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
        presenter_factory = self._presenter_factory
        if presenter_factory is None:
            presenter_factory = _SlangPyNativeWindowPresenter

        presenter = presenter_factory(
            width=session_desc.video_width,
            height=session_desc.video_height,
            title=self.title,
        )
        try:
            set_callbacks = getattr(presenter, "set_input_callbacks", None)
            if not callable(set_callbacks):
                raise TypeError(
                    "A native-window presenter must accept input callbacks."
                )
            if not callable(getattr(presenter, "process_events", None)):
                raise TypeError(
                    "A native-window presenter must implement process_events()."
                )
            if not callable(getattr(presenter, "present_frame", None)):
                raise TypeError(
                    "A native-window presenter must implement present_frame()."
                )
            set_callbacks(
                on_keyboard_event=self._on_keyboard_event,
                on_mouse_event=self._on_mouse_event,
            )
        except BaseException:
            presenter.close()
            raise

        self._session_started_ns = self._clock_ns()
        self._session_desc = session_desc
        self._close_event_enqueued.clear()
        self._cuda_device_index = (
            torch.cuda.current_device() if torch.cuda.is_initialized() else None
        )
        self._presentation_thread_id = None
        self._poll_input_events = None
        self._poll_input_thread_id = None
        self._pending_printable_keys.clear()
        self._pressed_key_values.clear()
        self._presenter = presenter

    def get_user_input_events(self) -> UserInputEvents:
        """Pump GLFW and return native input events not yet read."""
        presenter = self._presenter
        if presenter is not None:
            self._poll_input_events = []
            self._poll_input_thread_id = threading.get_ident()
            try:
                presenter.process_events()
                if _presenter_should_close(presenter):
                    self._on_window_closed()
                polled_input_events = self._poll_input_events
            finally:
                self._poll_input_events = None
                self._poll_input_thread_id = None
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
        """Convert and submit one UI result from the presentation thread.

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
        if self._close_event_enqueued.is_set():
            return
        self._bind_cuda_device()
        frames = result_to_rgb24_tensor(result, self._session_desc_or_raise())
        for frame in frames:
            presentation_frame: object = frame
            if frame.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(frame.device))
                presentation_frame = _CudaPresentationFrame(frame, ready_event)
            if self._close_event_enqueued.is_set():
                return
            if not presenter.present_frame(presentation_frame):
                self._on_window_closed()
                return

    def close(self) -> None:
        """Release SlangPy and the GLFW window on the runtime's I/O thread."""
        presenter = self._presenter
        self._presenter = None
        self._session_started_ns = None
        self._session_desc = None
        self._cuda_device_index = None
        self._presentation_thread_id = None
        self._poll_input_events = None
        self._poll_input_thread_id = None
        self._pending_printable_keys.clear()
        self._pressed_key_values.clear()
        if presenter is not None:
            presenter.close()

    def _bind_cuda_device(self) -> None:
        """Bind the I/O thread's CUDA primary context to presentation."""
        thread_id = threading.get_ident()
        if self._presentation_thread_id == thread_id:
            return
        if self._cuda_device_index is not None:
            torch.cuda.set_device(self._cuda_device_index)
            torch.cuda.current_stream()
        self._presentation_thread_id = thread_id

    def _session_desc_or_raise(self) -> SessionDesc:
        session_desc = self._session_desc
        if session_desc is None:
            raise RuntimeError("Native window has no active session description.")
        return session_desc

    def _put_input(self, event: UserInputEvent) -> None:
        poll_input_events = self._current_poll_input_events()
        if poll_input_events is not None:
            poll_input_events.append(event)
            return
        started_ns = self._session_started_ns
        elapsed_ns = 0 if started_ns is None else max(0, self._clock_ns() - started_ns)
        self._input_events.put(
            replace(event, timestamp=uint64(elapsed_ns // 1_000))
        )

    def _current_poll_input_events(self) -> list[UserInputEvent] | None:
        """Return the input batch only from its event-polling thread."""
        if self._poll_input_thread_id == threading.get_ident():
            return self._poll_input_events
        return None

    def _on_keyboard_event(self, event: Any) -> None:
        if _is_keyboard_input(event):
            text = _keyboard_input_text(event)
            if text is not None:
                poll_input_events = self._current_poll_input_events()
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
            poll_input_events = self._current_poll_input_events()
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

    def _on_mouse_event(self, event: Any) -> None:
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
        if self._close_event_enqueued.is_set():
            return
        self._close_event_enqueued.set()
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
        self._keyboard_event_callback: Callable[[Any], None] | None = None
        self._mouse_event_callback: Callable[[Any], None] | None = None
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event
        self._cuda_interop_unavailable_reason: str | None = None
        self._device = self._create_device()
        self._surface = self._device.create_surface(self._window)
        self._surface.configure(
            width=self._width,
            height=self._height,
            format=self._choose_surface_format(),
        )
        self._display_texture = self._device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=self._width,
            height=self._height,
            usage=(
                spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access
                | spy.TextureUsage.copy_destination
            ),
            label="flashdreams_v2_native_window_texture",
        )
        self._cuda_rgb_interop = self._create_cuda_rgb_interop()
        self._host_upload = np.empty(
            (self._height, self._width, 4),
            dtype=np.uint8,
        )
        self._host_upload[..., 3] = 255

    def set_input_callbacks(
        self,
        *,
        on_keyboard_event: Callable[[Any], None] | None = None,
        on_mouse_event: Callable[[Any], None] | None = None,
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

    def present_frame(self, frame: object) -> bool:
        """Present one RGB frame without pumping window events."""
        if self._closed:
            return False

        cuda_resident = _is_cuda_resident(frame)
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

        rgb = self._as_host_rgb(frame)
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
        if self._cuda_rgb_interop is not None:
            self._cuda_rgb_interop.close()
            self._cuda_rgb_interop = None
        self._window.close()

    def _on_keyboard_event(self, event: Any) -> None:
        """Forward one keyboard event to the runtime callback."""
        if self._keyboard_event_callback is not None:
            self._keyboard_event_callback(event)

    def _on_mouse_event(self, event: Any) -> None:
        """Forward one mouse event to the runtime callback."""
        if self._mouse_event_callback is not None:
            self._mouse_event_callback(event)

    def _wait_for_presentation_progress(self) -> bool:
        """Yield briefly while waiting for CUDA or Vulkan progress."""
        if self._closed:
            return False
        time.sleep(_CUDA_COPY_POLL_SECONDS)
        return True

    def _create_device(self) -> Any:
        """Create the Vulkan device, sharing the active CUDA context if possible."""
        existing_device_handles = self._cuda_existing_device_handles()
        device_kwargs: dict[str, object] = {
            "type": self._spy.DeviceType.vulkan,
            "enable_debug_layers": False,
            "enable_cuda_interop": bool(existing_device_handles),
            "enable_cuda_launch_from_gfx": False,
            "enable_ray_tracing": False,
        }
        if existing_device_handles:
            device_kwargs["existing_device_handles"] = existing_device_handles
        try:
            device_factory: Any = self._spy.Device
            return device_factory(**device_kwargs)
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

    def _cuda_existing_device_handles(self) -> list[Any]:
        """Return SlangPy handles for the CUDA context current on this thread."""
        try:
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            torch.cuda.set_device(torch.cuda.current_device())
            torch.cuda.current_stream()
        except Exception:
            self._cuda_interop_unavailable_reason = "CUDA context unavailable"
            return []

        get_handles = getattr(
            self._spy,
            "get_cuda_current_context_native_handles",
            None,
        )
        if not callable(get_handles):
            self._cuda_interop_unavailable_reason = "native handles unavailable"
            return []
        try:
            return list(get_handles())
        except Exception as exc:
            self._cuda_interop_unavailable_reason = (
                f"native handle query failed ({type(exc).__name__}: {exc})"
            )
            return []

    def _create_cuda_rgb_interop(self) -> _CudaRGBInterop | None:
        """Create the GPU-resident RGB upload path when SlangPy supports it."""
        if not self._device.supports_cuda_interop:
            reason = self._cuda_interop_unavailable_reason or "unsupported"
            _LOGGER.info(
                "Native-window CUDA interop unavailable (%s); using host upload",
                reason,
            )
            return None
        try:
            interop = _CudaRGBInterop(
                spy=self._spy,
                device=self._device,
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

    def _choose_surface_format(self) -> Any:
        """Select a linear surface format for byte-exact RGB presentation."""
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
    def _as_host_rgb(frame: object) -> np.ndarray:
        """Return one CPU frame as contiguous uint8 RGB data."""
        to_numpy = getattr(frame, "to_numpy", None)
        if callable(to_numpy):
            frame = to_numpy()
        return np.ascontiguousarray(frame, dtype=np.uint8)


class _CudaRGBInterop:
    """Map triple-buffered SlangPy shared storage into CUDA tensors."""

    def __init__(self, *, spy: Any, device: Any, width: int, height: int) -> None:
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
            shared_buffer.rgba_tensor = shared_buffer.buffer.to_torch(
                type=spy.DataType.uint8,
                shape=[self._height, self._width, 4],
            )
        first_tensor = self._buffers[0].rgba_tensor
        if first_tensor is None:
            raise RuntimeError("Shared RGBA buffer was not mapped into CUDA.")
        self._cuda_device = first_tensor.device
        self._copy_stream = torch.cuda.Stream(device=self._cuda_device)
        self._next_buffer_index = 0

    def as_cuda_rgb_frame(self, frame: object) -> _CudaRGBFrame | None:
        """Return a device-compatible CUDA RGB view when available."""
        to_cuda_tensor = getattr(frame, "to_cuda_tensor", None)
        try:
            tensor = to_cuda_tensor() if callable(to_cuda_tensor) else frame
        except RuntimeError:
            return None
        if not torch.is_tensor(tensor):
            return None
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.uint8
            or tensor.ndim != 3
            or tuple(tensor.shape) != (self._height, self._width, 3)
        ):
            return None
        if self._device_index(tensor.device) != self._device_index(self._cuda_device):
            return None
        to_cuda_event = getattr(frame, "to_cuda_event", None)
        source_event = to_cuda_event() if callable(to_cuda_event) else None
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

    def ready_rgba_buffer(self) -> tuple[_SharedRGBABuffer, Any] | None:
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
    def _device_index(device: Any) -> int:
        """Return a concrete CUDA device index."""
        index = device.index
        return 0 if index is None else int(index)


@dataclass(frozen=True, slots=True)
class _CudaRGBFrame:
    """CUDA RGB tensor plus its producer-completion event."""

    tensor: Tensor
    """CUDA uint8 RGB tensor to copy."""

    source_event: Any | None
    """Producer event the copy stream waits for."""


@dataclass(slots=True)
class _SharedRGBABuffer:
    """Track one shared buffer's CUDA-copy and Vulkan-submit ownership."""

    buffer: Any
    """SlangPy shared buffer."""

    row_pitch: int
    """Bytes occupied by each row."""

    size_bytes: int
    """Total shared-buffer size in bytes."""

    rgba_tensor: Tensor | None = None
    """CUDA tensor mapped over ``buffer``."""

    copy_done_event: Any | None = None
    """CUDA event completing the pending RGB-to-RGBA copy."""

    pending_submit_id: int | None = None
    """Vulkan submission currently reading the buffer."""


def _is_cuda_resident(frame: object) -> bool:
    """Return whether ``frame`` wraps a CUDA tensor without synchronizing it."""
    to_cuda_tensor = getattr(frame, "to_cuda_tensor", None)
    try:
        tensor = to_cuda_tensor() if callable(to_cuda_tensor) else frame
    except RuntimeError:
        return False
    return bool(getattr(tensor, "is_cuda", False))


def _cuda_event_ready(event: Any | None) -> bool:
    """Return whether a CUDA event has completed without blocking the host."""
    if event is None:
        return True
    try:
        return bool(event.query())
    except RuntimeError:
        return False


def _keyboard_event(event: Any) -> KeyboardUserInputEvent | None:
    is_press = getattr(event, "is_key_press", None)
    is_release = getattr(event, "is_key_release", None)
    if callable(is_press) and is_press():
        state = KeyboardInputState.PRESSED
    elif callable(is_release) and is_release():
        state = KeyboardInputState.RELEASED
    else:
        return None
    key = _runtime_key_name(getattr(event, "key", ""))
    if not key:
        return None
    return KeyboardUserInputEvent(timestamp=uint64(0), key=key, state=state)


def _runtime_key_name(value: Any) -> str:
    """Return the browser-style key value used by runtime input events."""
    name = _enum_name(value)
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


def _is_keyboard_input(event: Any) -> bool:
    """Return whether SlangPy resolved the event to a text codepoint."""
    is_input = getattr(event, "is_input", None)
    return bool(callable(is_input) and is_input())


def _keyboard_input_text(event: Any) -> str | None:
    """Return the Unicode character carried by a SlangPy input event."""
    try:
        codepoint = int(event.codepoint)
        if codepoint <= 0:
            return None
        return chr(codepoint)
    except (AttributeError, OverflowError, TypeError, ValueError):
        return None


def _mouse_event(
    event: Any, *, width: int, height: int
) -> MouseUserInputEvent | None:
    event_type = _enum_name(getattr(event, "type", "")).lower()
    x, y = _pair(getattr(event, "pos", (0.0, 0.0)))
    normalized_x = min(1.0, max(0.0, x / width))
    normalized_y = min(1.0, max(0.0, y / height))
    if event_type == "move":
        return MouseUserInputEvent(
            timestamp=uint64(0), x=normalized_x, y=normalized_y
        )
    if event_type in {"button_down", "button_up"}:
        button_name = _enum_name(getattr(event, "button", "")).lower()
        button = {"left": 0, "middle": 1, "right": 2}.get(button_name)
        if button is None:
            return None
        return MouseUserInputEvent(
            timestamp=uint64(0),
            action="button",
            x=normalized_x,
            y=normalized_y,
            button=button,
            pressed=event_type == "button_down",
        )
    if event_type == "scroll":
        wheel_x, wheel_y = _pair(getattr(event, "scroll", (0.0, 0.0)))
        return MouseUserInputEvent(
            timestamp=uint64(0),
            action="wheel",
            x=normalized_x,
            y=normalized_y,
            wheel_x=wheel_x,
            wheel_y=wheel_y,
        )
    return None


def _pair(value: Any) -> tuple[float, float]:
    try:
        return float(value.x), float(value.y)
    except AttributeError:
        return float(value[0]), float(value[1])


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value).rsplit(".", maxsplit=1)[-1]


def _presenter_should_close(presenter: Any) -> bool:
    value = getattr(presenter, "should_close", False)
    return bool(value() if callable(value) else value)


__all__ = ["NativeWindowClientWindow"]
