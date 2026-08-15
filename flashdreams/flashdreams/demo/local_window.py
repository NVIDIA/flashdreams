# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SlangPy local-window event bridging and CUDA/Vulkan presentation."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from flashdreams.demo.local_input import SlangPyLocalInputHandler


@dataclass(slots=True)
class LocalWindowInputBridge:
    """Share one SlangPy presenter between local input and output objects."""

    _handler: SlangPyLocalInputHandler | None = None
    """Input handler receiving callbacks from the active presenter."""

    _presenter: Any | None = None
    """Presenter whose SDL event queue is pumped at input sample boundaries."""

    _events_pumped_in_background: bool = False
    """Whether the presenter owns a background event-pump thread."""

    def bind_handler(self, handler: SlangPyLocalInputHandler) -> None:
        """Bind the handler created for the current application run."""
        self._handler = handler

    def bind_presenter(self, presenter: Any) -> None:
        """Bind presenter event callbacks to the current input handler."""
        self._presenter = presenter
        self._events_pumped_in_background = False
        self._bind_callbacks(presenter)

    def bind_background_presenter(self, presenter: Any) -> None:
        """Bind callbacks without pumping events from the application thread."""
        self._presenter = presenter
        self._events_pumped_in_background = True
        self._bind_callbacks(presenter)

    def _bind_callbacks(self, presenter: Any) -> None:
        handler = self._handler
        if handler is None or not handler.accepts_window_events:
            return
        set_callbacks = getattr(presenter, "set_input_callbacks", None)
        if not callable(set_callbacks):
            raise TypeError(
                "Local-window presenters must implement set_input_callbacks() "
                "when the application declares live inputs."
            )
        set_callbacks(
            on_keyboard_event=handler.on_keyboard_event,
            on_gamepad_event=handler.on_gamepad_event,
            on_gamepad_state=handler.on_gamepad_state,
        )

    def process_events(self) -> None:
        """Pump pending SDL events through the active presenter."""
        if self._presenter is None or self._events_pumped_in_background:
            return
        process_events = getattr(self._presenter, "process_events", None)
        if callable(process_events):
            process_events()


class SlangPyLocalWindowPresenter:
    """Present CUDA-backed RGB frames through one Vulkan swapchain."""

    def __init__(self, *, width: int, height: int, title: str) -> None:
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
        self._window = spy.Window(
            width=self._width,
            height=self._height,
            title=title,
            resizable=False,
        )
        self._keyboard_event_callback: Callable[[Any], None] | None = None
        self._mouse_event_callback: Callable[[Any], None] | None = None
        self._gamepad_event_callback: Callable[[Any], None] | None = None
        self._gamepad_state_callback: Callable[[Any], None] | None = None
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event
        self._window.on_gamepad_event = self._on_gamepad_event
        self._window.on_gamepad_state = self._on_gamepad_state
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
            label="flashdreams_local_window_texture",
        )
        self._cuda_rgb_interop = self._create_cuda_rgb_interop()
        self._host_upload = np.empty(
            (self._height, self._width, 4),
            dtype=np.uint8,
        )
        self._host_upload[..., 3] = 255
        self._closed = False

    def set_input_callbacks(
        self,
        *,
        on_keyboard_event: Callable[[Any], None] | None = None,
        on_mouse_event: Callable[[Any], None] | None = None,
        on_gamepad_event: Callable[[Any], None] | None = None,
        on_gamepad_state: Callable[[Any], None] | None = None,
    ) -> None:
        """Bind local input callbacks to the SDL-backed window."""
        self._keyboard_event_callback = on_keyboard_event
        self._mouse_event_callback = on_mouse_event
        self._gamepad_event_callback = on_gamepad_event
        self._gamepad_state_callback = on_gamepad_state

    @property
    def should_close(self) -> bool:
        """Return whether the user requested window closure."""
        return self._closed or self._window.should_close()

    def present(self, frame: object) -> bool:
        """Present one ordered RGB frame and return whether the window remains open."""
        self.process_events()
        if self.should_close:
            return False

        if self._cuda_rgb_interop is not None:
            cuda_frame = self._cuda_rgb_interop.as_cuda_rgb_frame(frame)
            if cuda_frame is not None:
                while not self._cuda_rgb_interop.enqueue_rgb_to_shared_rgba(cuda_frame):
                    if not self._wait_for_progress():
                        return False
                while not self._submit_ready_cuda_rgb():
                    if not self._wait_for_progress():
                        return False
                return True

        rgb = self._as_host_rgb(frame)
        if tuple(rgb.shape) != (self._height, self._width, 3):
            raise ValueError(
                "Native-window frame shape does not match the configured surface: "
                f"{tuple(rgb.shape)} != {(self._height, self._width, 3)}."
            )
        self._host_upload[..., :3] = rgb
        self._present_host_upload()
        return True

    def wait_until(self, deadline_s: float) -> bool:
        """Process events while pacing presentation to an absolute deadline."""
        while True:
            self.process_events()
            if self.should_close:
                return False
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0:
                return True
            time.sleep(min(remaining_s, 0.001))

    def process_events(self) -> None:
        """Process pending local window events."""
        if not self._closed:
            self._window.process_events()

    def close(self) -> None:
        """Release CUDA interoperability and local window resources."""
        if self._closed:
            return
        self._closed = True
        if self._cuda_rgb_interop is not None:
            self._cuda_rgb_interop.close()
            self._cuda_rgb_interop = None
        self._window.close()

    def _on_keyboard_event(self, event: Any) -> None:
        is_press = getattr(event, "is_key_press", None)
        key_name = getattr(getattr(event, "key", None), "name", "")
        if callable(is_press) and is_press() and key_name == "escape":
            self._closed = True
        if self._keyboard_event_callback is not None:
            self._keyboard_event_callback(event)

    def _on_mouse_event(self, event: Any) -> None:
        if self._mouse_event_callback is not None:
            self._mouse_event_callback(event)

    def _on_gamepad_event(self, event: Any) -> None:
        if self._gamepad_event_callback is not None:
            self._gamepad_event_callback(event)

    def _on_gamepad_state(self, state: Any) -> None:
        if self._gamepad_state_callback is not None:
            self._gamepad_state_callback(state)

    def _wait_for_progress(self) -> bool:
        self.process_events()
        if self.should_close:
            return False
        time.sleep(0.0002)
        return True

    def _create_device(self) -> Any:
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
            logger.info(
                "[local-window] CUDA interop device creation failed; "
                "using Vulkan host upload ({})",
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
        try:
            import torch

            if not torch.cuda.is_initialized():
                torch.cuda.init()
            # CUDA contexts are thread-local. Model initialization can leave
            # PyTorch's primary context alive but no longer current on this
            # thread; current_stream() alone does not restore that binding.
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
        if not self._device.supports_cuda_interop:
            reason = self._cuda_interop_unavailable_reason or "unsupported"
            logger.info(
                "[local-window] cuda_interop={}; using Vulkan host upload",
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
            logger.info(
                "[local-window] cuda_interop=unavailable; "
                "using Vulkan host upload ({})",
                exc,
            )
            return None
        logger.info("[local-window] cuda_interop=enabled")
        return interop

    def _submit_ready_cuda_rgb(self) -> bool:
        assert self._cuda_rgb_interop is not None
        ready = self._cuda_rgb_interop.ready_rgba_buffer()
        if ready is None or not self._surface.config:
            return False
        rgba_buffer, cuda_stream = ready
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            return False

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
        to_numpy = getattr(frame, "to_numpy", None)
        if callable(to_numpy):
            frame = to_numpy()
        return np.ascontiguousarray(frame, dtype=np.uint8)


class _CudaRGBInterop:
    """Map triple-buffered SlangPy shared storage into CUDA tensors."""

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
                    label=f"flashdreams_native_cuda_rgba_{index}",
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
        if not self._torch.is_tensor(tensor):
            return None
        if (
            not tensor.is_cuda
            or tensor.dtype != self._torch.uint8
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
        with self._torch.cuda.stream(self._copy_stream):
            rgb = frame.tensor
            if not rgb.is_contiguous():
                rgb = rgb.contiguous()
            rgba[..., :3].copy_(rgb, non_blocking=True)
            rgba[..., 3].fill_(255)
            rgb.record_stream(self._copy_stream)
            rgba.record_stream(self._copy_stream)
            done = self._torch.cuda.Event()
            done.record(self._copy_stream)
        shared_buffer.copy_done_event = done
        return True

    def ready_rgba_buffer(self) -> tuple[_SharedRGBABuffer, Any] | None:
        """Return the next copy-complete shared buffer and CUDA stream handle."""
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

    def close(self) -> None:
        """Synchronize and release the copy stream."""
        self._copy_stream.synchronize()

    def _acquire_buffer(self) -> _SharedRGBABuffer | None:
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
        index = device.index
        return 0 if index is None else int(index)


class _CudaRGBFrame:
    """CUDA RGB tensor plus its producer-completion event."""

    def __init__(self, *, tensor: Any, source_event: Any | None) -> None:
        self.tensor = tensor
        self.source_event = source_event


class _SharedRGBABuffer:
    """SlangPy shared buffer with CUDA-copy and Vulkan-submit ownership."""

    def __init__(self, *, buffer: Any, row_pitch: int, size_bytes: int) -> None:
        self.buffer = buffer
        self.row_pitch = row_pitch
        self.size_bytes = size_bytes
        self.rgba_tensor: Any | None = None
        self.copy_done_event: Any | None = None
        self.pending_submit_id: int | None = None


def _cuda_event_ready(event: Any | None) -> bool:
    if event is None:
        return True
    try:
        return bool(event.query())
    except RuntimeError:
        return False


@dataclass(frozen=True, slots=True)
class _QueuedLocalWindowResult:
    """One generation-tagged result awaiting local presentation."""

    generation: int
    """Generation that produced the result."""

    frames: Sequence[object]
    """Lazy frames prepared on the model execution thread."""

    frame_count: int
    """Number of prepared frames in the queued chunk."""


class _LocalWindowPresentationWorker:
    """Own one presenter and its event loop on a dedicated thread."""

    def __init__(
        self,
        *,
        presenter_factory: Callable[..., Any],
        presenter_kwargs: Mapping[str, object],
        presenter_opened: Callable[[Any], None] | None,
        fps: float,
        max_pending_chunks: int,
    ) -> None:
        self._presenter_factory = presenter_factory
        self._presenter_kwargs = presenter_kwargs
        self._presenter_opened = presenter_opened
        self._frame_interval_s = 1.0 / fps
        self._fps = fps
        self._pending: queue.Queue[_QueuedLocalWindowResult] = queue.Queue(
            maxsize=max_pending_chunks
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._window_closed = threading.Event()
        self._state_lock = threading.Lock()
        self._generation = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="flashdreams-local-window",
            daemon=True,
        )

    @property
    def should_stop(self) -> bool:
        """Return whether the window closed or its worker failed."""
        with self._state_lock:
            failed = self._error is not None
        return self._window_closed.is_set() or failed

    def start(self) -> None:
        """Start the presenter thread and wait for window initialization."""
        self._thread.start()
        self._ready.wait()
        self.raise_if_failed()

    def begin_generation(self, generation: int) -> None:
        """Advance the generation and discard queued stale results."""
        with self._state_lock:
            self._generation = generation
        self._discard_pending()

    def submit(
        self,
        frames: Sequence[object],
        *,
        frame_count: int,
        generation: int,
    ) -> tuple[int, float]:
        """Queue prepared frames and return replacements plus queued duration."""
        self.raise_if_failed()
        if self.should_stop:
            return 0, 0.0
        item = _QueuedLocalWindowResult(
            generation=generation,
            frames=frames,
            frame_count=frame_count,
        )
        replaced = 0
        while True:
            try:
                self._pending.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._pending.get_nowait()
                except queue.Empty:
                    continue
                replaced += 1
        queued_frames = self._pending.qsize() * max(0, frame_count)
        return replaced, queued_frames / self._fps

    def close(self) -> None:
        """Stop the worker and release the presenter on its owning thread."""
        self._stop.set()
        self._discard_pending()
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Raise a presentation failure on the application thread."""
        with self._state_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Local-window presentation thread failed.") from error

    def _run(self) -> None:
        presenter: Any | None = None
        try:
            presenter = self._presenter_factory(**self._presenter_kwargs)
            if self._presenter_opened is not None:
                self._presenter_opened(presenter)
            self._ready.set()
            next_deadline_s: float | None = None
            active_generation = self._generation
            while not self._stop.is_set():
                presenter.process_events()
                if _presenter_should_close(presenter):
                    self._window_closed.set()
                    break
                try:
                    item = self._pending.get(timeout=0.001)
                except queue.Empty:
                    continue
                if self._is_stale(item.generation):
                    continue
                if item.generation != active_generation:
                    active_generation = item.generation
                    next_deadline_s = None
                for frame in item.frames:
                    if self._is_stale(item.generation):
                        break
                    if not presenter.present(frame):
                        self._window_closed.set()
                        break
                    now_s = time.monotonic()
                    if next_deadline_s is None:
                        next_deadline_s = now_s + self._frame_interval_s
                    else:
                        next_deadline_s = max(
                            now_s,
                            next_deadline_s + self._frame_interval_s,
                        )
                    if not presenter.wait_until(next_deadline_s):
                        self._window_closed.set()
                        break
                if self._window_closed.is_set():
                    break
        except BaseException as exc:
            with self._state_lock:
                self._error = exc
        finally:
            self._ready.set()
            self._stop.set()
            if presenter is not None:
                try:
                    presenter.close()
                except BaseException as exc:
                    with self._state_lock:
                        if self._error is None:
                            self._error = exc

    def _is_stale(self, generation: int) -> bool:
        if self._stop.is_set():
            return True
        with self._state_lock:
            return generation != self._generation

    def _discard_pending(self) -> None:
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                return


def _presenter_should_close(presenter: Any) -> bool:
    value = getattr(presenter, "should_close", False)
    return bool(value() if callable(value) else value)


__all__ = ["LocalWindowInputBridge", "SlangPyLocalWindowPresenter"]
