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
from dataclasses import replace
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
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor

if TYPE_CHECKING:
    import slangpy as spy

_LOGGER = logging.getLogger(__name__)

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

_STANDARD_GAMEPAD_BUTTON_BITS: tuple[int | None, ...] = (
    1,  # A
    2,  # B
    3,  # X
    4,  # Y
    5,  # left bumper
    6,  # right bumper
    None,  # left trigger
    None,  # right trigger
    7,  # back
    8,  # start
    10,  # left stick
    11,  # right stick
    12,  # d-pad up
    14,  # d-pad down
    15,  # d-pad left
    13,  # d-pad right
    9,  # guide
)
"""SlangPy button bit indices in the browser standard-gamepad order."""


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
        self._pending_printable_keys: deque[tuple[str, KeyboardUserInputEvent]] = (
            deque()
        )
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
                on_gamepad_event=self._on_gamepad_event,
                on_gamepad_state=self._on_gamepad_state,
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
            pending_before_poll = tuple(self._pending_printable_keys)
            self._poll_input_events = []
            try:
                presenter.process_events()
                # ponytail: Text delayed by more than one poll can still arrive
                # as a second press. Split physical and text runtime events if a
                # client needs a longer coalescing window.
                pending_to_flush = (
                    tuple(self._pending_printable_keys)
                    if presenter.should_close
                    else pending_before_poll
                )
                for pending in pending_to_flush:
                    if pending in self._pending_printable_keys:
                        self._pending_printable_keys.remove(pending)
                        self._put_input(pending[1])
                if presenter.should_close:
                    self._on_window_closed()
                polled_input_events = self._poll_input_events
            finally:
                self._poll_input_events = None
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

        CUDA output remains GPU-resident. Conversion orders the result before
        the current consumer stream reads it.

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
            if self._close_event_enqueued:
                return
            if not presenter.present_frame(frame):
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
        self._input_events.put(replace(event, timestamp=uint64(elapsed_ns // 1_000)))

    def _on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if _is_keyboard_input(event):
            text = _keyboard_input_text(event)
            if text is not None:
                if self._pending_printable_keys:
                    physical_key, keyboard_event = self._pending_printable_keys.pop()
                    self._pressed_key_values[physical_key] = text
                    self._put_input(replace(keyboard_event, key=text))
            return

        keyboard_event = _keyboard_event(event)
        if keyboard_event is None:
            return
        if (
            keyboard_event.state is KeyboardInputState.PRESSED
            and len(keyboard_event.key) == 1
        ):
            self._pending_printable_keys.append((keyboard_event.key, keyboard_event))
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
                self._put_input(pending_key[1])
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

    def _on_gamepad_event(self, event: spy.GamepadEvent) -> None:
        gamepad_event = _gamepad_connection_event(event)
        if gamepad_event is not None:
            self._put_input(gamepad_event)

    def _on_gamepad_state(self, state: spy.GamepadState) -> None:
        self._put_input(_gamepad_state_event(state))

    def _on_window_closed(self) -> None:
        if self._close_event_enqueued:
            return
        self._close_event_enqueued = True
        self._put_input(CloseUserInputEvent(timestamp=uint64(0)))


class _SlangPyNativeWindowPresenter:
    """Own the SlangPy window and forward frames to its presentation context."""

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
        try:
            self._presentation = _PresentationContext(
                spy=spy,
                window=self._window,
                width=self._width,
                height=self._height,
            )
        except BaseException:
            self._window.close()
            raise
        self._keyboard_event_callback: Callable[[spy.KeyboardEvent], None] | None = None
        self._mouse_event_callback: Callable[[spy.MouseEvent], None] | None = None
        self._gamepad_event_callback: Callable[[spy.GamepadEvent], None] | None = None
        self._gamepad_state_callback: Callable[[spy.GamepadState], None] | None = None
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event
        self._window.on_gamepad_event = self._on_gamepad_event
        self._window.on_gamepad_state = self._on_gamepad_state

    def set_input_callbacks(
        self,
        *,
        on_keyboard_event: Callable[[spy.KeyboardEvent], None] | None = None,
        on_mouse_event: Callable[[spy.MouseEvent], None] | None = None,
        on_gamepad_event: Callable[[spy.GamepadEvent], None] | None = None,
        on_gamepad_state: Callable[[spy.GamepadState], None] | None = None,
    ) -> None:
        """Bind runtime input callbacks to the SlangPy window."""
        self._keyboard_event_callback = on_keyboard_event
        self._mouse_event_callback = on_mouse_event
        self._gamepad_event_callback = on_gamepad_event
        self._gamepad_state_callback = on_gamepad_state

    @property
    def should_close(self) -> bool:
        """Return whether the user or runtime requested window closure."""
        return self._closed or self._window.should_close()

    def process_events(self) -> None:
        """Pump pending events with SlangPy's standard window API."""
        if not self._closed:
            self._window.process_events()

    def present_frame(self, frame: Tensor) -> bool:
        """Present one RGB frame without pumping window events."""
        if self._closed:
            return False
        self._presentation.present(frame)
        return True

    def close(self) -> None:
        """Release presentation resources and the native window."""
        if self._closed:
            return
        self._closed = True
        window = self._window
        presentation = self._presentation

        # SlangPy keeps the assigned bound methods alive. Clear them before
        # releasing the presentation context so the window cannot retain this
        # presenter, its CUDA-mapped Torch tensor, and the backing SGL device
        # in a reference cycle until interpreter shutdown.
        window.on_keyboard_event = None
        window.on_mouse_event = None
        window.on_gamepad_event = None
        window.on_gamepad_state = None
        self._keyboard_event_callback = None
        self._mouse_event_callback = None
        self._gamepad_event_callback = None
        self._gamepad_state_callback = None
        self._presentation = cast(Any, None)

        try:
            presentation.close()
        finally:
            window.close()

    def _on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        """Forward one keyboard event to the runtime callback."""
        if self._keyboard_event_callback is not None:
            self._keyboard_event_callback(event)

    def _on_mouse_event(self, event: spy.MouseEvent) -> None:
        """Forward one mouse event to the runtime callback."""
        if self._mouse_event_callback is not None:
            self._mouse_event_callback(event)

    def _on_gamepad_event(self, event: spy.GamepadEvent) -> None:
        """Forward one gamepad lifecycle or button event."""
        if self._gamepad_event_callback is not None:
            self._gamepad_event_callback(event)

    def _on_gamepad_state(self, state: spy.GamepadState) -> None:
        """Forward one gamepad state snapshot."""
        if self._gamepad_state_callback is not None:
            self._gamepad_state_callback(state)


class _PresentationContext:
    """Own one render device and normalize every frame onto it."""

    def __init__(
        self, *, spy: Any, window: spy.Window, width: int, height: int
    ) -> None:
        self._spy = spy
        self._width = width
        self._height = height
        self._device = self._create_device()
        self._surface = self._device.create_surface(window)
        self._surface.configure(
            width=width,
            height=height,
            format=self._choose_surface_format(),
        )
        self._display_texture = self._device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=width,
            height=height,
            usage=(
                spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination
            ),
            label="flashdreams_v2_native_window_texture",
        )
        self._host_upload = np.empty((height, width, 4), dtype=np.uint8)
        self._host_upload[..., 3] = 255
        self._cuda_buffer: spy.Buffer | None = None
        self._cuda_rgba: Tensor | None = None
        self._render_device = torch.device("cpu")
        self._has_cuda_submission = False
        self._create_cuda_upload()

    def present(self, frame: Tensor) -> None:
        """Copy one RGB tensor to the render device and present it."""
        expected_shape = (self._height, self._width, 3)
        if frame.dtype != torch.uint8 or tuple(frame.shape) != expected_shape:
            raise ValueError(
                "Native-window frames must be uint8 RGB tensors with shape "
                f"{expected_shape}; got {tuple(frame.shape)} and {frame.dtype}."
            )
        if not self._surface.config:
            return
        surface_texture = self._surface.acquire_next_image()
        if not surface_texture:
            return

        frame = self._frame_on_render_device(frame)
        encoder = self._device.create_command_encoder()
        if self._cuda_rgba is not None:
            # CUDA frame, copy to CUDA buffer and then to texture
            self._copy_cuda_frame(frame, encoder)
        else:
            # Non-CUDA frame, copy to host and then to texture
            self._host_upload[..., :3] = np.ascontiguousarray(frame.numpy())
            self._display_texture.copy_from_numpy(self._host_upload)
        encoder.blit(surface_texture, self._display_texture)
        self._device.submit_command_buffer(encoder.finish())
        self._surface.present()
        if self._cuda_rgba is not None:
            self._has_cuda_submission = True
        del surface_texture

    def close(self) -> None:
        """Wait for pending presentation work before releasing resources."""
        device = self._device
        if device is None:
            return
        device.wait_for_idle()

        # The Torch tensor is an external-memory view owned by the SlangPy
        # buffer. Destroy the view first, followed by its backing resources and
        # finally the device. Letting the cycle reach Py_Finalize reverses this
        # relationship and can terminate inside sgl.dll on Windows.
        self._cuda_rgba = None
        self._cuda_buffer = None
        self._display_texture = cast(Any, None)
        self._surface = cast(Any, None)
        self._device = cast(Any, None)
        self._render_device = torch.device("cpu")
        self._has_cuda_submission = False

    def _frame_on_render_device(self, frame: Tensor) -> Tensor:
        """Return ``frame`` on the device owned by this context."""
        if frame.device == self._render_device:
            return frame
        return frame.to(self._render_device)

    def _copy_cuda_frame(self, frame: Tensor, encoder: Any) -> None:
        """Copy RGB into the CUDA-mapped RGBA buffer."""
        rgba = self._cuda_rgba
        buffer = self._cuda_buffer
        assert rgba is not None
        assert buffer is not None
        with torch.cuda.device(self._render_device):
            stream = torch.cuda.current_stream()
            stream_handle = int(stream.cuda_stream)
            if self._has_cuda_submission:
                self._device.sync_to_cuda(stream_handle)
            if not frame.is_contiguous():
                frame = frame.contiguous()
            rgba[..., :3].copy_(frame, non_blocking=True)
            rgba[..., 3].fill_(255)
            self._device.sync_to_device(stream_handle)
        encoder.copy_buffer_to_texture(
            self._display_texture,
            0,
            0,
            [0, 0, 0],
            buffer,
            0,
            self._width * self._height * 4,
            self._width * 4,
            [self._width, self._height, 1],
        )

    def _create_device(self) -> spy.Device:
        """Create a Vulkan device sharing the UI thread's CUDA context."""
        handles = self._cuda_context_handles()
        try:
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_interop=bool(handles),
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
                existing_device_handles=handles or None,
            )
        except RuntimeError as exc:
            _LOGGER.info(
                "Native-window CUDA interop unavailable; using host upload: %s",
                exc,
            )
            return self._spy.Device(
                type=self._spy.DeviceType.vulkan,
                enable_debug_layers=False,
                enable_cuda_launch_from_gfx=False,
                enable_ray_tracing=False,
            )

    def _cuda_context_handles(self) -> list[spy.NativeHandle]:
        """Return handles for the UI thread's current CUDA device."""
        if not torch.cuda.is_available():
            return []
        try:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(cuda_device):
                torch.cuda.current_stream()
                return list(self._spy.get_cuda_current_context_native_handles())
        except Exception as exc:
            _LOGGER.info(
                "Native-window CUDA context unavailable; using host upload: %s",
                exc,
            )
            return []

    def _create_cuda_upload(self) -> None:
        """Map one shared RGBA buffer into the presentation CUDA context."""
        if not self._device.supports_cuda_interop:
            return
        try:
            buffer = self._device.create_buffer(
                size=self._width * self._height * 4,
                usage=self._spy.BufferUsage.shared | self._spy.BufferUsage.copy_source,
                label="flashdreams_v2_native_cuda_rgba",
            )
            rgba = cast(
                Tensor,
                buffer.to_torch(
                    type=self._spy.DataType.uint8,
                    shape=[self._height, self._width, 4],
                ),
            )
        except Exception as exc:
            _LOGGER.info(
                "Native-window CUDA buffer unavailable; using host upload: %s",
                exc,
            )
            return
        self._cuda_buffer = buffer
        self._cuda_rgba = rgba
        self._render_device = rgba.device

    def _choose_surface_format(self) -> spy.Format:
        """Select a supported linear surface format."""
        supported = list(self._surface.info.formats)
        for candidate in (
            self._spy.Format.rgba8_unorm,
            self._spy.Format.bgra8_unorm,
            self._spy.Format.bgrx8_unorm,
        ):
            if candidate in supported:
                return candidate
        raise RuntimeError(
            "Native-window output requires a linear swapchain; "
            f"supported formats: {supported}."
        )


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
        return MouseUserInputEvent(timestamp=uint64(0), x=normalized_x, y=normalized_y)
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


def _gamepad_connection_event(
    event: spy.GamepadEvent,
) -> GamepadUserInputEvent | None:
    if event.is_connect():
        action = "connected"
    elif event.is_disconnect():
        action = "disconnected"
    else:
        return None
    return GamepadUserInputEvent(
        timestamp=uint64(0),
        action=action,
        index=0,
        mapping="standard",
    )


def _gamepad_state_event(state: spy.GamepadState) -> GamepadUserInputEvent:
    axes = tuple(
        _clamp(float(value), low=-1.0, high=1.0)
        for value in (state.left_x, state.left_y, state.right_x, state.right_y)
    )
    # SlangPy reports trigger axes in [-1, 1]; gamepad buttons use [0, 1].
    left_trigger, right_trigger = (
        (_clamp(float(value), low=-1.0, high=1.0) + 1.0) / 2.0
        for value in (state.left_trigger, state.right_trigger)
    )
    button_bits = int(state.buttons)
    pressed = tuple(
        (
            left_trigger > 0.0
            if index == 6
            else right_trigger > 0.0
            if index == 7
            else bit is not None and bool(button_bits & (1 << bit))
        )
        for index, bit in enumerate(_STANDARD_GAMEPAD_BUTTON_BITS)
    )
    buttons = tuple(
        (
            left_trigger
            if index == 6
            else right_trigger
            if index == 7
            else float(pressed[index])
        )
        for index in range(len(_STANDARD_GAMEPAD_BUTTON_BITS))
    )
    return GamepadUserInputEvent(
        timestamp=uint64(0),
        action="state",
        index=0,
        mapping="standard",
        axes=axes,
        buttons=buttons,
        pressed=pressed,
    )


def _clamp(value: float, *, low: float, high: float) -> float:
    return min(high, max(low, value))


__all__ = ["NativeWindowClientWindow"]
