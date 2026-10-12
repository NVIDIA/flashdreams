# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlangPy native client window with GPU-resident presentation."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import queue
import sys
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
    FullscreenDisabledOptimalSizeUserInputEvent,
    FullscreenEnabledOptimalSizeUserInputEvent,
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


class _X11ClientMessage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", ctypes.c_long * 5),
        ("padding", ctypes.c_long * 12),
    ]


def _x11_property(xlib: Any, display: Any, window: int, name: str) -> list[int]:
    """Read an EWMH property containing 32-bit integers or atoms."""
    actual_type = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    count = ctypes.c_ulong()
    remaining = ctypes.c_ulong()
    data = ctypes.POINTER(ctypes.c_ubyte)()
    xlib.XGetWindowProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    atom = xlib.XInternAtom(display, name.encode(), False)
    status = xlib.XGetWindowProperty(
        display,
        window,
        atom,
        0,
        1024,
        False,
        0,
        ctypes.byref(actual_type),
        ctypes.byref(actual_format),
        ctypes.byref(count),
        ctypes.byref(remaining),
        ctypes.byref(data),
    )
    try:
        if status != 0 or actual_format.value != 32 or not data:
            return []
        values = ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))
        return [int(values[index]) for index in range(count.value)]
    finally:
        if data:
            xlib.XFree.argtypes = [ctypes.c_void_p]
            xlib.XFree(data)


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
        self._window_size: tuple[int, int] | None = None
        self._input_events: queue.SimpleQueue[UserInputEvent] = queue.SimpleQueue()
        self._close_event_enqueued = False
        self._presenter: _SlangPyNativeWindowPresenter | None = None
        self._poll_input_events: list[UserInputEvent] | None = None
        self._pending_printable_keys: deque[tuple[str, KeyboardUserInputEvent]] = (
            deque()
        )
        self._pressed_key_values: dict[str, str] = {}
        self._hide_cursor = False
        self._lock_cursor_to_window = False
        self._fullscreen_request: bool | None = None

    def request_fullscreen(self, is_fullscreen_enabled: bool) -> None:
        """Request a native mode change and acknowledge it on a later input poll."""
        presenter = self._presenter
        if presenter is not None and presenter.request_fullscreen(
            is_fullscreen_enabled
        ):
            self._fullscreen_request = is_fullscreen_enabled
        else:
            self._fullscreen_request = None

    def request_hide_cursor(self, hide_cursor: bool) -> None:
        """Show or hide the cursor in the native window."""
        presenter = self._presenter
        if presenter is not None:
            presenter.configure_cursor(
                hide_cursor=hide_cursor,
                lock_cursor_to_window=self._lock_cursor_to_window,
            )
        self._hide_cursor = hide_cursor

    def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None:
        """Release or capture pointer motion in the native window."""
        super().request_lock_cursor_to_window(lock_cursor_to_window)
        presenter = self._presenter
        if presenter is not None:
            presenter.configure_cursor(
                hide_cursor=self._hide_cursor,
                lock_cursor_to_window=lock_cursor_to_window,
            )
        self._lock_cursor_to_window = lock_cursor_to_window

    def request_new_window_size(self, new_window_size: tuple[int, int]) -> None:
        """Resize the open native window without replacing it.

        Args:
            new_window_size: Requested ``(width, height)`` in pixels.

        Raises:
            RuntimeError: The window is not open or cannot be resized.
        """
        presenter = self._presenter
        if presenter is None:
            raise RuntimeError(
                "NativeWindowClientWindow.open() must run before resizing."
            )
        self._window_size = presenter.resize(*new_window_size)

    def open(self, session_desc: SessionDesc) -> None:
        """Open a session in the GLFW window on the runtime's UI thread.

        Replacement sessions reuse the existing presenter and clear input
        buffered for the completed session.

        Args:
            session_desc: Resolved output dimensions and tensor layout.

        Raises:
            RuntimeError: Window initialization fails.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "NativeWindowClientWindow.open() must run on the process main thread for event polling."
            )
        presenter = self._presenter
        if presenter is None:
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
                presenter.configure_cursor(
                    hide_cursor=self._hide_cursor,
                    lock_cursor_to_window=self._lock_cursor_to_window,
                )
            except BaseException:
                presenter.close()
                raise

        self._session_started_ns = self._clock_ns()
        self._session_desc = session_desc
        self._window_size = presenter.size
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
                self._window_size = presenter.size
                if self._fullscreen_request is not None:
                    size = presenter.fullscreen_size(self._fullscreen_request)
                    if size is not None:
                        event_type = (
                            FullscreenEnabledOptimalSizeUserInputEvent
                            if self._fullscreen_request
                            else FullscreenDisabledOptimalSizeUserInputEvent
                        )
                        self._put_input(
                            event_type(
                                timestamp=uint64(0), width=size[0], height=size[1]
                            )
                        )
                        self._fullscreen_request = None
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
        window_size = self._window_size
        if window_size is None:
            raise RuntimeError("Native window has no active presentation size.")
        frames = result_to_rgb24_tensor(
            result,
            self._session_desc_or_raise(),
            window_size,
        )
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
        self._window_size = None
        self._fullscreen_request = None
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
        window_size = self._window_size
        if window_size is None:
            return
        mouse_event = _mouse_event(
            event,
            width=window_size[0],
            height=window_size[1],
            lock_cursor_to_window=self._lock_cursor_to_window,
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

    def __init__(
        self,
        *,
        width: int,
        height: int,
        title: str,
    ) -> None:
        """Create a programmatically resizable SlangPy window and Vulkan surface.

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
        self._windowed_placement: tuple[Any, int] | None = None
        self._fullscreen_enabled = False
        self._optimal_size: tuple[int, int] | None = None
        self._window = spy.Window(
            width=self._width,
            height=self._height,
            title=title,
            resizable=False,
        )
        self._native_hwnd = None
        self._xdisplay: Any = None
        self._xwindow: int | None = None
        if sys.platform == "win32":
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetActiveWindow.restype = ctypes.c_void_p
            self._native_hwnd = user32.GetActiveWindow()
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
        self._width, self._height = self._presentation.size
        self._pending_resize: tuple[int, int] | None = None
        self._keyboard_event_callback: Callable[[spy.KeyboardEvent], None] | None = None
        self._mouse_event_callback: Callable[[spy.MouseEvent], None] | None = None
        self._gamepad_event_callback: Callable[[spy.GamepadEvent], None] | None = None
        self._gamepad_state_callback: Callable[[spy.GamepadState], None] | None = None
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event
        self._window.on_gamepad_event = self._on_gamepad_event
        self._window.on_gamepad_state = self._on_gamepad_state
        self._window.on_resize = self._on_resize

    @property
    def size(self) -> tuple[int, int]:
        """Return the current presentation-surface size in pixels."""
        return self._width, self._height

    def configure_cursor(
        self, *, hide_cursor: bool, lock_cursor_to_window: bool
    ) -> None:
        """Change the native cursor mode while the window is running."""
        if lock_cursor_to_window:
            cursor_mode = self._spy.CursorMode.disabled
        elif hide_cursor:
            cursor_mode = self._spy.CursorMode.hidden
        else:
            cursor_mode = self._spy.CursorMode.normal
        self._window.cursor_mode = cursor_mode

    def resize(self, width: int, height: int) -> tuple[int, int]:
        """Resize a windowed surface, preserving fullscreen monitor dimensions."""
        if self._closed:
            raise RuntimeError("Cannot resize a closed native window.")
        if getattr(self, "_fullscreen_enabled", False):
            return self.size
        self._window.resize(width, height)
        return self.size

    def request_fullscreen(self, enabled: bool) -> bool:
        """Ask the native window system to switch fullscreen mode."""
        if self._closed:
            return False
        if sys.platform == "win32":
            return self._request_fullscreen_windows(enabled)
        if sys.platform.startswith("linux"):
            return self._request_fullscreen_x11(enabled)
        return False

    def fullscreen_size(self, enabled: bool) -> tuple[int, int] | None:
        """Return usable dimensions only after the requested mode is accepted."""
        if sys.platform.startswith("linux"):
            display, window, xlib = self._x11_window()
            states = _x11_property(xlib, display, window, "_NET_WM_STATE")
            atom = xlib.XInternAtom(display, b"_NET_WM_STATE_FULLSCREEN", False)
            if (atom in states) != enabled:
                return None
            self._fullscreen_enabled = enabled
            if enabled:
                return self.size
            root = xlib.XDefaultRootWindow(display)
            work = _x11_property(xlib, display, root, "_NET_WORKAREA")
            desktop = _x11_property(xlib, display, root, "_NET_CURRENT_DESKTOP")
            offset = 4 * (desktop[0] if desktop else 0)
            if len(work) < offset + 4:
                return None
            # ponytail: EWMH work area spans the desktop on multi-monitor X11;
            # use RandR monitor work areas if per-monitor windowed limits are needed.
            extents = _x11_property(xlib, display, window, "_NET_FRAME_EXTENTS")
            if len(extents) != 4:
                extents = [0, 0, 0, 0]
            return (
                work[offset + 2] - extents[0] - extents[1],
                work[offset + 3] - extents[2] - extents[3],
            )
        elif self._fullscreen_enabled != enabled:
            return None
        return self._optimal_size

    def _request_fullscreen_windows(self, enabled: bool) -> bool:
        """Switch borderless fullscreen using the window's current monitor."""
        from ctypes import wintypes

        class MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("monitor", wintypes.RECT),
                ("work", wintypes.RECT),
                ("flags", wintypes.DWORD),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = self._native_hwnd
        if hwnd is None:
            return False
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.MonitorFromWindow.restype = wintypes.HANDLE
        user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        get_style = (
            user32.GetWindowLongPtrW
            if ctypes.sizeof(ctypes.c_void_p) == 8
            else user32.GetWindowLongW
        )
        set_style = (
            user32.SetWindowLongPtrW
            if ctypes.sizeof(ctypes.c_void_p) == 8
            else user32.SetWindowLongW
        )
        get_style.argtypes = [wintypes.HWND, ctypes.c_int]
        get_style.restype = ctypes.c_ssize_t
        set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_style.restype = ctypes.c_ssize_t
        info = MonitorInfo()
        info.size = ctypes.sizeof(info)
        monitor = user32.MonitorFromWindow(hwnd, 2)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return False
        style = get_style(hwnd, -16)
        current_rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(current_rect)):
            return False
        if enabled and self._windowed_placement is None:
            self._windowed_placement = current_rect, style
        target_style = (
            style & ~0x00CF0000
            if enabled
            else (self._windowed_placement[1] if self._windowed_placement else style)
        )
        rect = (
            info.monitor
            if enabled
            else (
                self._windowed_placement[0]
                if self._windowed_placement
                else current_rect
            )
        )
        ctypes.set_last_error(0)
        set_style(hwnd, -16, target_style)
        if ctypes.get_last_error():
            return False
        if not user32.SetWindowPos(
            hwnd,
            None,
            rect.left,
            rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top,
            0x0034,
        ):
            set_style(hwnd, -16, style)
            return False
        usable = info.monitor if enabled else info.work
        if not enabled:
            # Convert the work area to client dimensions, excluding restored chrome.
            user32.AdjustWindowRectEx.argtypes = [
                ctypes.POINTER(wintypes.RECT),
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            chrome = wintypes.RECT()
            if not user32.AdjustWindowRectEx(
                ctypes.byref(chrome), target_style, False, get_style(hwnd, -20)
            ):
                return False
            self._optimal_size = (
                usable.right - usable.left - (chrome.right - chrome.left),
                usable.bottom - usable.top - (chrome.bottom - chrome.top),
            )
            self._windowed_placement = None
        else:
            self._optimal_size = usable.right - usable.left, usable.bottom - usable.top
        self._fullscreen_enabled = enabled
        return True

    def _x11_window(self) -> tuple[Any, int, Any]:
        """Return SlangPy's X11 display, window, and the system Xlib."""
        xlib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        xlib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        xlib.XInternAtom.restype = ctypes.c_ulong
        xlib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        xlib.XDefaultRootWindow.restype = ctypes.c_ulong
        if self._xdisplay is None:
            xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
            xlib.XOpenDisplay.restype = ctypes.c_void_p
            self._xdisplay = xlib.XOpenDisplay(None)
        display = self._xdisplay
        if not display:
            raise RuntimeError("Cannot open the native X11 display.")
        if self._xwindow is None:
            root = xlib.XDefaultRootWindow(display)
            xlib.XFetchName.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            xlib.XFree.argtypes = [ctypes.c_void_p]
            for candidate in reversed(
                _x11_property(xlib, display, root, "_NET_CLIENT_LIST")
            ):
                if _x11_property(xlib, display, candidate, "_NET_WM_PID") != [
                    os.getpid()
                ]:
                    continue
                name = ctypes.c_char_p()
                try:
                    if (
                        xlib.XFetchName(display, candidate, ctypes.byref(name))
                        and name.value == self._window.title.encode()
                    ):
                        self._xwindow = candidate
                        break
                finally:
                    if name:
                        xlib.XFree(name)
        if self._xwindow is None:
            raise RuntimeError("The native window has not been registered with X11.")
        return display, self._xwindow, xlib

    def _request_fullscreen_x11(self, enabled: bool) -> bool:
        """Send an EWMH fullscreen request without waiting for the window manager."""
        try:
            display, window, xlib = self._x11_window()
        except (OSError, RuntimeError):
            return False
        root = xlib.XDefaultRootWindow(display)
        supported = _x11_property(xlib, display, root, "_NET_SUPPORTED")
        fullscreen = xlib.XInternAtom(display, b"_NET_WM_STATE_FULLSCREEN", False)
        if fullscreen not in supported:
            return False
        event = _X11ClientMessage()
        event.type = 33
        event.display = display
        event.window = window
        event.message_type = xlib.XInternAtom(display, b"_NET_WM_STATE", False)
        event.format = 32
        event.data[0], event.data[1], event.data[3] = int(enabled), fullscreen, 1
        xlib.XSendEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_long,
            ctypes.c_void_p,
        ]
        if not xlib.XSendEvent(
            display, root, False, (1 << 19) | (1 << 20), ctypes.byref(event)
        ):
            return False
        xlib.XFlush.argtypes = [ctypes.c_void_p]
        xlib.XFlush(display)
        return True

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
        """Pump events and apply the latest acknowledged native resize."""
        if not self._closed:
            self._window.process_events()
            pending_resize = self._pending_resize
            if pending_resize is not None:
                self._pending_resize = None
                self._width, self._height = self._presentation.resize_surface(
                    *pending_resize
                )

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
        window.on_resize = None
        self._keyboard_event_callback = None
        self._mouse_event_callback = None
        self._gamepad_event_callback = None
        self._gamepad_state_callback = None
        self._presentation = cast(Any, None)
        self._pending_resize = None

        try:
            presentation.close()
        finally:
            window.close()
            if getattr(self, "_xdisplay", None):
                xlib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
                xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
                xlib.XCloseDisplay(self._xdisplay)
                self._xdisplay = None

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

    def _on_resize(self, width: int, height: int) -> None:
        """Record a resize acknowledged by the native window system."""
        if self._closed or width <= 0 or height <= 0:
            return
        self._pending_resize = int(width), int(height)


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
        self._display_texture: Any = None
        self._host_upload = np.empty((0, 0, 4), dtype=np.uint8)
        self._cuda_buffer: spy.Buffer | None = None
        self._cuda_rgba: Tensor | None = None
        self._render_device = torch.device("cpu")
        self._has_cuda_submission = False
        self._create_frame_resources(*self._configured_surface_size())

    @property
    def size(self) -> tuple[int, int]:
        """Return the configured swapchain extent."""
        return self._width, self._height

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
        try:
            surface_texture = self._surface.acquire_next_image()
        except RuntimeError as error:
            _LOGGER.warning(
                "Native surface acquisition failed; recreating swapchain: %s", error
            )
            # SlangRHI collapses recoverable out-of-date acquisition into a
            # generic failure. Recreate the swapchain and drop this frame.
            self.resize_surface(self._width, self._height, force=True)
            return
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

    def resize_surface(
        self, width: int, height: int, *, force: bool = False
    ) -> tuple[int, int]:
        """Reconfigure the output surface after pending presentation completes."""
        if not force and (width, height) == (self._width, self._height):
            return self.size
        self._device.wait_for_idle()
        self._cuda_rgba = None
        self._cuda_buffer = None
        self._display_texture = None
        self._surface.configure(
            width=width,
            height=height,
            format=self._choose_surface_format(),
        )
        self._create_frame_resources(*self._configured_surface_size())
        return self.size

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

    def _create_frame_resources(self, width: int, height: int) -> None:
        """Create upload resources matching the presentation surface."""
        self._width = width
        self._height = height
        self._display_texture = self._device.create_texture(
            format=self._spy.Format.rgba8_unorm,
            width=width,
            height=height,
            usage=(
                self._spy.TextureUsage.shader_resource
                | self._spy.TextureUsage.copy_destination
            ),
            label="flashdreams_v2_native_window_texture",
        )
        self._host_upload = np.empty((height, width, 4), dtype=np.uint8)
        self._host_upload[..., 3] = 255
        self._render_device = torch.device("cpu")
        self._has_cuda_submission = False
        self._create_cuda_upload()

    def _configured_surface_size(self) -> tuple[int, int]:
        """Return the pixel extent SlangRHI accepted for the swapchain."""
        config = self._surface.config
        if config is None:
            raise RuntimeError("SlangRHI did not configure the native surface.")
        return int(config.width), int(config.height)

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
    event: spy.MouseEvent,
    *,
    width: int,
    height: int,
    lock_cursor_to_window: bool = False,
) -> MouseUserInputEvent | None:
    x, y = float(event.pos.x), float(event.pos.y)
    normalized_x = x / width
    normalized_y = y / height
    if not lock_cursor_to_window:
        normalized_x = min(1.0, max(0.0, normalized_x))
        normalized_y = min(1.0, max(0.0, normalized_y))
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
