# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GPU-resident v2 native client window."""

from __future__ import annotations

import queue
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, call

import pytest
import torch

from flashdreams.api_v2.client_window import ClientWindowLink
from flashdreams.runtime_v2 import native_window_client_window as native_window_module
from flashdreams.runtime_v2.native_window_client_window import (
    NativeWindowClientWindow,
)
from flashdreams.runtime_v2.selected_file import (
    DEFAULT_MAX_FILE_BYTES,
    FileSelectionRequest,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FullscreenEnabledOptimalWindowSizeUserInputEvent,
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    SelectedFilesUserInputEvent,
)
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    import slangpy as spy

pytestmark = pytest.mark.ci_cpu


def _file_request(
    request_id: str,
    initial_path: str,
    *,
    accept: tuple[str, ...] = (),
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_file_count: int = 1,
) -> FileSelectionRequest:
    return FileSelectionRequest(
        request_id=request_id,
        initial_path=initial_path,
        accept=accept,
        max_file_bytes=max_file_bytes,
        max_file_count=max_file_count,
    )


def _fake_dialog(
    monkeypatch: pytest.MonkeyPatch,
    choose: Callable[[FileSelectionRequest], list[str]],
) -> None:
    """Run ``choose`` on the dialog thread instead of Tk."""
    monkeypatch.setattr(native_window_module, "_choose_paths", choose)


def _wait_for_input_events(window: NativeWindowClientWindow) -> list[object]:
    for _ in range(500):
        events = window.get_user_input_events().get_events()
        if events:
            return list(events)
        time.sleep(0.01)
    return []


@pytest.mark.parametrize("fullscreen", [False, True])
def test_fullscreen_creation_reports_size_on_first_poll(fullscreen: bool) -> None:
    presenter = _Presenter()
    presenter.size = (1920, 1080)
    factory = Mock(return_value=presenter)
    window = NativeWindowClientWindow(fullscreen=fullscreen, presenter_factory=factory)
    for _ in range(2):
        window.open(_session_desc())
        events = window.get_user_input_events().get_events()
        if fullscreen:
            assert len(events) == 1
            event = events[0]
            assert isinstance(event, FullscreenEnabledOptimalWindowSizeUserInputEvent)
            assert (event.width, event.height) == presenter.size
        else:
            assert events == []
        assert window.get_user_input_events().get_events() == []
    factory.assert_called_once_with(
        width=2, height=2, title="FlashDreams", fullscreen=fullscreen
    )
    window.close()


@pytest.mark.parametrize("fullscreen", [False, True])
def test_presenter_uses_slangpy_creation_mode(
    fullscreen: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes = SimpleNamespace(normal=object(), fullscreen=object())
    native_window = Mock(width=1920, height=1080)
    spy = SimpleNamespace(Window=Mock(return_value=native_window), WindowMode=modes)
    monkeypatch.setitem(sys.modules, "slangpy", spy)
    presentation = Mock(size=(1920, 1080))
    context = Mock(return_value=presentation)
    monkeypatch.setattr(native_window_module, "_PresentationContext", context)
    presenter = native_window_module._SlangPyNativeWindowPresenter(
        width=640, height=480, title="Test", fullscreen=fullscreen
    )
    spy.Window.assert_called_once_with(
        width=640,
        height=480,
        title="Test",
        mode=modes.fullscreen if fullscreen else modes.normal,
        resizable=True,
    )
    context.assert_called_once_with(
        spy=spy, window=native_window, width=1920, height=1080
    )
    assert presenter.resize(320, 240) == (1920, 1080)
    native_window.resize.assert_called_once_with(320, 240)
    presenter.close()


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        video_width=2,
        video_height=2,
        frames_per_second_for_ui=30,
        frames_per_second_for_step=30,
    )


def _result(value: int = 0) -> StepResult:
    return StepResult(
        step_index=value,
        output=torch.full((1, 3, 2, 2), value, dtype=torch.uint8),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
    )


def _keyboard_edges(events: Sequence[object]) -> list[tuple[str, KeyboardInputState]]:
    return [
        (event.key, event.state)
        for event in events
        if isinstance(event, KeyboardUserInputEvent)
    ]


class _KeyboardEvent:
    def __init__(self, key: str, *, pressed: bool) -> None:
        self.key = SimpleNamespace(name=key)
        self._pressed = pressed

    def is_key_press(self) -> bool:
        return self._pressed

    def is_key_release(self) -> bool:
        return not self._pressed

    def is_input(self) -> bool:
        return False


class _KeyboardRepeatEvent(_KeyboardEvent):
    def __init__(self, key: str) -> None:
        super().__init__(key, pressed=False)

    def is_key_release(self) -> bool:
        return False


class _TextInputEvent:
    def __init__(self, text: str) -> None:
        self.codepoint = ord(text)

    def is_input(self) -> bool:
        return True

    def is_key_press(self) -> bool:
        return False

    def is_key_release(self) -> bool:
        return False


class _MouseMoveEvent:
    def __init__(self, x: float, y: float) -> None:
        self.pos = SimpleNamespace(x=x, y=y)

    def is_move(self) -> bool:
        return True

    def is_button_down(self) -> bool:
        return False

    def is_button_up(self) -> bool:
        return False

    def is_scroll(self) -> bool:
        return False


class _MouseButtonEvent(_MouseMoveEvent):
    def __init__(
        self, x: float, y: float, *, pressed: bool, button: str = "left"
    ) -> None:
        super().__init__(x, y)
        self.button = SimpleNamespace(name=button)
        self._pressed = pressed

    def is_move(self) -> bool:
        return False

    def is_button_down(self) -> bool:
        return self._pressed

    def is_button_up(self) -> bool:
        return not self._pressed


class _GamepadEvent:
    def __init__(self, action: str) -> None:
        self.action = action

    def is_connect(self) -> bool:
        return self.action == "connected"

    def is_disconnect(self) -> bool:
        return self.action == "disconnected"


class _GamepadState:
    left_x = -0.25
    left_y = 0.5
    right_x = 0.75
    right_y = -1.25
    left_trigger = 1.25
    right_trigger = -0.5
    buttons = (1 << 1) | (1 << 9) | (1 << 13)


class _Presenter:
    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}
        self.pending_events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self.event_threads: list[int] = []
        self.presentation_threads: list[int] = []
        self.close_threads: list[int] = []
        self.presented: list[torch.Tensor] = []
        self.resizes: list[tuple[int, int]] = []
        self.size = (2, 2)
        self.cursor_options: list[tuple[bool, bool]] = []
        self.should_close = False

    @property
    def input_size(self) -> tuple[int, int]:
        return self.size

    def configure_cursor(
        self, *, hide_cursor: bool, lock_cursor_to_window: bool
    ) -> None:
        self.cursor_options.append((hide_cursor, lock_cursor_to_window))

    def set_input_callbacks(self, **callbacks: Any) -> None:
        self.callbacks = callbacks

    def process_events(self) -> None:
        self.event_threads.append(threading.get_ident())
        while True:
            try:
                kind, event = self.pending_events.get_nowait()
            except queue.Empty:
                return
            if kind == "close":
                self.should_close = True
            else:
                callback = (
                    "on_gamepad_state"
                    if kind == "gamepad_state"
                    else f"on_{kind}_event"
                )
                self.callbacks[callback](event)

    def present_frame(self, frame: object) -> bool:
        self.presentation_threads.append(threading.get_ident())
        assert isinstance(frame, torch.Tensor)
        self.presented.append(frame)
        return not self.should_close

    def resize(self, width: int, height: int) -> tuple[int, int]:
        self.resizes.append((width, height))
        self.size = (width, height)
        return self.size

    def close(self) -> None:
        self.close_threads.append(threading.get_ident())


def _presenter_factory(
    presenter: _Presenter,
) -> Callable[..., native_window_module._SlangPyNativeWindowPresenter]:
    return cast(Any, lambda **_kwargs: presenter)


def test_native_window_accepts_manual_resize_and_restores_after_minimizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_window = SimpleNamespace(
        width=2,
        height=2,
        process_events=Mock(),
        should_close=lambda: False,
        close=Mock(),
    )
    spy = SimpleNamespace(
        Window=Mock(return_value=native_window),
        WindowMode=SimpleNamespace(normal=object()),
        CursorMode=SimpleNamespace(normal=object()),
    )
    presentation = SimpleNamespace(
        size=(2, 2),
        resize_surface=Mock(side_effect=lambda width, height: (width, height)),
        present=Mock(),
        close=Mock(),
    )
    monkeypatch.setitem(sys.modules, "slangpy", spy)
    monkeypatch.setattr(
        native_window_module, "_PresentationContext", Mock(return_value=presentation)
    )
    window = NativeWindowClientWindow()
    window.open(_session_desc())
    spy.Window.assert_called_once_with(
        width=2,
        height=2,
        title="FlashDreams",
        mode=spy.WindowMode.normal,
        resizable=True,
    )

    pending_resizes: list[tuple[int, int]] = []

    def process_resizes() -> None:
        for width, height in pending_resizes:
            native_window.on_resize(width, height)

    for resizes, expected_size in (
        ([(4, 3), (5, 4)], (5, 4)),
        ([(0, 0)], (5, 4)),
        ([(8, 6)], (8, 6)),
    ):
        pending_resizes[:] = resizes
        native_window.process_events.side_effect = process_resizes
        assert window.get_user_input_events().get_events() == []
        window.write(_result())
        (frame,) = presentation.present.call_args.args
        width, height = expected_size
        assert frame.shape == (height, width, 3)

        native_window.process_events.side_effect = None
        native_window.on_mouse_event(_MouseMoveEvent(width / 2, height / 4))
        (event,) = window.get_user_input_events().get_events()
        assert isinstance(event, MouseUserInputEvent)
        assert (event.x, event.y) == (0.5, 0.25)

    assert presentation.resize_surface.call_args_list == [call(5, 4), call(8, 6)]
    window.close()


@pytest.mark.parametrize(("position", "opens_link"), ((1.0, False), (2.0, True)))
def test_native_mouse_and_links_use_resize_acknowledged_in_the_same_poll(
    monkeypatch: pytest.MonkeyPatch, position: float, opens_link: bool
) -> None:
    native_window = SimpleNamespace(
        width=2,
        height=2,
        process_events=Mock(),
        should_close=lambda: False,
        close=Mock(),
    )
    spy = SimpleNamespace(
        Window=Mock(return_value=native_window),
        WindowMode=SimpleNamespace(normal=object()),
        CursorMode=SimpleNamespace(normal=object()),
    )
    presentation = SimpleNamespace(
        size=(2, 2),
        resize_surface=Mock(return_value=(4, 4)),
        present=Mock(),
        close=Mock(),
    )
    open_browser = Mock(return_value=True)
    monkeypatch.setattr(native_window_module.webbrowser, "open_new_tab", open_browser)
    monkeypatch.setitem(sys.modules, "slangpy", spy)
    monkeypatch.setattr(
        native_window_module, "_PresentationContext", Mock(return_value=presentation)
    )
    window = NativeWindowClientWindow()
    window.open(_session_desc())
    link = ClientWindowLink(
        label="Project", url="https://example.com", bounds=(0.4, 0.4, 0.6, 0.6)
    )
    window.request_links((link,))

    def process_events() -> None:
        native_window.on_resize(4, 4)
        native_window.on_mouse_event(_MouseMoveEvent(position, position))
        for pressed in (True, False):
            native_window.on_mouse_event(
                _MouseButtonEvent(position, position, pressed=pressed)
            )
        presentation.resize_surface.assert_not_called()

    native_window.process_events.side_effect = process_events
    events = window.get_user_input_events().get_events()

    assert len(events) == (1 if opens_link else 3)
    for event in events:
        assert isinstance(event, MouseUserInputEvent)
        assert (event.x, event.y) == (position / 4, position / 4)
    if opens_link:
        open_browser.assert_called_once_with(link.url)
    else:
        open_browser.assert_not_called()
    presentation.resize_surface.assert_called_once_with(4, 4)
    window.write(_result())
    (frame,) = presentation.present.call_args.args
    assert frame.shape == (4, 4, 3)
    window.close()


def test_slangpy_presenter_uses_standard_window_event_pump() -> None:
    process_count = 0

    def process_events() -> None:
        nonlocal process_count
        process_count += 1

    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._closed = False
    presenter._window = SimpleNamespace(process_events=process_events)
    presenter._pending_resize = None

    presenter.process_events()

    assert process_count == 1


def test_slangpy_presenter_configures_acknowledged_window_extent() -> None:
    calls: list[tuple[object, ...]] = []
    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)

    def process_events() -> None:
        calls.append(("process_events",))
        presenter._on_resize(5, 4)

    window = SimpleNamespace(
        resize=lambda width, height: calls.append(("resize", width, height)),
        process_events=process_events,
    )
    presentation = SimpleNamespace(
        resize_surface=lambda width, height: (
            calls.append(("resize_surface", width, height)) or (width, height)
        )
    )
    presenter._closed = False
    presenter._window = window
    presenter._presentation = presentation
    presenter._fullscreen = False
    presenter._width = 2
    presenter._height = 2
    presenter._pending_resize = None

    assert presenter.resize(8, 6) == (2, 2)
    assert calls == [("resize", 8, 6)]

    presenter.process_events()

    assert presenter.size == (5, 4)
    assert calls == [
        ("resize", 8, 6),
        ("process_events",),
        ("resize_surface", 5, 4),
    ]


def test_slangpy_presenter_maps_cursor_options_independently() -> None:
    modes = SimpleNamespace(
        normal=object(),
        hidden=object(),
        disabled=object(),
    )
    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._spy = SimpleNamespace(CursorMode=modes)
    presenter._window = SimpleNamespace(cursor_mode=None)

    presenter.configure_cursor(hide_cursor=False, lock_cursor_to_window=False)
    assert presenter._window.cursor_mode is modes.normal

    presenter.configure_cursor(hide_cursor=True, lock_cursor_to_window=False)
    assert presenter._window.cursor_mode is modes.hidden

    presenter.configure_cursor(hide_cursor=False, lock_cursor_to_window=True)
    assert presenter._window.cursor_mode is modes.disabled


def test_slangpy_presenter_waits_for_gpu_work_before_releasing_resources() -> None:
    calls: list[str] = []
    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._closed = False
    presenter._keyboard_event_callback = object()
    presenter._mouse_event_callback = object()
    presenter._gamepad_event_callback = object()
    presenter._gamepad_state_callback = object()
    presenter._presentation = SimpleNamespace(
        close=lambda: calls.append("presentation.close")
    )
    presenter._window = SimpleNamespace(
        on_resize=object(),
        on_keyboard_event=object(),
        on_mouse_event=object(),
        on_gamepad_event=object(),
        on_gamepad_state=object(),
        close=lambda: calls.append("window.close"),
    )

    presenter.close()
    presenter.close()

    assert calls == ["presentation.close", "window.close"]
    assert presenter._window.on_resize is None
    assert presenter._window.on_keyboard_event is None
    assert presenter._window.on_mouse_event is None
    assert presenter._window.on_gamepad_event is None
    assert presenter._window.on_gamepad_state is None
    assert presenter._keyboard_event_callback is None
    assert presenter._mouse_event_callback is None
    assert presenter._gamepad_event_callback is None
    assert presenter._gamepad_state_callback is None
    assert presenter._presentation is None


def test_presentation_context_releases_cuda_view_before_backing_resources() -> None:
    released: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def __del__(self) -> None:
            released.append(self.name)

    class Device(Resource):
        def wait_for_idle(self) -> None:
            released.append("device.wait_for_idle")

    context = object.__new__(native_window_module._PresentationContext)
    context._cuda_rgba = Resource("cuda_rgba")
    context._cuda_buffer = Resource("cuda_buffer")
    context._display_texture = Resource("display_texture")
    context._surface = Resource("surface")
    context._device = Device("device")
    context._render_device = torch.device("cuda")
    context._has_cuda_submission = True

    context.close()
    context.close()

    assert released == [
        "device.wait_for_idle",
        "cuda_rgba",
        "cuda_buffer",
        "display_texture",
        "surface",
        "device",
    ]
    assert context._device is None
    assert context._render_device == torch.device("cpu")
    assert context._has_cuda_submission is False


def test_slangpy_presenter_delegates_plain_tensor_to_presentation_context() -> None:
    presented: list[torch.Tensor] = []
    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._closed = False
    presenter._presentation = SimpleNamespace(
        present=lambda frame: presented.append(frame)
    )
    frame = torch.zeros((2, 2, 3), dtype=torch.uint8)

    assert presenter.present_frame(frame)
    assert presented == [frame]


def test_presentation_recovers_from_failed_surface_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = object.__new__(native_window_module._PresentationContext)
    acquire_next_image = Mock(side_effect=RuntimeError("SLANG_FAIL"))
    context._surface = SimpleNamespace(
        config=object(),
        acquire_next_image=acquire_next_image,
    )
    context._width = 2
    context._height = 2
    resize_surface = Mock(return_value=(2, 2))
    monkeypatch.setattr(context, "resize_surface", resize_surface)
    frame = torch.zeros((2, 2, 3), dtype=torch.uint8)

    context.present(frame)

    acquire_next_image.assert_called_once_with()
    resize_surface.assert_called_once_with(2, 2, force=True)


def test_presentation_surface_resize_recreates_sized_upload_resources() -> None:
    texture_requests: list[dict[str, Any]] = []
    configured = SimpleNamespace(width=5, height=4)
    surface = SimpleNamespace(
        info=SimpleNamespace(formats=[1]),
        config=configured,
        configure=Mock(),
    )
    device = SimpleNamespace(
        supports_cuda_interop=False,
        wait_for_idle=Mock(),
        create_texture=lambda **kwargs: texture_requests.append(kwargs) or object(),
    )
    context = object.__new__(native_window_module._PresentationContext)
    context._spy = SimpleNamespace(
        Format=SimpleNamespace(
            rgba8_unorm=1,
            bgra8_unorm=2,
            bgrx8_unorm=3,
        ),
        TextureUsage=SimpleNamespace(
            shader_resource=1,
            copy_destination=2,
        ),
    )
    context._width = 2
    context._height = 2
    context._device = device
    context._surface = surface
    context._display_texture = object()
    context._cuda_buffer = object()
    context._cuda_rgba = object()
    context._render_device = torch.device("cuda")
    context._has_cuda_submission = True

    assert context.resize_surface(4, 3) == (5, 4)

    device.wait_for_idle.assert_called_once_with()
    surface.configure.assert_called_once_with(width=4, height=3, format=1)
    assert texture_requests[-1]["width"] == 5
    assert texture_requests[-1]["height"] == 4
    assert context._host_upload.shape == (4, 5, 4)
    assert context._cuda_buffer is None
    assert context._cuda_rgba is None
    assert context._render_device == torch.device("cpu")
    assert context._has_cuda_submission is False


def test_native_window_resamples_resized_ui_to_the_window_surface() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    resized_ui = StepResult(
        step_index=0,
        output=torch.zeros((1, 3, 3, 4), dtype=torch.float32),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
    )

    window.write(resized_ui)
    window.close()

    assert presenter.presented[0].shape == (2, 2, 3)


def test_window_lifecycle_and_presentation_stay_on_the_ui_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui_thread = threading.get_ident()
    presenter = _Presenter()
    factory_threads: list[int] = []
    conversion_threads: list[int] = []
    real_conversion = result_to_rgb24_tensor

    def create_presenter(**_kwargs: object) -> _Presenter:
        factory_threads.append(threading.get_ident())
        return presenter

    def record_conversion(
        result: StepResult,
        desc: SessionDesc,
        presentation_size: tuple[int, int],
    ) -> torch.Tensor:
        conversion_threads.append(threading.get_ident())
        return real_conversion(result, desc, presentation_size)

    monkeypatch.setattr(
        native_window_module,
        "result_to_rgb24_tensor",
        record_conversion,
    )
    window = NativeWindowClientWindow(presenter_factory=cast(Any, create_presenter))
    window.open(_session_desc())
    window.get_user_input_events()
    window.write(_result())
    window.close()

    assert (
        factory_threads
        == presenter.event_threads
        == presenter.presentation_threads
        == presenter.close_threads
        == conversion_threads
        == [ui_thread]
    )


def test_replacement_session_reuses_native_window_and_rebases_input() -> None:
    presenter = _Presenter()
    factory_calls = 0
    clock_values = iter((1_000_000, 1_001_000, 2_000_000, 2_001_000))

    def create_presenter(**_kwargs: object) -> _Presenter:
        nonlocal factory_calls
        factory_calls += 1
        return presenter

    window = NativeWindowClientWindow(
        presenter_factory=cast(Any, create_presenter),
        clock_ns=lambda: next(clock_values),
    )
    window.open(_session_desc())
    window._on_keyboard_event(
        cast("spy.KeyboardEvent", _KeyboardEvent("up", pressed=True))
    )
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("up", pressed=True)))

    (event,) = window.get_user_input_events().get_events()
    window.close()

    assert factory_calls == 1
    assert event.get_timestamp() == 1
    assert len(presenter.close_threads) == 1


def test_presentation_context_copies_frames_to_its_render_device() -> None:
    source_device = torch.device("cuda", 2)
    render_device = torch.device("cuda", 1)
    copied = cast(torch.Tensor, object())
    copy_calls: list[torch.device] = []
    frame = cast(
        torch.Tensor,
        SimpleNamespace(
            device=source_device,
            to=lambda device: copy_calls.append(device) or copied,
        ),
    )
    context = object.__new__(native_window_module._PresentationContext)
    context._render_device = render_device

    assert context._frame_on_render_device(frame) is copied
    assert copy_calls == [render_device]


def test_presentation_context_reuses_frames_on_its_render_device() -> None:
    render_device = torch.device("cuda", 1)
    frame = cast(torch.Tensor, SimpleNamespace(device=render_device))
    context = object.__new__(native_window_module._PresentationContext)
    context._render_device = render_device

    assert context._frame_on_render_device(frame) is frame


def test_native_window_reports_input_and_close_from_event_pump() -> None:
    presenter = _Presenter()
    clock_values = iter((1_000_000, 1_001_000, 1_002_000, 1_003_000))
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
        clock_ns=lambda: next(clock_values),
    )
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("up", pressed=True)))
    presenter.pending_events.put(
        (
            "mouse",
            _MouseMoveEvent(1.0, 0.5),
        )
    )
    presenter.pending_events.put(("close", None))

    events = window.get_user_input_events().get_events()
    window.close()

    assert [event.get_timestamp() for event in events] == [1, 2, 3]
    keyboard = events[0]
    mouse = events[1]
    assert isinstance(keyboard, KeyboardUserInputEvent)
    assert keyboard.key == "ArrowUp"
    assert keyboard.state is KeyboardInputState.PRESSED
    assert isinstance(mouse, MouseUserInputEvent)
    assert mouse.action == "move"
    assert mouse.x == 0.5
    assert mouse.y == 0.25
    assert isinstance(events[2], CloseUserInputEvent)


def test_cursor_options_can_change_while_native_window_is_running() -> None:
    presenter = _Presenter()
    presenter_args: dict[str, object] = {}

    def create_presenter(**kwargs: object) -> _Presenter:
        presenter_args.update(kwargs)
        return presenter

    window = NativeWindowClientWindow(presenter_factory=cast(Any, create_presenter))
    window.request_hide_cursor(True)
    window.open(_session_desc())
    window.request_lock_cursor_to_window(True)
    presenter.pending_events.put(("mouse", _MouseMoveEvent(3.0, -1.0)))
    locked_events = window.get_user_input_events().get_events()

    window.request_hide_cursor(False)
    window.request_lock_cursor_to_window(False)
    presenter.pending_events.put(("mouse", _MouseMoveEvent(3.0, -1.0)))
    unlocked_events = window.get_user_input_events().get_events()
    window.close()
    window.request_lock_cursor_to_window(True)

    assert "hide_cursor" not in presenter_args
    assert "lock_cursor_to_window" not in presenter_args
    assert presenter.cursor_options == [
        (True, False),
        (True, True),
        (False, True),
        (False, False),
    ]
    locked_mouse = cast(MouseUserInputEvent, locked_events[0])
    unlocked_mouse = cast(MouseUserInputEvent, unlocked_events[0])
    assert (locked_mouse.x, locked_mouse.y) == (1.5, -0.5)
    assert (unlocked_mouse.x, unlocked_mouse.y) == (1.0, 0.0)


@pytest.mark.parametrize("window_size", ((2, 2), (8, 6)))
def test_native_link_consumes_click_and_opens_browser_on_release(
    monkeypatch: pytest.MonkeyPatch, window_size: tuple[int, int]
) -> None:
    open_browser = Mock(return_value=True)
    monkeypatch.setattr(native_window_module.webbrowser, "open_new_tab", open_browser)
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    window.request_new_window_size(window_size)
    link = ClientWindowLink(
        label="Project",
        url="https://example.com/project",
        bounds=(0.25, 0.25, 0.75, 0.75),
    )
    window.request_links((link,))
    open_browser.assert_not_called()

    for pressed in (True, False):
        presenter.pending_events.put(
            (
                "mouse",
                _MouseButtonEvent(
                    window_size[0] / 2, window_size[1] / 2, pressed=pressed
                ),
            )
        )
        assert window.get_user_input_events().get_events() == []
        if pressed:
            open_browser.assert_not_called()

    open_browser.assert_called_once_with(link.url)
    window.close()


@pytest.mark.parametrize(
    ("button", "locked", "press_x", "release_x", "event_count"),
    (
        ("right", False, 0.5, 0.5, 2),
        ("left", True, 0.5, 0.5, 2),
        ("left", False, 0.5, -0.25, 0),
        ("left", False, 0.5, 0.75, 0),
        ("left", False, 0.9, 0.5, 2),
    ),
)
def test_native_link_ignores_other_buttons_capture_and_cancelled_clicks(
    monkeypatch: pytest.MonkeyPatch,
    button: str,
    locked: bool,
    press_x: float,
    release_x: float,
    event_count: int,
) -> None:
    open_browser = Mock()
    monkeypatch.setattr(native_window_module.webbrowser, "open_new_tab", open_browser)
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    window.request_links(
        (
            ClientWindowLink(
                label="Project", url="https://example.com", bounds=(0, 0, 0.75, 1)
            ),
        )
    )
    window.request_lock_cursor_to_window(locked)
    for pressed, x in ((True, press_x), (False, release_x)):
        presenter.pending_events.put(
            ("mouse", _MouseButtonEvent(x * 2, 1, pressed=pressed, button=button))
        )

    assert len(window.get_user_input_events().get_events()) == event_count
    open_browser.assert_not_called()
    window.close()


@pytest.mark.parametrize("reset", ("replace_links", "open", "close"))
def test_native_window_clears_links_on_replacement_and_session_lifecycle(
    monkeypatch: pytest.MonkeyPatch, reset: str
) -> None:
    open_browser = Mock()
    monkeypatch.setattr(native_window_module.webbrowser, "open_new_tab", open_browser)
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    window.request_links(
        (
            ClientWindowLink(
                label="Project", url="https://example.com", bounds=(0, 0, 1, 1)
            ),
        )
    )
    presenter.pending_events.put(("mouse", _MouseButtonEvent(1, 1, pressed=True)))
    assert window.get_user_input_events().get_events() == []
    if reset == "replace_links":
        window.request_links(())
    elif reset == "open":
        window.open(_session_desc())
    else:
        window.close()
        assert window._links == ()
        window.open(_session_desc())
    presenter.pending_events.put(("mouse", _MouseButtonEvent(1, 1, pressed=False)))
    assert len(window.get_user_input_events().get_events()) == (
        0 if reset == "replace_links" else 1
    )
    for pressed in (True, False):
        presenter.pending_events.put(
            ("mouse", _MouseButtonEvent(1, 1, pressed=pressed))
        )

    assert len(window.get_user_input_events().get_events()) == 2
    open_browser.assert_not_called()
    window.close()


@pytest.mark.parametrize(
    "error", (None, webbrowser.Error("unavailable"), OSError("missing"))
)
def test_native_link_warns_when_browser_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception | None,
) -> None:
    open_browser = Mock(return_value=False, side_effect=error)
    monkeypatch.setattr(native_window_module.webbrowser, "open_new_tab", open_browser)
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    window.request_links(
        (
            ClientWindowLink(
                label="Project", url="https://example.com", bounds=(0, 0, 1, 1)
            ),
        )
    )
    for pressed in (True, False):
        presenter.pending_events.put(
            ("mouse", _MouseButtonEvent(1, 1, pressed=pressed))
        )

    assert window.get_user_input_events().get_events() == []
    open_browser.assert_called_once_with("https://example.com")
    assert "open link" in caplog.text
    window.close()


def test_native_window_reports_a_selected_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chosen = tmp_path / "seed.png"
    chosen.write_bytes(b"png-bytes")
    _fake_dialog(monkeypatch, lambda _: [str(chosen)])
    presenter = _Presenter()
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
    )
    window.open(_session_desc())
    window.request_file_selector(_file_request("open-1", str(tmp_path)))
    events = _wait_for_input_events(window)
    window.close()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SelectedFilesUserInputEvent)
    assert event.request_id == "open-1"
    assert event.status is SelectedFilesStatus.OK
    assert event.files[0].name == "seed.png"
    assert event.files[0].data == b"png-bytes"


def test_native_window_reports_cancelled_file_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_dialog(monkeypatch, lambda _: [])
    presenter = _Presenter()
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
    )
    window.open(_session_desc())
    window.request_file_selector(_file_request("open-1", str(tmp_path)))
    events = _wait_for_input_events(window)
    window.close()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SelectedFilesUserInputEvent)
    assert event.request_id == "open-1"
    assert event.status is SelectedFilesStatus.CANCELLED
    assert event.files == ()


def test_native_window_reports_unavailable_when_picker_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_display(_: FileSelectionRequest) -> list[str]:
        raise RuntimeError("Tk cannot open a display.")

    _fake_dialog(monkeypatch, no_display)
    presenter = _Presenter()
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
    )
    window.open(_session_desc())
    window.request_file_selector(_file_request("open-1", str(tmp_path)))
    events = _wait_for_input_events(window)
    window.close()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SelectedFilesUserInputEvent)
    assert event.request_id == "open-1"
    assert event.status is SelectedFilesStatus.UNAVAILABLE
    assert event.files == ()


def test_selected_files_from_paths_skips_unreadable_and_oversize(
    tmp_path: Path,
) -> None:
    def load(
        path: Path, **limits: Any
    ) -> tuple[tuple[object, ...], SelectedFilesStatus]:
        return native_window_module._selected_files_from_paths(
            (str(path),), _file_request("open-1", str(tmp_path), **limits)
        )

    files, status = load(tmp_path / "gone.png")
    assert files == ()
    assert status is SelectedFilesStatus.UNAVAILABLE

    too_big = tmp_path / "big.bin"
    too_big.write_bytes(b"abcd")
    files, status = load(too_big, max_file_bytes=3)
    assert files == ()
    assert status is SelectedFilesStatus.TOO_LARGE

    wrong_type = tmp_path / "notes.txt"
    wrong_type.write_bytes(b"hi")
    files, status = load(wrong_type, accept=(".png",))
    assert files == ()
    assert status is SelectedFilesStatus.DISALLOWED_TYPE

    too_big_wrong_type = tmp_path / "huge.txt"
    too_big_wrong_type.write_bytes(b"abcd")
    files, status = load(too_big_wrong_type, accept=(".png",), max_file_bytes=1)
    assert files == ()
    assert status is SelectedFilesStatus.DISALLOWED_TYPE


@pytest.mark.skipif(not Path("/dev/zero").exists(), reason="needs /dev/zero")
def test_selected_files_from_paths_rejects_a_non_regular_file(tmp_path: Path) -> None:
    endless = tmp_path / "endless.bin"
    endless.symlink_to("/dev/zero")
    files, status = native_window_module._selected_files_from_paths(
        (str(endless),), _file_request("open-1", str(tmp_path), accept=(".bin",))
    )
    assert files == ()
    assert status is SelectedFilesStatus.UNAVAILABLE


def test_native_window_reports_multiple_selected_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    _fake_dialog(monkeypatch, lambda _: [str(first), str(second)])
    presenter = _Presenter()
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
    )
    window.open(_session_desc())
    window.request_file_selector(
        _file_request("open-1", str(tmp_path), max_file_count=2)
    )
    events = _wait_for_input_events(window)
    window.close()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SelectedFilesUserInputEvent)
    assert event.status is SelectedFilesStatus.OK
    assert [chosen.name for chosen in event.files] == ["a.bin", "b.bin"]
    assert [chosen.data for chosen in event.files] == [b"one", b"two"]


def test_native_window_drops_a_flushed_dialog_and_is_busy_until_it_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chosen = tmp_path / "stale.png"
    chosen.write_bytes(b"png")
    user_closes = threading.Event()

    def choose(_: FileSelectionRequest) -> list[str]:
        user_closes.wait(5)
        return [str(chosen)]

    _fake_dialog(monkeypatch, choose)
    read: list[object] = []
    monkeypatch.setattr(
        native_window_module,
        "_selected_files_from_paths",
        lambda paths, request: read.append(paths) or ((), SelectedFilesStatus.OK),
    )
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    window.request_file_selector(_file_request("open-1", str(tmp_path)))
    window.flush_file_selectors()

    window.request_file_selector(_file_request("open-2", str(tmp_path)))
    busy = window.get_user_input_events().get_events()
    user_closes.set()
    thread = window._file_dialog_thread
    assert thread is not None
    thread.join(5)
    stale = window.get_user_input_events().get_events()
    window.close()

    assert len(busy) == 1
    assert isinstance(busy[0], SelectedFilesUserInputEvent)
    assert busy[0].request_id == "open-2"
    assert busy[0].status is SelectedFilesStatus.BUSY
    assert stale == []
    assert read == [], "a flushed dialog's files are never read"


def test_native_window_reports_standard_gamepad_events() -> None:
    presenter = _Presenter()
    clock_values = iter((1_000_000, 1_001_000, 1_002_000, 1_003_000))
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
        clock_ns=lambda: next(clock_values),
    )
    window.open(_session_desc())
    presenter.pending_events.put(("gamepad", _GamepadEvent("connected")))
    presenter.pending_events.put(("gamepad_state", _GamepadState()))
    presenter.pending_events.put(("gamepad", _GamepadEvent("disconnected")))

    events = window.get_user_input_events().get_events()
    window.close()

    assert [event.get_timestamp() for event in events] == [1, 2, 3]
    connected, state, disconnected = events
    assert isinstance(connected, GamepadUserInputEvent)
    assert connected.action == "connected"
    assert connected.mapping == "standard"
    assert isinstance(state, GamepadUserInputEvent)
    assert state.action == "state"
    assert state.axes == (-0.25, 0.5, 0.75, -1.0)
    assert state.buttons == (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    )
    assert state.pressed == (
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    )
    assert isinstance(disconnected, GamepadUserInputEvent)
    assert disconnected.action == "disconnected"


def test_native_text_input_uses_slangpy_resolved_shift_character() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    presenter.pending_events.put(
        ("keyboard", _KeyboardEvent("left_shift", pressed=True))
    )
    presenter.pending_events.put(("keyboard", _KeyboardEvent("a", pressed=True)))
    presenter.pending_events.put(("keyboard", _TextInputEvent("A")))
    presenter.pending_events.put(("keyboard", _KeyboardEvent("a", pressed=False)))
    presenter.pending_events.put(
        ("keyboard", _KeyboardEvent("left_shift", pressed=False))
    )

    events = window.get_user_input_events().get_events()
    window.close()

    keys = [
        (data.key, data.state)
        for event in events
        if isinstance(data := event, KeyboardUserInputEvent)
    ]
    assert keys == [
        ("Shift", KeyboardInputState.PRESSED),
        ("A", KeyboardInputState.PRESSED),
        ("A", KeyboardInputState.RELEASED),
        ("Shift", KeyboardInputState.RELEASED),
    ]


def test_native_text_input_discards_repeat_callbacks_after_release() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("d", pressed=True)))
    presenter.pending_events.put(("keyboard", _TextInputEvent("d")))

    pressed = window.get_user_input_events().get_events()
    presenter.pending_events.put(("keyboard", _KeyboardRepeatEvent("d")))
    assert window.get_user_input_events().get_events() == []
    presenter.pending_events.put(("keyboard", _TextInputEvent("d")))
    assert window.get_user_input_events().get_events() == []

    presenter.pending_events.put(("keyboard", _KeyboardRepeatEvent("d")))
    presenter.pending_events.put(("keyboard", _KeyboardEvent("d", pressed=False)))
    released = window.get_user_input_events().get_events()
    presenter.pending_events.put(("keyboard", _TextInputEvent("d")))
    assert window.get_user_input_events().get_events() == []
    window.close()

    assert _keyboard_edges([*pressed, *released]) == [
        ("d", KeyboardInputState.PRESSED),
        ("d", KeyboardInputState.RELEASED),
    ]


@pytest.mark.parametrize("text", ("a", "A"))
def test_native_text_input_coalesces_across_event_polls(text: str) -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("a", pressed=True)))

    assert window.get_user_input_events().get_events() == []

    presenter.pending_events.put(("keyboard", _TextInputEvent(text)))
    pressed = window.get_user_input_events().get_events()
    presenter.pending_events.put(("keyboard", _KeyboardEvent("a", pressed=False)))
    released = window.get_user_input_events().get_events()
    window.close()

    assert _keyboard_edges([*pressed, *released]) == [
        (text, KeyboardInputState.PRESSED),
        (text, KeyboardInputState.RELEASED),
    ]


def test_native_printable_key_without_text_is_flushed_after_one_poll() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("w", pressed=True)))

    assert window.get_user_input_events().get_events() == []
    pressed = window.get_user_input_events().get_events()
    presenter.pending_events.put(("keyboard", _KeyboardEvent("w", pressed=False)))
    released = window.get_user_input_events().get_events()
    window.close()

    assert _keyboard_edges([*pressed, *released]) == [
        ("w", KeyboardInputState.PRESSED),
        ("w", KeyboardInputState.RELEASED),
    ]


def test_native_printable_release_flushes_pending_press() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("z", pressed=True)))
    presenter.pending_events.put(("keyboard", _KeyboardEvent("z", pressed=False)))

    events = window.get_user_input_events().get_events()
    window.close()

    assert _keyboard_edges(events) == [
        ("z", KeyboardInputState.PRESSED),
        ("z", KeyboardInputState.RELEASED),
    ]


def test_native_close_flushes_pending_printable_press() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("w", pressed=True)))
    presenter.pending_events.put(("close", None))

    events = window.get_user_input_events().get_events()
    window.close()

    assert len(events) == 2
    pressed, closed = events
    assert isinstance(pressed, KeyboardUserInputEvent)
    assert pressed.key == "w"
    assert pressed.state is KeyboardInputState.PRESSED
    assert isinstance(closed, CloseUserInputEvent)


@pytest.mark.parametrize(
    ("slangpy_name", "runtime_key"),
    (
        ("left_shift", "Shift"),
        ("right_control", "Control"),
        ("left_alt", "Alt"),
        ("right_super", "Meta"),
    ),
)
def test_native_modifier_names_match_browser_key_values(
    slangpy_name: str,
    runtime_key: str,
) -> None:
    data = native_window_module._keyboard_event(
        cast("spy.KeyboardEvent", _KeyboardEvent(slangpy_name, pressed=True))
    )

    assert data is not None
    assert data.key == runtime_key
    assert data.state is KeyboardInputState.PRESSED


@pytest.mark.parametrize(
    ("slangpy_name", "runtime_key"),
    (
        ("escape", "Escape"),
        ("enter", "Enter"),
        ("keypad_enter", "Enter"),
        ("left", "ArrowLeft"),
        ("right", "ArrowRight"),
        ("up", "ArrowUp"),
        ("down", "ArrowDown"),
    ),
)
@pytest.mark.parametrize("pressed", (True, False))
def test_native_navigation_keys_match_browser_key_values(
    slangpy_name: str, runtime_key: str, pressed: bool
) -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    window.open(_session_desc())
    try:
        presenter.pending_events.put(
            ("keyboard", _KeyboardEvent(slangpy_name, pressed=pressed))
        )
        events = window.get_user_input_events().get_events()
        assert _keyboard_edges(events) == [
            (
                runtime_key,
                KeyboardInputState.PRESSED if pressed else KeyboardInputState.RELEASED,
            )
        ]
        assert window.get_user_input_events().get_events() == []
    finally:
        window.close()


@pytest.mark.parametrize(
    ("slangpy_name", "runtime_key"),
    (("space", " "), ("key7", "7"), ("minus", "-"), ("left_bracket", "[")),
)
def test_native_printable_key_names_become_text_input_values(
    slangpy_name: str,
    runtime_key: str,
) -> None:
    data = native_window_module._keyboard_event(
        cast("spy.KeyboardEvent", _KeyboardEvent(slangpy_name, pressed=True))
    )

    assert data is not None
    assert data.key == runtime_key
    assert data.state is KeyboardInputState.PRESSED


def test_write_before_open_is_rejected() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))

    with pytest.raises(RuntimeError, match="open"):
        window.write(_result())


def test_native_window_must_open_on_the_process_main_thread() -> None:
    presenter = _Presenter()
    window = NativeWindowClientWindow(presenter_factory=_presenter_factory(presenter))
    errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def open_window() -> None:
        try:
            window.open(_session_desc())
        except BaseException as error:
            errors.put(error)

    worker = threading.Thread(target=open_window)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    error = errors.get_nowait()
    assert isinstance(error, RuntimeError)
    assert "process main thread" in str(error)
