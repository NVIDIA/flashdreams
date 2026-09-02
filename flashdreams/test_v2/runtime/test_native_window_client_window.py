# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GPU-resident v2 native client window."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from flashdreams.runtime_v2 import native_window_client_window as native_window_module
from flashdreams.runtime_v2.native_window_client_window import (
    NativeWindowClientWindow,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

if TYPE_CHECKING:
    import slangpy as spy

pytestmark = pytest.mark.ci_cpu


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
        self.should_close = False

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

    def close(self) -> None:
        self.close_threads.append(threading.get_ident())


def _presenter_factory(
    presenter: _Presenter,
) -> Callable[..., native_window_module._SlangPyNativeWindowPresenter]:
    return cast(Any, lambda **_kwargs: presenter)


def test_slangpy_presenter_uses_standard_window_event_pump() -> None:
    process_count = 0

    def process_events() -> None:
        nonlocal process_count
        process_count += 1

    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._closed = False
    presenter._window = SimpleNamespace(process_events=process_events)

    presenter.process_events()

    assert process_count == 1


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
        on_keyboard_event=object(),
        on_mouse_event=object(),
        on_gamepad_event=object(),
        on_gamepad_state=object(),
        close=lambda: calls.append("window.close"),
    )

    presenter.close()
    presenter.close()

    assert calls == ["presentation.close", "window.close"]
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

    def record_conversion(result: StepResult, desc: SessionDesc) -> torch.Tensor:
        conversion_threads.append(threading.get_ident())
        return real_conversion(result, desc)

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
    assert keyboard.key == "up"
    assert keyboard.state is KeyboardInputState.PRESSED
    assert isinstance(mouse, MouseUserInputEvent)
    assert mouse.action == "move"
    assert mouse.x == 0.5
    assert mouse.y == 0.25
    assert isinstance(events[2], CloseUserInputEvent)


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


def test_device_conversion_does_not_materialize_a_host_array() -> None:
    source = StepResult(
        step_index=0,
        output=torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
    )

    frames = result_to_rgb24_tensor(source, _session_desc())

    assert isinstance(frames, torch.Tensor)
    assert frames.device == source.read_output().device
    assert frames.shape == (1, 2, 2, 3)
    assert frames.dtype is torch.uint8
    assert torch.all(frames == 128)


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
