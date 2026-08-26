# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GPU-resident v2 native client window."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
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
                self.callbacks[f"on_{kind}_event"](event)

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
    presenter._cuda_rgb_interop = SimpleNamespace(
        close=lambda: calls.append("interop.close")
    )
    presenter._device = SimpleNamespace(
        wait_for_idle=lambda: calls.append("device.wait_for_idle")
    )
    presenter._display_texture = object()
    presenter._surface = object()
    presenter._window = SimpleNamespace(close=lambda: calls.append("window.close"))

    presenter.close()
    presenter.close()

    assert calls == ["interop.close", "device.wait_for_idle", "window.close"]
    assert presenter._cuda_rgb_interop is None
    assert presenter._device is None


def test_slangpy_presenter_drops_frame_when_shared_buffers_are_busy() -> None:
    class BusyInterop:
        def as_cuda_rgb_frame(self, _frame: object) -> object:
            return object()

        def enqueue_rgb_to_shared_rgba(self, _frame: object) -> bool:
            return False

    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._closed = False
    presenter._device = object()
    presenter._cuda_rgb_interop = BusyInterop()

    assert presenter.present_frame(
        native_window_module._CudaPresentationFrame(
            tensor=cast(
                torch.Tensor,
                SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0)),
            ),
            ready_event=cast(torch.cuda.Event, object()),
        )
    )


def test_slangpy_presenter_drops_ready_frame_when_swapchain_is_unavailable() -> None:
    shared_buffer = object()
    discarded: list[object] = []

    class ReadyInterop:
        def ready_rgba_buffer(self) -> tuple[object, object]:
            return shared_buffer, object()

        def discard_ready(self, buffer: object) -> None:
            discarded.append(buffer)

    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._cuda_rgb_interop = ReadyInterop()
    presenter._device = object()
    presenter._display_texture = object()
    presenter._surface = SimpleNamespace(
        config=True,
        acquire_next_image=lambda: None,
    )

    assert presenter._submit_ready_cuda_rgb()
    assert discarded == [shared_buffer]


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


def test_slangpy_presenter_binds_interop_to_the_first_cuda_frame_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    cuda_device = torch.device("cuda", 2)
    cuda_tensor = cast(
        torch.Tensor,
        SimpleNamespace(is_cuda=True, device=cuda_device),
    )

    class BusyInterop:
        def as_cuda_rgb_frame(self, _frame: object) -> object:
            return object()

        def enqueue_rgb_to_shared_rgba(self, _frame: object) -> bool:
            return False

    presenter = object.__new__(native_window_module._SlangPyNativeWindowPresenter)
    presenter._closed = False
    presenter._device = None
    presenter._cuda_rgb_interop = None

    def initialize(*, enable_cuda_interop: bool) -> None:
        calls.append(("initialize", enable_cuda_interop))
        presenter._device = object()
        presenter._cuda_rgb_interop = BusyInterop()

    presenter._initialize_render_resources = initialize
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: calls.append(("set_device", device)),
    )
    frame = native_window_module._CudaPresentationFrame(
        tensor=cuda_tensor,
        ready_event=cast(torch.cuda.Event, object()),
    )

    assert presenter.present_frame(frame)
    assert calls == [
        ("set_device", cuda_device),
        ("initialize", True),
    ]


def test_native_window_reports_input_and_close_from_event_pump() -> None:
    presenter = _Presenter()
    clock_values = iter((1_000_000, 1_001_000, 1_002_000, 1_003_000))
    window = NativeWindowClientWindow(
        presenter_factory=_presenter_factory(presenter),
        clock_ns=lambda: next(clock_values),
    )
    window.open(_session_desc())
    presenter.pending_events.put(("keyboard", _KeyboardEvent("w", pressed=True)))
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
    assert keyboard.key == "w"
    assert keyboard.state is KeyboardInputState.PRESSED
    assert isinstance(mouse, MouseUserInputEvent)
    assert mouse.action == "move"
    assert mouse.x == 0.5
    assert mouse.y == 0.25
    assert isinstance(events[2], CloseUserInputEvent)


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
    assert frames.device == source.output.device
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
