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

"""CPU tests for shared SlangPy local-window input handling."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flashdreams.demo import (
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    LocalWindowIOFactory,
    LocalWindowOutputSink,
    SessionInfo,
)
from flashdreams.demo import application as application_module
from flashdreams.demo.local_input import SlangPyLocalInputHandler
from flashdreams.demo.local_window import SlangPyLocalWindowPresenter
from flashdreams.runtime import DRIVER_COMMAND, StepRequirements, StepResult
from flashdreams.runtime.inputs import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    CanonicalModality,
)

pytestmark = pytest.mark.ci_cpu


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


class _KeyboardEvent:
    def __init__(self, key: str, event_type: str) -> None:
        self.key = SimpleNamespace(name=key)
        self._event_type = event_type

    def is_key_press(self) -> bool:
        return self._event_type == "press"

    def is_key_release(self) -> bool:
        return self._event_type == "release"

    def is_key_repeat(self) -> bool:
        return self._event_type == "repeat"


def test_local_input_handler_tracks_keyboard_levels() -> None:
    clock = _Clock()
    handler = SlangPyLocalInputHandler(
        CanonicalInputSchema(modalities=(DRIVER_COMMAND,)),
        clock=clock,
    )
    handler.open(SessionInfo())

    handler.on_keyboard_event(_KeyboardEvent("w", "press"))
    pressed = handler.current_inputs()
    clock.value += 0.1
    held = handler.current_inputs()
    handler.on_keyboard_event(_KeyboardEvent("w", "release"))
    released = handler.current_inputs()

    assert pressed.values["driver_command"]["throttle"] == 1.0
    assert pressed.window.start_s == 0.0
    assert held.window.start_s == pressed.window.end_s
    assert released.window.start_s == held.window.end_s
    assert released.window.end_s > released.window.start_s
    assert held.values["driver_command"]["throttle"] == 1.0
    assert released.values["driver_command"]["throttle"] == 0.0
    assert released.metadata["canonical_sources"] == {"driver_command": "keyboard"}


def test_local_input_handler_uses_active_sdl_gamepad_axes() -> None:
    handler = SlangPyLocalInputHandler(
        CanonicalInputSchema(modalities=(DRIVER_COMMAND,))
    )
    handler.open(SessionInfo())

    handler.on_gamepad_state(
        SimpleNamespace(left_x=0.25, left_trigger=0.4, right_trigger=0.75)
    )
    inputs = handler.current_inputs()

    assert inputs.values["driver_command"] == {
        "throttle": 0.75,
        "brake": 0.4,
        "steer": -0.25,
        "stop": False,
        "reverse": False,
    }
    assert inputs.metadata["canonical_sources"] == {"driver_command": "gamepad"}


class _Presenter:
    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}
        self.pending_events: list[_KeyboardEvent] = []
        self.process_count = 0
        self.event_processed = threading.Event()
        self.frame_presented = threading.Event()
        self.present_started = threading.Event()
        self.allow_present = threading.Event()
        self.allow_present.set()
        self.present_threads: list[int] = []
        self.close_thread: int | None = None
        self.should_close = False

    def set_input_callbacks(self, **callbacks: Any) -> None:
        self.callbacks = callbacks

    def process_events(self) -> None:
        self.process_count += 1
        while self.pending_events:
            self.callbacks["on_keyboard_event"](self.pending_events.pop(0))
            self.event_processed.set()

    def present(self, _frame: object) -> bool:
        self.present_threads.append(threading.get_ident())
        self.present_started.set()
        self.allow_present.wait(timeout=2.0)
        self.frame_presented.set()
        return True

    def wait_until(self, _deadline_s: float) -> bool:
        return True

    def close(self) -> None:
        self.close_thread = threading.get_ident()


def test_local_window_rebinds_cuda_context_before_native_handle_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    calls: list[object] = []
    handles = [object()]
    presenter = object.__new__(SlangPyLocalWindowPresenter)
    presenter._spy = SimpleNamespace(
        get_cuda_current_context_native_handles=lambda: (
            calls.append("handles") or handles
        )
    )
    presenter._cuda_interop_unavailable_reason = None

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: calls.append("current_device") or 2,
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: calls.append(("set_device", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: calls.append("current_stream"),
    )

    assert presenter._cuda_existing_device_handles() == handles
    assert calls == [
        "current_device",
        ("set_device", 2),
        "current_stream",
        "handles",
    ]


def test_local_window_factory_shares_presenter_with_input_handler() -> None:
    presenter = _Presenter()
    factory = LocalWindowIOFactory(presenter_factory=lambda **_kwargs: presenter)
    handler = factory.create_input_handler(
        CanonicalInputSchema(modalities=(DRIVER_COMMAND,))
    )
    output = factory.create_output_sink()
    handler.open(SessionInfo())
    output.open(SessionInfo(video_width=64, video_height=32, frames_per_second=16.0))
    presenter.pending_events.append(_KeyboardEvent("a", "press"))
    assert presenter.event_processed.wait(timeout=1.0)

    inputs = handler.current_inputs()

    assert presenter.process_count >= 1
    assert inputs.values["driver_command"]["steer"] == 1.0
    output.close()
    handler.close()


class _InteractiveApplicationSession(IFlashDreamsApplicationSession):
    def __init__(
        self,
        *,
        presenter: _Presenter,
        windows: list[CanonicalInputWindow],
    ) -> None:
        self.presenter = presenter
        self.windows = windows
        self.step_index = 0
        self.closed = False

    def init(self) -> None:
        return None

    def session_info(self) -> SessionInfo:
        return SessionInfo(
            output_layout="tchw",
            steady_output_frame_count=1,
            frames_per_second=16.0,
            video_width=2,
            video_height=2,
        )

    def next_step_requirements(self) -> StepRequirements | None:
        if self.step_index >= 2:
            return None
        if self.step_index == 1:
            assert self.presenter.event_processed.wait(timeout=2.0)
        return StepRequirements(step_index=self.step_index)

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        self.windows.append(inputs)
        result = StepResult.from_video_chunk(
            step_index=self.step_index,
            video_chunk=torch.zeros((1, 3, 2, 2)),
            layout="tchw",
        )
        self.step_index += 1
        return result

    def close(self) -> None:
        self.closed = True


class _InteractiveApplication(IFlashDreamsApplication):
    def __init__(self, presenter: _Presenter) -> None:
        self.presenter = presenter
        self.windows: list[CanonicalInputWindow] = []
        self.session: _InteractiveApplicationSession | None = None

    @property
    def input_schema(self) -> CanonicalInputSchema:
        return CanonicalInputSchema(modalities=(DRIVER_COMMAND,))

    def init(self, commandline_args: tuple[str, ...] | list[str]) -> None:
        assert not commandline_args

    def create_session(self) -> IFlashDreamsApplicationSession:
        self.session = _InteractiveApplicationSession(
            presenter=self.presenter,
            windows=self.windows,
        )
        return self.session


class _EventAfterFirstFramePresenter(_Presenter):
    def present(self, frame: object) -> bool:
        if not self.present_threads:
            self.pending_events.append(_KeyboardEvent("w", "press"))
        return super().present(frame)


def test_public_application_keeps_local_window_input_and_output_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presenter = _EventAfterFirstFramePresenter()
    application = _InteractiveApplication(presenter)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )

    artifacts = application_module.run_application(
        "interactive-fake",
        io_factory=LocalWindowIOFactory(presenter_factory=lambda **_kwargs: presenter),
    )

    assert artifacts == ()
    assert len(application.windows) == 2
    assert application.windows[0].values["driver_command"]["throttle"] == 0.0
    assert application.windows[1].values["driver_command"]["throttle"] == 1.0
    assert presenter.callbacks
    assert presenter.present_threads
    assert presenter.close_thread is not None
    assert application.session is not None and application.session.closed


def test_local_window_output_owns_presenter_thread_and_queues_writes() -> None:
    caller_thread = threading.get_ident()
    factory_threads: list[int] = []
    presenter = _Presenter()
    presenter.allow_present.clear()

    def create_presenter(**_kwargs: object) -> _Presenter:
        factory_threads.append(threading.get_ident())
        return presenter

    sink = LocalWindowOutputSink(
        fps=16.0,
        presenter_factory=create_presenter,
    )
    sink.open(
        SessionInfo(
            video_width=2,
            video_height=2,
            frames_per_second=16.0,
        )
    )
    sink.begin_generation(0)

    decision = sink.write(
        StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((1, 3, 2, 2)),
            layout="tchw",
        )
    )

    assert presenter.present_started.wait(timeout=1.0)
    assert not presenter.frame_presented.is_set()
    presenter.allow_present.set()
    assert presenter.frame_presented.wait(timeout=1.0)
    sink.close()

    assert decision.metadata["presentation_backend"] == "slangpy"
    assert decision.metadata["cuda_resident"] is False
    assert factory_threads == presenter.present_threads
    assert factory_threads == [presenter.close_thread]
    assert factory_threads[0] != caller_thread


def test_local_window_output_replaces_oldest_pending_chunk() -> None:
    presenter = _Presenter()
    presenter.allow_present.clear()
    sink = LocalWindowOutputSink(
        fps=16.0,
        max_pending_chunks=1,
        presenter_factory=lambda **_kwargs: presenter,
    )
    sink.open(
        SessionInfo(
            video_width=2,
            video_height=2,
            frames_per_second=16.0,
        )
    )

    def result(step_index: int) -> StepResult:
        return StepResult.from_video_chunk(
            step_index=step_index,
            video_chunk=torch.zeros((1, 3, 2, 2)),
            layout="tchw",
        )

    try:
        first = sink.write(result(0))
        assert presenter.present_started.wait(timeout=1.0)
        second = sink.write(result(1))
        replacement = sink.write(result(2))

        assert not first.dropped
        assert first.drop_policy == "none"
        assert not second.dropped
        assert second.drop_policy == "none"
        assert not replacement.dropped
        assert replacement.drop_policy == "drop_oldest"
        assert replacement.metadata["replaced_pending_chunks"] == 1
    finally:
        presenter.allow_present.set()
        sink.close()


def test_local_input_handler_rejects_unknown_canonical_modality() -> None:
    schema = CanonicalInputSchema(
        modalities=(
            CanonicalModality(
                name="camera_look",
                payload_fields=frozenset({"yaw", "pitch"}),
            ),
        )
    )

    with pytest.raises(ValueError, match="camera_look"):
        SlangPyLocalInputHandler(schema)
