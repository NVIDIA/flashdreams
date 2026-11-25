# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for browser ownership boundaries across both runtime loops."""

import pickle
import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from numpy import uint64

from flashdreams.api_v2.loop import IModelLoop, IUILoop
from flashdreams.runtime_v2.coordination import StepAgreement
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.runtime_v2.selected_file import (
    FileSelectionRequest,
    SelectedFile,
    SelectedFilesStatus,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    InputOwnerChangedUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    NumeralKeypadUserInputEvent,
    QueryStringUserInputEvent,
    ResetUserInputEvent,
    SelectedFilesUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _key(key: str, timestamp: int = 1) -> KeyboardUserInputEvent:
    return KeyboardUserInputEvent(
        timestamp=uint64(timestamp), key=key, state=KeyboardInputState.PRESSED
    )


def _boundary(timestamp: int = 0) -> InputOwnerChangedUserInputEvent:
    return InputOwnerChangedUserInputEvent(timestamp=uint64(timestamp))


class _Model(IModelLoop[list]):
    def step(self, step_index, events):
        self.state.append((step_index, events.get_events()))
        return []

    def reset(self):
        raise AssertionError("ownership must not reset model history")


class _UI(IUILoop[list]):
    def step(self, step_index, events):
        self.state.append((step_index, events.get_events()))
        return []

    def reset(self):
        raise AssertionError("ownership must not reset application history")


def _register(loop):
    loop.register_session_loop_objects(
        state=[],
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    return loop


def _window(monkeypatch):
    pytest.importorskip("aiortc")
    from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
    from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

    monkeypatch.delenv("FLASHDREAMS_WEBRTC_TOKEN_FILE", raising=False)
    monkeypatch.delenv("FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(WebRTCServer, "_run_server", lambda self: self._started.set())
    return WebRTCClientWindow()


@pytest.mark.parametrize("outputless", [False, True])
def test_stalled_model_receives_only_boundary_and_new_owner_input(
    monkeypatch, outputless
):
    window = _window(monkeypatch)
    buffer = EventBuffer()
    buffer.register(0)
    buffer.register(1)
    entered, resume = threading.Event(), threading.Event()
    published = []

    class StalledModel(_Model):
        def step(self, step_index, events):
            super().step(step_index, events)
            if step_index == 0:
                entered.set()
                assert resume.wait(5)
            if outputless:
                return []
            return [
                StepResult(
                    step_index=step_index,
                    output=torch.zeros(1, 3, 1, 1),
                    frame_count=1,
                    output_layout=VideoTensorLayout.tchw,
                )
            ]

    model, ui = _register(StalledModel()), _register(_UI())
    buffer.append(UserInputEvents([_key("held")]))
    worker = threading.Thread(
        target=model._run_model_loop,
        kwargs=dict(
            event_buffer=buffer,
            reader_id=1,
            max_steps=2,
            publish=lambda generation, result, elapsed: published.append(
                (generation, result[0].step_index)
            ),
        ),
    )
    worker.start()
    try:
        assert entered.wait(5)
        # The runtime has already collected this old owner's input while the
        # admitted model step is stalled; clearing only the window cannot help.
        old = [
            _key("old", 900),
            QueryStringUserInputEvent(timestamp=uint64(901), query_string="old=1"),
            NumeralKeypadUserInputEvent(timestamp=uint64(902), value=9),
        ]
        for event in old:
            window.server._input_callback(event)
        buffer.append(window.get_user_input_events())
        window._viewer_changed(None)
        window._viewer_changed(True)
        new = _key("new", 2)
        window.server._input_callback(new)
        buffer.append(window.get_user_input_events())
        events, generation = buffer.read(0)
        run = ui._begin_run(events, generation)
        ui.step(run.step_index, ui.user_events)
        ui._finish_run([], step_completed=True)
        resume.set()
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert model._failure_queue.empty()
    assert [index for index, _ in model.state] == [0, 1]
    for batch in (model.state[1][1], ui.state[0][1]):
        assert len(batch) == 2
        assert isinstance(batch[0], FocusUserInputEvent) and not batch[0].focused
        assert isinstance(batch[0], InputOwnerChangedUserInputEvent)
        assert isinstance(batch[1], KeyboardUserInputEvent) and batch[1].key == "new"
    assert buffer.generation == model._generation == ui._generation == 0
    assert published == ([] if outputless else [(0, 0), (0, 1)])


@pytest.mark.parametrize("loop_type", [_Model, _UI])
def test_latest_handoff_discards_deferred_events_before_application_step(loop_type):
    loop = _register(loop_type())
    loop._pending_user_events = [
        _key("old", 1000),
        QueryStringUserInputEvent(timestamp=uint64(1001), query_string="old=1"),
        NumeralKeypadUserInputEvent(timestamp=uint64(1002), value=9),
    ]
    latest = _boundary(3)
    current = _key("current", 1)
    events = UserInputEvents([_boundary(900), _key("middle", 999), latest, current])
    run = loop._begin_run(events, 0)
    loop.step(run.step_index, loop.user_events)
    assert loop.state == [(0, [latest, current])]


def test_boundary_survives_ui_ticks_without_an_application_step():
    ui = _register(_UI())
    boundary = _boundary()
    ui._begin_run(UserInputEvents([boundary]), 0)
    for _ in range(3):
        ui._finish_run(None, step_completed=False)
        ui._begin_run(UserInputEvents([]), 0)
    new = _key("new")
    run = ui._begin_run(UserInputEvents([new]), 0)
    ui.step(run.step_index, ui.user_events)
    ui._finish_run([], step_completed=True)
    assert ui.state == [(3, [boundary, new])]
    assert ui._pending_user_events == []


def test_handoff_cannot_discard_a_deferred_terminal_close():
    loop = _register(_Model())
    loop._pending_user_events = [CloseUserInputEvent(timestamp=uint64(500))]
    assert loop._begin_run(
        UserInputEvents([_boundary(), _key("new")]), 0
    ).stop_requested


def test_admitted_reset_remains_agreed_after_its_old_input_is_purged():
    buffer = EventBuffer()
    for reader in (0, 1):
        buffer.register(reader)
    buffer.append(
        UserInputEvents([ResetUserInputEvent(timestamp=uint64(1)), _key("old")])
    )
    assert buffer.read(0)[1] == 1
    buffer.append(UserInputEvents([_boundary(), _key("new")]))
    for reader in (0, 1):
        events, generation = buffer.read(reader)
        loop = _register(_Model())
        resets = []
        loop.reset = lambda: resets.append(True)
        run = loop._begin_run(events, generation)
        assert resets == [True] and run.step_index == 0
        assert loop._generation == generation == 1
        assert [type(event) for event in loop.user_events.get_events()] == [
            InputOwnerChangedUserInputEvent,
            KeyboardUserInputEvent,
        ]


@pytest.mark.parametrize("read_first", [False, True])
def test_buffer_purges_retired_input_and_remaps_each_reader_without_reset(read_first):
    buffer = EventBuffer()
    for reader in (0, 1):
        buffer.register(reader)
    close = CloseUserInputEvent(timestamp=uint64(5))
    buffer.append(UserInputEvents([_key("old"), close]), received_ns=10)
    if read_first:
        buffer.read(0)
    first, latest, current = _boundary(), _boundary(), _key("new")
    buffer.append(UserInputEvents([first, _key("middle")]), received_ns=20)
    buffer.read(0)
    buffer.collect_garbage()
    buffer.append(UserInputEvents([latest, current]), received_ns=30)
    fast, generation = buffer.read(0)
    assert fast.get_events() == [latest, current]
    assert buffer.last_read_received_ns(0) == [30, 30]
    slow, generation = buffer.read(1)
    assert slow.get_events() == [close, latest, current]
    assert buffer.last_read_received_ns(1) == [10, 30, 30]
    assert generation == 0
    assert buffer.collect_garbage() == 3
    assert buffer.read(1)[0].get_events() == []


def test_handoff_keeps_already_admitted_reset_generation_and_terminal_close(
    monkeypatch,
):
    window = _window(monkeypatch)
    close = CloseUserInputEvent(timestamp=uint64(10))
    window.server._input_callback(close)
    window._viewer_changed(None)
    window._viewer_changed(True)
    window._viewer_changed(None)
    buffer = EventBuffer()
    buffer.register(0)
    buffer.register(1)
    buffer.append(UserInputEvents([ResetUserInputEvent(timestamp=uint64(1))]))
    assert buffer.read(0)[1] == 1
    buffer.append(window.get_user_input_events())
    for reader in (0, 1):
        events, generation = buffer.read(reader)
        assert generation == 1
        assert isinstance(events.get_events()[0], CloseUserInputEvent)
        loop = _register(_Model())
        assert loop._begin_run(events, generation).stop_requested


def test_step_agreement_transmits_boundary_and_generation_without_local_objects(
    monkeypatch,
):
    from flashdreams.runtime_v2 import coordination

    coordinator = object.__new__(StepAgreement)
    coordinator._ctx = SimpleNamespace(is_main=True)
    coordinator._group = object()
    wire = []

    def broadcast(payload, **kwargs):
        if payload[0] is not None:
            wire.append(pickle.dumps(payload[0]))
        payload[0] = pickle.loads(wire[-1])

    monkeypatch.setattr(coordination.dist, "broadcast_object_list", broadcast)
    events = UserInputEvents([_boundary(), _key("new")])
    leader_events, leader_generation = coordinator.inputs(events, 7)
    coordinator._ctx = SimpleNamespace(is_main=False)
    worker_events, worker_generation = coordinator.inputs(UserInputEvents([]), 0)
    assert leader_generation == worker_generation == 7
    for received in (leader_events, worker_events):
        loop = _register(_Model())
        loop._generation = 7
        loop._pending_user_events = [_key("old")]
        loop._begin_run(received, 7)
        batch = loop.user_events.get_events()
        assert isinstance(batch[0], InputOwnerChangedUserInputEvent)
        assert batch[0].__dataclass_fields__.keys() == {
            "_type_name_owners",
            "timestamp",
            "focused",
        }
        assert batch[1].key == "new"


def test_file_completion_cannot_cross_retirement_boundary(monkeypatch):
    window = _window(monkeypatch)
    window.server._event_origin_ns = time.monotonic_ns()
    window._active_file_selector = (
        1,
        FileSelectionRequest(request_id="old-pick", initial_path="/tmp"),
    )
    window._file_selector_reports.append(
        (1, SelectedFilesStatus.OK, (SelectedFile(name="old.bin", data=b"old"),))
    )
    completing, resume, retired = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = window._complete_file_selector
    collected = []

    def complete(*args):
        completing.set()
        assert resume.wait(5)
        original(*args)

    def retire():
        window._viewer_changed(None)
        retired.set()

    monkeypatch.setattr(window, "_complete_file_selector", complete)
    collector = threading.Thread(
        target=lambda: collected.append(window.get_user_input_events())
    )
    collector.start()
    retiring = threading.Thread(target=retire)
    try:
        assert completing.wait(5)
        locked = window._input_lock.acquire(blocking=False)
        if locked:
            window._input_lock.release()
        assert not locked, "report completion must be atomic with input snapshot"
        retiring.start()
    finally:
        resume.set()
        collector.join(5)
        if retiring.ident is not None:
            retiring.join(5)
    assert not collector.is_alive() and retired.is_set()
    assert isinstance(collected[0].get_events()[0], SelectedFilesUserInputEvent)
    buffer = EventBuffer()
    buffer.register(0)
    buffer.append(collected[0])
    buffer.append(window.get_user_input_events())
    assert [type(event) for event in buffer.read(0)[0].get_events()] == [
        InputOwnerChangedUserInputEvent
    ]
