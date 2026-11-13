# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for UI renderer preparation before gameplay clocks start."""

import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from crazy_robotaxi.session import ModelState
from crazy_robotaxi.ui import TaxiHudState
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    state_from_initial_pose,
)
from omnidreams_game_engine.types import TrajectoryChunk

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _UILoop(IUILoop):
    def step(self, step_index, events):
        return []


@contextmanager
def _presentation_context(active):
    active.append(threading.get_ident())
    try:
        yield
    finally:
        active.pop()


def _state(monkeypatch, *, no_ui=False, fail=False):
    calls = []
    active = []
    stop = threading.Event()
    hud = TaxiHudState(width=8, height=4, calibration=None)
    hud.show_hdmap = False

    class Renderer:
        def __init__(self):
            assert active == [threading.get_ident()]
            calls.append("factory")
            if fail:
                raise ValueError("scene upload failed")

        def render_views(self, trajectory, *, include_main):
            assert active == [threading.get_ident()]
            assert len(trajectory.timestamps_us) == 1
            assert trajectory.vehicle_states[0].speed_mps == 0
            calls.append("first_draw")
            return None, SimpleNamespace(is_cuda=True, device="cuda:0")

        def close(self):
            calls.append("renderer_close")

    def synchronized():
        assert active == [threading.get_ident()]
        calls.append("stream_ready")

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=synchronized),
    )
    monkeypatch.setattr(
        "crazy_robotaxi.session.build_world_renderer", lambda *args: Renderer()
    )
    gameplay = SimpleNamespace(
        start=lambda: calls.append("gameplay_start"),
        restore_input=lambda events: calls.append("restore_input"),
    )
    monkeypatch.setattr(
        "crazy_robotaxi.session.build_gameplay_process", lambda **kwargs: gameplay
    )
    ui = _UILoop()
    ui.register_session_loop_objects(
        state=hud, frequency=0, shutdown_event=stop, failure_queue=queue.Queue()
    )
    state = ModelState.__new__(ModelState)
    state.scene = SimpleNamespace(
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_yaw_rad=0.0,
        initial_speed_mps=0.0,
        initial_timestamp_us=123,
    )
    pose = state.scene.initial_rig_to_world
    vehicle = state_from_initial_pose(pose, 0.0, 0.0)
    state.initial_trajectory = TrajectoryChunk(
        timestamps_us=np.asarray([123]),
        rig_poses_world=pose[None],
        vehicle_states=(vehicle,),
        boundary_state_after_chunk=vehicle,
    )
    state.session_desc = SimpleNamespace(metadata={})
    state.rollout = object()
    state.prewarm_complete = True
    state.gameplay = None
    state.shutdown_event = stop
    state.ui_loop = ui
    state.driver_input = object()
    state.config = SimpleNamespace(
        no_ui=no_ui,
        game=None,
        renderer=SimpleNamespace(raster=None, bev=None),
        game_mode="free-roam",
        race_course_id=None,
        race_times_path=None,
        live_edit=None,
        controls=None,
        device="cuda",
    )
    return state, ui, calls, active


def _run(state):
    done = threading.Event()
    errors = []

    def run():
        try:
            state.ensure_rollout()
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, done, errors


def _drain(ui, active):
    with _presentation_context(active):
        ui._begin_run(UserInputEvents([]), 0)


@pytest.mark.parametrize("held_input", [False, True])
def test_scene_and_first_draw_complete_before_gameplay_start(monkeypatch, held_input):
    state, ui, calls, active = _state(monkeypatch)
    if held_input:
        ui.state.consume_input_events(
            UserInputEvents(
                [
                    KeyboardUserInputEvent(
                        timestamp=np.uint64(123),
                        key="w",
                        state=KeyboardInputState.PRESSED,
                    )
                ]
            )
        )
    thread, done, errors = _run(state)
    try:
        # Wait for the actual startup request rather than sleeping through it.
        operation = ui._message_queue.get(timeout=2)
        assert not done.is_set() and calls == []
        ui._message_queue.put(operation)
        _drain(ui, active)
        assert done.wait(2)
        _drain(ui, active)
        assert not errors
        assert calls == ["factory", "first_draw", "stream_ready", "gameplay_start"] + (
            ["restore_input"] if held_input else []
        )
        assert ui.state._world_renderer is not None
        # The post-start attach must retain the already prepared renderer.
        assert ui.state.gameplay is state.gameplay
    finally:
        state.shutdown_event.set()
        thread.join(2)
        assert not thread.is_alive()


def test_ui_preparation_failure_reaches_model_without_starting_gameplay(monkeypatch):
    state, ui, calls, active = _state(monkeypatch, fail=True)
    thread, done, errors = _run(state)
    try:
        operation = ui._message_queue.get(timeout=2)
        ui._message_queue.put(operation)
        _drain(ui, active)
        assert done.wait(2)
        assert len(errors) == 1 and str(errors[0]) == "scene upload failed"
        assert calls == ["factory"]
    finally:
        state.shutdown_event.set()
        thread.join(2)
        assert not thread.is_alive()


def test_shutdown_cancels_pending_ui_startup_without_deadlock(monkeypatch):
    state, ui, calls, active = _state(monkeypatch)
    thread, done, errors = _run(state)
    try:
        operation = ui._message_queue.get(timeout=2)
        state.shutdown_event.set()
        assert done.wait(2)
        with _presentation_context(active):
            operation.operation(ui.state)
        assert not errors and calls == []
    finally:
        state.shutdown_event.set()
        thread.join(2)
        assert not thread.is_alive()


def test_no_ui_starts_without_renderer_or_ui_acknowledgement(monkeypatch):
    state, ui, calls, active = _state(monkeypatch, no_ui=True)
    thread, done, errors = _run(state)
    try:
        assert done.wait(2)
        assert not errors and calls == ["gameplay_start"]
        _drain(ui, active)
        assert ui.state.gameplay is state.gameplay
        assert ui.state._world_renderer is None
    finally:
        state.shutdown_event.set()
        thread.join(2)
        assert not thread.is_alive()


def test_new_renderer_preparation_and_detach_close_previous_resources(monkeypatch):
    state, ui, calls, active = _state(monkeypatch)
    trajectory = state.initial_trajectory
    import crazy_robotaxi.session as session

    def factory():
        return session.build_world_renderer(
            state.scene,
            state.config.renderer.raster,
            state.config.renderer.bev,
            state.config.device,
        )

    with _presentation_context(active):
        ui.state.prepare_world_renderer(factory, trajectory)
        first = ui.state._world_renderer
        ui.state.prepare_world_renderer(factory, trajectory)
        assert ui.state._world_renderer is not first
        ui.state.attach_gameplay(None)
    assert calls == [
        "factory",
        "first_draw",
        "stream_ready",
        "renderer_close",
        "factory",
        "first_draw",
        "stream_ready",
        "renderer_close",
    ]
    assert ui.state._world_renderer is None
