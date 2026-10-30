# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gameplay process scheduling and ordinary input dispatch under model stalls."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import json
import sys
import threading
import time

import numpy as np
import pytest

from crazy_robotaxi.debug_input import HeadlessInput
from crazy_robotaxi.gameplay import GameplayProcess
from omnidreams_game_engine.engine import SimulationStep
from omnidreams_game_engine.input import (
    DriverInput,
    _keyboard_command,
    _gamepad_command,
    _wheel_command,
    _normalize_drive_key,
)
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


@dataclass(frozen=True)
class _Snapshot:
    session_state: str = "playing"


class _Simulation:
    def set_physx_debug_enabled(self, value):
        pass


class _Engine:
    def __init__(self, frame_interval_s, **kwargs):
        assert kwargs["render_conditions"] is False
        self.frame_interval_s = frame_interval_s
        self.simulation = _Simulation()
        self.is_running = True
        self.tick = 0
        self.state = VehicleState(0, 0, 0, 0, 0, 0)

    def advance(self, commands):
        command = commands[0]
        self.state = replace(
            self.state, x_m=self.state.x_m + command.throttle * self.frame_interval_s
        )
        step = SimulationStep(
            TrajectoryChunk(
                timestamps_us=np.array(
                    [round(self.tick * self.frame_interval_s * 1e6)]
                ),
                rig_poses_world=np.eye(4, dtype=np.float32)[None],
                vehicle_states=(self.state,),
                boundary_state_after_chunk=self.state,
                applied_commands=commands,
            ),
            (_Snapshot(),),
        )
        self.tick += 1
        return step

    def close(self):
        pass


def _driver():
    return DriverInput(
        keyboard_command=_keyboard_command,
        gamepad_command=_gamepad_command,
        wheel_command=_wheel_command,
        key_normalizer=_normalize_drive_key,
    )


def test_physics_keeps_clock_during_parent_gil_stall_and_consumes_ui_input(
    tmp_path: Path,
):
    control = HeadlessInput()
    control.capture = cast(Any, SimpleNamespace(directory=tmp_path))
    worker = GameplayProcess(
        {"frame_interval_s": 1 / 120}, _driver(), control, 4, engine_factory=_Engine
    )
    previous_interval = sys.getswitchinterval()
    try:
        worker.start()
        control.accept(
            {
                "id": "press",
                "timestamp_us": 100_000,
                "type": "keyboard",
                "key": "w",
                "pressed": True,
            }
        )
        while (control.elapsed_us() or 0) < 100_000:
            time.sleep(0.002)
        # This is the batch the ordinary UI reader dispatches, not an input
        # callback installed directly on the headless scheduler.
        worker.submit_input(UserInputEvents(control.due()))
        time.sleep(0.05)
        sys.setswitchinterval(0.3)
        end = time.monotonic() + 0.65
        while time.monotonic() < end:
            pass
        sys.setswitchinterval(previous_interval)
        time.sleep(0.05)
        worker.check()
        window = worker.timeline.prepare(9, threading.Event())
        assert window is not None
        worker.timeline.commit(window)
        second = worker.timeline.prepare(8, threading.Event())
        assert second is not None
        assert second.end_tick == worker.timeline.gameplay_tick
        assert second.recorded_step().trajectory.vehicle_states[-1].x_m > 0.5
    finally:
        sys.setswitchinterval(previous_interval)
        worker.close()
    rows = [
        json.loads(line)
        for line in (tmp_path / "gameplay-4.jsonl").read_text().splitlines()
    ]
    gaps = np.diff([row["started_ns"] for row in rows]) / 1e9
    sim = (
        rows[-1]["simulation_timestamp_us"] - rows[0]["simulation_timestamp_us"]
    ) / 1e6
    wall = (rows[-1]["started_ns"] - rows[0]["started_ns"]) / 1e9
    assert 0.95 < sim / wall < 1.05
    assert np.percentile(gaps, 95) < 0.04
    applied = next(row for row in rows if row["transition_timestamp_us"] == 100_000)
    assert (applied["started_ns"] - control.ready_ns) / 1e6 - 100 < 100
    messages = []
    while not control.messages.empty():
        messages.append(control.messages.get()[1])
    assert any(
        m.get("status") == "consumed" and m.get("consumer") == "gameplay"
        for m in messages
    )
    assert not worker._process.is_alive()


def _broken_engine(**kwargs):
    raise ValueError("test physics initialization failure")


def test_process_failure_is_reported_and_can_be_closed():
    worker = GameplayProcess(
        {"frame_interval_s": 1 / 120}, _driver(), None, 1, engine_factory=_broken_engine
    )
    try:
        with pytest.raises(RuntimeError, match="test physics initialization failure"):
            worker.start()
    finally:
        worker.close()


def test_stationary_world_retains_image_and_ar_cursor_until_motion(monkeypatch):
    import queue
    from unittest.mock import Mock

    import torch
    from crazy_robotaxi.controls import ControlsConfig
    from crazy_robotaxi.free_roam import FreeRoamSnapshot
    from crazy_robotaxi.live_edit.config import LiveEditConfig
    from crazy_robotaxi.session import (
        CrazyRobotaxiModelLoop,
        ModelState,
        _stationary_world,
    )
    from omnidreams_game_engine.timeline import WorldSample, WorldTimeline
    from flashdreams.runtime_v2.session_desc import SessionDesc

    snapshot = FreeRoamSnapshot()
    vehicle = VehicleState(0, 0, 0, 0, 0, 0)
    path = TrajectoryChunk(
        np.array([0]), np.eye(4, dtype=np.float32)[None], (vehicle,), vehicle
    )
    timeline = WorldTimeline()
    for tick in range(33):
        timeline.append(
            WorldSample(
                SimulationStep(
                    replace(path, timestamps_us=np.array([tick * 8333])), (snapshot,)
                )
            )
        )
    rollout = SimpleNamespace(
        engine=SimpleNamespace(current_game_frame=snapshot),
        scene=SimpleNamespace(prompt="A road."),
        frame_count=lambda index: 8,
        step=Mock(
            side_effect=AssertionError("Advanced model while world was stationary")
        ),
    )
    monkeypatch.setattr(ModelState, "ensure_rollout", lambda self: rollout)
    published = []
    config = SimpleNamespace(
        controls=ControlsConfig(),
        realtime_gameplay=True,
        debug_capture=False,
        profile_input_latency=False,
        live_edit=LiveEditConfig(),
        total_blocks=None,
        pipeline_profiling=False,
        no_ui=False,
    )
    state = ModelState(
        pipeline_factory=lambda: None,
        scene_factory=cast(Any, lambda *args: None),
        config=cast(Any, config),
        session_desc=SessionDesc(video_width=8, video_height=4),
        driver_input=_driver(),
        ui_loop=cast(Any, SimpleNamespace(_invoke_async=published.append)),
    )
    state.game_selected = True
    state.gameplay = cast(Any, SimpleNamespace(timeline=timeline, check=lambda: None))
    state.last_video = torch.ones(1, 3, 4, 8)
    state.last_hdmap = torch.zeros(1, 3, 4, 8)
    state.last_pose = np.eye(4, dtype=np.float32)
    state.blocks_generated = 3
    loop = CrazyRobotaxiModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=30,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    result = loop.step(0, UserInputEvents([]))
    assert state.blocks_generated == 3 and not rollout.step.called
    assert torch.equal(
        result[0].read_output(), torch.ones_like(result[0].read_output())
    )
    assert result[0].metrics is not None
    assert result[0].metrics["reused_video"] == 1
    assert timeline.generation_tick > 0
    moving = replace(path, vehicle_states=(replace(vehicle, speed_mps=1),))
    assert not _stationary_world(moving, state.last_pose)
    shifted = path.rig_poses_world.copy()
    shifted[0, 0, 3] = 0.1
    assert not _stationary_world(
        replace(path, rig_poses_world=shifted), state.last_pose
    )


def test_lagging_physics_applies_wall_clock_inputs_and_acknowledges_each_batch_part(
    monkeypatch,
):
    """Slow simulation cannot defer due keys or acknowledge future batch members."""
    from collections import deque
    from queue import Queue
    from unittest.mock import Mock

    import crazy_robotaxi.gameplay as gameplay
    from crazy_robotaxi.debug_input import control_event

    clock = [0.0]
    stopped = threading.Event()
    monkeypatch.setattr(gameplay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gameplay.time, "monotonic_ns", lambda: round(clock[0] * 1e9))

    def key(identifier, timestamp, name, pressed):
        return control_event(
            dict(
                id=identifier,
                timestamp_us=timestamp,
                type="keyboard",
                key=name,
                pressed=pressed,
            )
        )

    first = UserInputEvents(
        [key("press", 50_000, "w", True), key("release", 350_000, "w", False)]
    )
    second = UserInputEvents(
        [key("turn", 350_000, "a", True), key("straight", 650_000, "a", False)]
    )
    incoming = deque(
        [("start", 0.0, 0.0), ("input", 4, first, 0.0), ("input", 5, second, 0.0)]
    )
    inputs = SimpleNamespace(poll=lambda *args: bool(incoming), recv=incoming.popleft)
    outputs = Queue()
    commands = []

    class SlowEngine(_Engine):
        def advance(self, batch):
            commands.append(batch[0])
            result = super().advance(batch)
            clock[0] += 0.2  # Each fixed 8.3 ms physics step costs 200 ms wall time.
            if self.tick == 5:
                stopped.set()
            return result

    gameplay._run_gameplay(
        {"frame_interval_s": 1 / 120},
        _driver(),
        inputs,
        outputs,
        stopped,
        None,
        engine_factory=SlowEngine,
    )
    messages = list(outputs.queue)
    assert not any(message[0] == "error" for message in messages)
    assert [command.throttle for command in commands] == [0, 1, 0, 0, 0]
    assert [command.steer for command in commands] == [0, 0, 1, 1, 0]
    acknowledgments = [message for message in messages if message[0] == "consumed"]
    assert acknowledgments == [
        ("consumed", 4, [0], 200_000_000, 1),
        ("consumed", 4, [1], 400_000_000, 2),
        ("consumed", 5, [0], 400_000_000, 2),
        ("consumed", 5, [1], 800_000_000, 4),
    ]
    # The bridge acknowledges original parent-side event objects, not deserialized
    # worker copies, and retains future members of a partially consumed batch.
    bridge = object.__new__(GameplayProcess)
    bridge._outputs = Queue()
    bridge._input_lock = threading.Lock()
    bridge._pending_inputs = {
        4: dict(enumerate(first.get_events())),
        5: dict(enumerate(second.get_events())),
    }
    bridge._debug = Mock()
    for message in acknowledgments:
        bridge._outputs.put(message)
    bridge._outputs.put(("closed",))
    bridge._receive()
    assert bridge._pending_inputs == {}
    calls = bridge._debug.consumed.call_args_list
    assert all(
        call.args[0].get_events()[0] is original
        for call, original in zip(
            calls, [*first.get_events(), *second.get_events()], strict=True
        )
    )
    assert [call.kwargs["consumed_ns"] for call in calls] == [
        200_000_000,
        400_000_000,
        400_000_000,
        800_000_000,
    ]
