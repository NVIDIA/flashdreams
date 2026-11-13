# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gameplay process scheduling and ordinary input dispatch under model stalls."""

import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from omnidreams_game_engine.engine import SimulationStep
from omnidreams_game_engine.gameplay import GameplayProcess, SimulationDriver
from omnidreams_game_engine.input import (
    DriverInput,
    _gamepad_command,
    _keyboard_command,
    _normalize_drive_key,
    _wheel_command,
)
from omnidreams_game_engine.types import TrajectoryChunk, VehicleState

from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
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


def _simulation(factory=_Engine):
    return SimulationDriver(
        cast(Any, factory(frame_interval_s=1 / 120, render_conditions=False)), _driver()
    )


def key(identifier, timestamp, name, pressed):
    return KeyboardUserInputEvent(
        timestamp=np.uint64(timestamp),
        key=name,
        state=KeyboardInputState.PRESSED if pressed else KeyboardInputState.RELEASED,
    )


class _Observer:
    def __init__(self):
        self.rows = []
        self.acknowledged = []

    def start(self):
        self.ready_ns = time.monotonic_ns()
        return self.ready_ns / 1e9

    def consumed(self, events, **timing):
        self.acknowledged.append((events, timing))

    def sample(self, sample, timing):
        self.rows.append(
            dict(
                asdict(timing),
                simulation_timestamp_us=int(sample.step.trajectory.timestamps_us[0]),
                transition_timestamp_us=sample.transition_timestamp_us,
            )
        )


def test_physics_keeps_clock_during_parent_gil_stall_and_consumes_ui_input():
    control = _Observer()
    worker = GameplayProcess(_simulation, control)
    previous_interval = sys.getswitchinterval()
    try:
        worker.start()
        time.sleep(0.1)
        worker.submit_input(UserInputEvents([key("press", 100_000, "w", True)]))
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
    rows = control.rows
    gaps = np.diff([row["started_ns"] for row in rows]) / 1e9
    sim = (
        rows[-1]["simulation_timestamp_us"] - rows[0]["simulation_timestamp_us"]
    ) / 1e6
    wall = (rows[-1]["started_ns"] - rows[0]["started_ns"]) / 1e9
    assert 0.95 < sim / wall < 1.05
    assert np.percentile(gaps, 95) < 0.04
    applied = next(row for row in rows if row["transition_timestamp_us"] == 100_000)
    assert (applied["started_ns"] - control.ready_ns) / 1e6 - 100 < 100
    assert len(control.acknowledged) == 1
    assert control.acknowledged[0][0].get_events()[0].key == "w"
    assert not worker._process.is_alive()


def _broken_engine(**kwargs):
    raise ValueError("test physics initialization failure")


@pytest.mark.parametrize("input_origin", [None, 100.0])
def test_delayed_first_input_does_not_delay_subsequent_live_controls(
    monkeypatch, input_origin
):
    """Apply a fresh release immediately after a delayed initial press."""
    from collections import deque
    from queue import Queue

    import omnidreams_game_engine.gameplay as gameplay

    clock = [100.2]
    stopped = threading.Event()
    monkeypatch.setattr(gameplay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gameplay.time, "monotonic_ns", lambda: round(clock[0] * 1e9))
    incoming = deque(
        [
            ("start", 100.2, input_origin),
            ("input", 0, UserInputEvents([key("press", 100_000, "w", True)]), 100.2),
        ]
    )
    inputs = SimpleNamespace(poll=lambda *args: bool(incoming), recv=incoming.popleft)
    outputs = Queue()
    commands = []

    class Engine(_Engine):
        def advance(self, batch):
            commands.append(batch[0])
            result = super().advance(batch)
            clock[0] += 0.02
            if self.tick == 1:
                incoming.append(
                    (
                        "input",
                        1,
                        UserInputEvents([key("release", 220_000, "w", False)]),
                        clock[0],
                    )
                )
            if self.tick == 3:
                stopped.set()
            return result

    gameplay._run_gameplay(
        partial(_simulation, Engine), inputs, outputs, stopped, False
    )
    assert not any(message[0] == "error" for message in outputs.queue)
    assert [command.throttle for command in commands] == [1, 0, 0]


def test_process_failure_is_reported_and_can_be_closed():
    worker = GameplayProcess(partial(_simulation, _broken_engine))
    try:
        with pytest.raises(RuntimeError, match="test physics initialization failure"):
            worker.start()
    finally:
        worker.close()


def test_lagging_physics_applies_wall_clock_inputs_and_acknowledges_each_batch_part(
    monkeypatch,
):
    """Slow simulation cannot defer due keys or acknowledge future batch members."""
    from collections import deque
    from queue import Queue
    from unittest.mock import Mock

    import omnidreams_game_engine.gameplay as gameplay

    clock = [0.0]
    stopped = threading.Event()
    monkeypatch.setattr(gameplay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gameplay.time, "monotonic_ns", lambda: round(clock[0] * 1e9))

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
        partial(_simulation, SlowEngine), inputs, outputs, stopped, False
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
    bridge._observer = Mock()
    for message in acknowledgments:
        bridge._outputs.put(message)
    bridge._outputs.put(("closed",))
    bridge._receive()
    assert bridge._pending_inputs == {}
    calls = bridge._observer.consumed.call_args_list
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


def test_worker_restores_held_driving_state():
    worker = GameplayProcess(_simulation)
    try:
        worker.start()
        worker.restore_input(UserInputEvents([key("held", 0, "w", True)]))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            worker.check()
            latest = worker.timeline.latest
            if (
                latest is not None
                and latest.step.trajectory.vehicle_states[-1].x_m > 0.02
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Restored throttle did not reach gameplay")
    finally:
        worker.close()
