# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live physics clock must survive a stalled generation consumer."""

import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from crazy_robotaxi.controls import ControlsConfig
from crazy_robotaxi.debug_input import HeadlessInput
from crazy_robotaxi.free_roam import FreeRoamGameRules
from crazy_robotaxi.gameplay import RealtimeGameplay
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.engine import GameEngine
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
)
from omnidreams_game_engine.types import VehicleState

from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_physics_consumes_timed_press_release_while_generation_is_blocked():
    stop = threading.Event()
    debug = HeadlessInput()
    integrated = []
    closed = []

    def integrate(state, command, dt, config):
        integrated.append((command, dt, threading.current_thread().name))
        return replace(state, x_m=state.x_m + command.throttle * dt)

    def factory():
        simulation = EgoVehicleKinematics(
            initial_state=VehicleState(0, 0, 0, 0, 0, 0),
            vehicle_config=VehicleConfig(),
            ground_snapper=None,
            initial_timestamp_us=0,
            integrate_fn=integrate,
            include_initial_state_in_first_chunk=True,
        )
        original_close = simulation.close

        def close():
            closed.append(threading.current_thread().name)
            original_close()

        simulation.close = close
        return GameEngine(
            simulation=simulation,
            rules=FreeRoamGameRules(),
            condition_renderer=None,
            frame_interval_s=1 / 30,
            physics_frame_interval_s=1 / 120,
        )

    driver = DriverInput()
    gameplay = RealtimeGameplay(
        engine_factory=factory,
        driver=driver,
        controls=ControlsConfig(),
        shutdown=stop,
        debug=debug,
    )
    try:
        first = gameplay.timeline.prepare(1, stop)
        assert first is not None
        # Feed a future press/release pair before either deadline, as a file can.
        gameplay.submit_input(
            UserInputEvents(
                [
                    KeyboardUserInputEvent(
                        timestamp=np.uint64(50_000),
                        key="w",
                        state=KeyboardInputState.PRESSED,
                    ),
                    KeyboardUserInputEvent(
                        timestamp=np.uint64(150_000),
                        key="w",
                        state=KeyboardInputState.RELEASED,
                    ),
                ]
            )
        )
        deadline = time.monotonic() + 2
        while gameplay.timeline.gameplay_tick < 30 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert gameplay.timeline.gameplay_tick >= 30
        assert gameplay.timeline.generation_tick == -1
        # The first submitted chunk is unchanged despite 30 later physics ticks.
        assert first.recorded_step().trajectory.vehicle_states[0].x_m == 0
        gameplay.timeline.commit(first)
        chunk = gameplay.timeline.prepare(7, stop)
        assert chunk is not None
        step = chunk.recorded_step()
        assert step.trajectory.vehicle_states[-1].x_m == pytest.approx(0.1, abs=1 / 120)
        assert all(
            dt == 1 / 120 and owner == "robotaxi-gameplay"
            for _, dt, owner in integrated
        )
        assert integrated[-1][0].throttle == 0
        assert not driver._pending_transitions
        gameplay.timeline.commit(chunk)
    finally:
        gameplay.close()
    assert closed == ["robotaxi-gameplay"]
    count = len(integrated)
    stop.wait(0.02)
    assert len(integrated) == count
