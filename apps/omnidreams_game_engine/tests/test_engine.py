# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the model-thread game engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from omnidreams_game_engine.config import ChunkConfig, VehicleConfig
from omnidreams_game_engine.contracts import GameUpdate
from omnidreams_game_engine.engine import GameEngine
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    sample_chunk_trajectory,
)
from omnidreams_game_engine.types import (
    ConditionBatch,
    DriverCommand,
    TrajectoryChunk,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def _state(x_m: float = 0.0) -> VehicleState:
    return VehicleState(
        x_m=x_m,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )


@dataclass
class _Simulation:
    current_state: VehicleState = field(default_factory=_state)
    closed: bool = False

    def pose_chunk(self, *, commands, chunk_size, frame_interval_s, **kwargs):
        del kwargs, frame_interval_s
        assert chunk_size == len(commands)
        states = tuple(_state(float(index)) for index in range(chunk_size))
        self.current_state = states[-1]
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], chunk_size, axis=0)
        poses[:, 0, 3] = np.arange(chunk_size, dtype=np.float32)
        return TrajectoryChunk(
            timestamps_us=np.arange(chunk_size, dtype=np.int64),
            rig_poses_world=poses,
            vehicle_states=states,
            boundary_state_after_chunk=states[-1],
            applied_commands=tuple(commands),
        )

    def close(self):
        self.closed = True


class _Rules:
    is_running = True

    def snapshot(self, vehicle_state):
        return ("snapshot", vehicle_state.x_m)

    def advance_frames(self, trajectory, frame_interval_s):
        del frame_interval_s
        return GameUpdate(
            tuple(("frame", state.x_m) for state in trajectory.vehicle_states)
        )

    def submit_text(self, value, vehicle_state):
        return (value, vehicle_state.x_m)


class _Renderer:
    closed = False

    def load_scene(self, scene):
        del scene

    def render(self, trajectory):
        count = len(trajectory.timestamps_us)
        return ConditionBatch(torch.zeros(1, 1, count, 3, 4, 6))

    def close(self):
        self.closed = True


def test_engine_aligns_simulation_rules_and_conditioning() -> None:
    simulation = _Simulation()
    renderer = _Renderer()
    engine = GameEngine(
        simulation=simulation,
        rules=_Rules(),
        condition_renderer=renderer,
        frame_interval_s=1.0 / 30.0,
    )

    result = engine.step((DriverCommand(throttle=1.0), DriverCommand()))

    assert len(result.trajectory.vehicle_states) == 2
    assert result.game_frames == (("frame", 0.0), ("frame", 1.0))
    assert result.condition.hdmap_bvtchw.shape == (1, 1, 2, 3, 4, 6)
    assert all(
        result.metrics[name] >= 0.0
        for name in (
            "simulation_wall_ms",
            "simulation_cpu_ms",
            "rules_wall_ms",
            "rules_cpu_ms",
            "conditioning_wall_ms",
            "conditioning_cpu_ms",
            "engine_step_wall_ms",
            "engine_step_cpu_ms",
        )
    )
    assert engine.current_game_frame == ("snapshot", 1.0)
    assert engine.submit_text("CAB") == ("CAB", 1.0)

    engine.close()
    assert simulation.closed
    assert renderer.closed


def test_engine_rejects_frame_misalignment() -> None:
    class MisalignedRules(_Rules):
        def advance_frames(self, trajectory, frame_interval_s):
            del trajectory, frame_interval_s
            return GameUpdate(("only-one",))

    engine = GameEngine(
        simulation=_Simulation(),
        rules=MisalignedRules(),
        condition_renderer=_Renderer(),
        frame_interval_s=1.0,
    )

    with pytest.raises(ValueError, match="Game frames must align"):
        engine.step((DriverCommand(), DriverCommand()))


def test_trajectory_sampling_copies_slotted_start_state() -> None:
    start_state = _state(12.0)

    trajectory = sample_chunk_trajectory(
        start_state=start_state,
        start_timestamp_us=100,
        commands=(DriverCommand(),),
        chunk_size=1,
        chunk_config=ChunkConfig(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        include_start_state=True,
    )

    assert trajectory.vehicle_states[0] == start_state
    assert trajectory.vehicle_states[0] is not start_state
    assert trajectory.boundary_state_after_chunk is trajectory.vehicle_states[0]
