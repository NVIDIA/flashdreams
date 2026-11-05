# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic simulation for application tests using the real gameplay process."""

import time
from dataclasses import replace

import numpy as np
from crazy_robotaxi.controls import ControlsConfig
from crazy_robotaxi.gameplay import TaxiSimulationDriver
from omnidreams_game_engine.engine import SimulationStep
from omnidreams_game_engine.input import DriverInput
from omnidreams_game_engine.types import (
    PhysicsDebugFrame,
    TrajectoryChunk,
    VehicleState,
)


class TestEngine:
    """Small CPU simulation with recorded commands and optional debug geometry."""

    __test__ = False
    frame_interval_s = 1 / 120
    is_running = True

    def __init__(self, snapshot):
        self.current_game_frame = snapshot
        self.simulation = self
        self.state = VehicleState(0, 0, 0, 0, 0, 0)
        self.tick = 0
        self.debug = False

    def set_physx_debug_enabled(self, enabled):
        self.debug = enabled

    def advance(self, commands):
        started = time.monotonic_ns()
        (command,) = commands
        speed = command.throttle * 10
        self.state = replace(
            self.state,
            x_m=self.state.x_m + speed * self.frame_interval_s,
            speed_mps=speed,
        )
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = self.state.x_m
        debug = ()
        if self.debug:
            debug = (
                PhysicsDebugFrame(
                    ego_position_m=np.asarray(
                        [self.state.x_m, 0, 0.8], dtype=np.float32
                    ),
                    ego_orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
                    ego_dimensions_lwh=np.asarray([4.8, 2, 1.6], dtype=np.float32),
                    actor_positions_m=np.empty((0, 3), dtype=np.float32),
                    actor_orientations_xyzw=np.empty((0, 4), dtype=np.float32),
                    actor_dimensions_lwh=np.empty((0, 3), dtype=np.float32),
                    barrier_segments_xy_m=np.empty((0, 2, 2), dtype=np.float32),
                    barrier_thicknesses_m=np.empty(0, dtype=np.float32),
                    barrier_heights_m=np.empty(0, dtype=np.float32),
                    vehicle_state=self.state,
                    actor_collision=False,
                    static_barrier_collision=False,
                    physics_started_ns=started,
                    physics_completed_ns=time.monotonic_ns(),
                ),
            )
        trajectory = TrajectoryChunk(
            timestamps_us=np.array([round(self.tick * self.frame_interval_s * 1e6)]),
            rig_poses_world=pose[None],
            vehicle_states=(self.state,),
            boundary_state_after_chunk=self.state,
            applied_commands=commands,
            physics_debug_frames=debug,
        )
        self.tick += 1
        return SimulationStep(trajectory, (self.current_game_frame,))

    def close(self):
        pass


def simulation(snapshot, driver=None, controls=ControlsConfig(), observe_physics=False):
    """Build a picklable test simulation inside the real gameplay worker."""
    engine = TestEngine(snapshot)
    engine.set_physx_debug_enabled(observe_physics)
    return TaxiSimulationDriver(engine, driver or DriverInput(), controls)
