# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
from omnidreams.interactive_drive.config import VehicleConfig
from omnidreams.interactive_drive.input.waypoint_trajectory import (
    Waypoint,
    WaypointTrajectory,
    WaypointTrajectoryInputBackend,
    WaypointTrajectoryPlaybackInputBackend,
)
from omnidreams.interactive_drive.simulation.waypoint_interpolation import (
    InterpolatedWaypointTrajectorySimulation,
)
from omnidreams.interactive_drive.types import DriverCommand, VehicleState


def _initial_state() -> VehicleState:
    return VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )


def test_waypoint_trajectory_backend_reports_finished_at_route_end() -> None:
    state = VehicleState(
        x_m=9.8,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=1.0,
        steer_rad=0.0,
    )
    backend = WaypointTrajectoryInputBackend(
        WaypointTrajectory(
            waypoints=(Waypoint(0.0, 0.0), Waypoint(10.0, 0.0)),
            waypoint_tolerance_m=0.5,
            stop_at_end=True,
        ),
        state_provider=lambda: state,
    )

    assert backend.finished is False
    command = backend.sample().command

    assert backend.finished is True
    assert command.throttle == 0.0
    assert command.brake > 0.0


def test_waypoint_playback_backend_reports_simulation_completion() -> None:
    finished = False
    backend = WaypointTrajectoryPlaybackInputBackend(
        finished_provider=lambda: finished,
        clock=lambda: 12.5,
    )

    sample = backend.sample()

    assert sample.sample_time == 12.5
    assert sample.command == DriverCommand()
    assert backend.finished is False

    finished = True

    assert backend.finished is True


def test_interpolated_waypoint_simulation_samples_position_by_frame() -> None:
    simulation = InterpolatedWaypointTrajectorySimulation(
        initial_state=_initial_state(),
        trajectory=WaypointTrajectory(
            waypoints=(Waypoint(0.0, 0.0), Waypoint(10.0, 0.0)),
            speed_mps=3.0,
        ),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=1_000,
    )

    chunk = simulation.pose_chunk(
        command=DriverCommand(throttle=1.0, steer=1.0),
        chunk_size=4,
        frame_interval_s=1.0 / 30.0,
        extrapolation_offset_s=0.0,
    )

    assert chunk.timestamps_us.tolist() == [1_000, 34_333, 67_666, 100_999]
    assert chunk.rig_poses_world[:, 0, 3].tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.3]
    )
    assert chunk.rig_poses_world[:, 1, 3].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0]
    )
    assert simulation.current_state.x_m == pytest.approx(0.3)
    assert simulation.current_state.speed_mps == pytest.approx(3.0)
    assert simulation.finished is False


def test_interpolated_waypoint_simulation_clamps_and_finishes_at_route_end() -> None:
    simulation = InterpolatedWaypointTrajectorySimulation(
        initial_state=_initial_state(),
        trajectory=WaypointTrajectory(
            waypoints=(Waypoint(0.0, 0.0), Waypoint(1.0, 0.0)),
            speed_mps=30.0,
        ),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=0,
    )

    chunk = simulation.pose_chunk(
        command=DriverCommand(),
        chunk_size=4,
        frame_interval_s=1.0 / 30.0,
        extrapolation_offset_s=0.0,
    )

    assert chunk.rig_poses_world[:, 0, 3].tolist() == pytest.approx(
        [0.0, 1.0, 1.0, 1.0]
    )
    assert simulation.current_state.x_m == pytest.approx(1.0)
    assert simulation.current_state.speed_mps == 0.0
    assert simulation.finished is True


def test_interpolated_waypoint_simulation_uses_route_tangent_yaw() -> None:
    simulation = InterpolatedWaypointTrajectorySimulation(
        initial_state=_initial_state(),
        trajectory=WaypointTrajectory(
            waypoints=(Waypoint(0.0, 0.0), Waypoint(0.0, 10.0)),
            speed_mps=3.0,
        ),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=0,
    )

    simulation.pose_chunk(
        command=DriverCommand(),
        chunk_size=1,
        frame_interval_s=1.0 / 30.0,
        extrapolation_offset_s=0.0,
    )

    assert simulation.current_state.yaw_rad == pytest.approx(math.pi / 2.0)
