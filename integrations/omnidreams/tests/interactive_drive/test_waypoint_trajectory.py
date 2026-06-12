# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams.interactive_drive.input.waypoint_trajectory import (
    Waypoint,
    WaypointTrajectory,
    WaypointTrajectoryInputBackend,
)
from omnidreams.interactive_drive.types import VehicleState


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
