# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from interactive_drive.config import VehicleConfig
from interactive_drive.simulation.ego_vehicle_kinematics import EgoVehicleKinematics
from interactive_drive.types import DriverCommand, VehicleState


def _initial_state() -> VehicleState:
    return VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )


def test_pose_chunk_rejects_nonzero_extrapolation_for_stage_one() -> None:
    simulation = EgoVehicleKinematics(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=0,
    )
    with pytest.raises(NotImplementedError):
        simulation.pose_chunk(
            command=DriverCommand(),
            chunk_size=4,
            frame_interval_s=1.0 / 30.0,
            extrapolation_offset_s=0.1,
        )


def test_pose_chunk_advances_state_to_chunk_boundary() -> None:
    """Sim advances by ``chunk_size * frame_interval_s`` per chunk request.

    This is the contract that keeps sim wall-clock cadence tied to display
    cadence rather than poll-loop cadence: the loop calls ``pose_chunk``
    once per chunk it needs, and authoritative state moves forward by
    exactly that chunk's worth of integration.
    """
    simulation = EgoVehicleKinematics(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=0,
    )
    chunk = simulation.pose_chunk(
        command=DriverCommand(throttle=1.0),
        chunk_size=4,
        frame_interval_s=1.0 / 30.0,
        extrapolation_offset_s=0.0,
    )
    assert simulation.current_state == chunk.boundary_state_after_chunk
    assert simulation.current_state.speed_mps > 0.0


def test_pose_chunk_chains_across_calls() -> None:
    """Successive ``pose_chunk`` calls start from the previous boundary state.

    Concretely: requesting two back-to-back chunks must produce the same
    final state as requesting one chunk twice as long. If state didn't
    persist between calls, sim time would silently rewind every chunk
    request.
    """
    chunk_size = 3
    frame_interval_s = 1.0 / 30.0
    a = EgoVehicleKinematics(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=0,
    )
    a.pose_chunk(
        command=DriverCommand(throttle=1.0),
        chunk_size=chunk_size,
        frame_interval_s=frame_interval_s,
        extrapolation_offset_s=0.0,
    )
    a.pose_chunk(
        command=DriverCommand(throttle=1.0),
        chunk_size=chunk_size,
        frame_interval_s=frame_interval_s,
        extrapolation_offset_s=0.0,
    )

    b = EgoVehicleKinematics(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
        initial_timestamp_us=0,
    )
    b.pose_chunk(
        command=DriverCommand(throttle=1.0),
        chunk_size=chunk_size * 2,
        frame_interval_s=frame_interval_s,
        extrapolation_offset_s=0.0,
    )

    assert a.current_state == b.current_state
