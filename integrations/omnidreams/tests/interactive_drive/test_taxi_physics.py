# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for the Taxi-only PhysX adapter."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from omnidreams.interactive_drive.crazy_robotaxi.driving import TaxiVehicleConfig
from omnidreams.interactive_drive.crazy_robotaxi.physics import (
    TaxiPhysicsWorld,
    inset_vehicle_chassis,
    select_traffic_tracks,
)
from omnidreams.interactive_drive.simulation.components import (
    rigid_body_model_from_vehicle_config,
)
from omnidreams.interactive_drive.simulation.game_physics import GamePhysicsWorld
from omnidreams.interactive_drive.types import VehicleState

pytestmark = pytest.mark.ci_cpu


def test_taxi_traffic_filter_is_stable_and_keeps_non_motor_actors() -> None:
    tracks = tuple(
        SimpleNamespace(track_id=f"car-{index}", object_type="Car")
        for index in range(10)
    ) + (SimpleNamespace(track_id="person-1", object_type="Pedestrian"),)

    selected = select_traffic_tracks(tracks, 0.4, "scene-a")

    assert selected == select_traffic_tracks(tracks, 0.4, "scene-a")
    assert len([track for track in selected if track.object_type == "Car"]) == 4
    assert tracks[-1] in selected


def test_taxi_chassis_inset_does_not_change_visual_extents() -> None:
    model = rigid_body_model_from_vehicle_config(TaxiVehicleConfig())
    assert model.vehicle is not None

    inset = inset_vehicle_chassis(model)

    assert inset.half_extents_m == model.half_extents_m
    assert inset.vehicle is not None
    assert inset.vehicle.chassis_half_extents_m[0] == pytest.approx(
        model.vehicle.chassis_half_extents_m[0] - 0.16
    )
    assert inset.vehicle.chassis_half_extents_m[1] == pytest.approx(
        model.vehicle.chassis_half_extents_m[1] - 0.16
    )


def test_taxi_physics_keeps_app_heading_after_contact_resolution() -> None:
    incoming = VehicleState(
        x_m=1.0,
        y_m=2.0,
        z_m=0.0,
        yaw_rad=0.75,
        speed_mps=4.0,
        steer_rad=0.2,
        velocity_x_mps=3.0,
        velocity_y_mps=1.0,
        yaw_rate_radps=0.4,
    )
    physx_state = replace(
        incoming,
        x_m=1.2,
        y_m=2.1,
        yaw_rad=-1.0,
        yaw_rate_radps=-2.0,
        velocity_x_mps=2.0,
        velocity_y_mps=-0.5,
    )
    world = object.__new__(TaxiPhysicsWorld)
    with patch.object(GamePhysicsWorld, "step", return_value=(physx_state, ())):
        resolved, _samples = world.step(incoming, timestamp_us=1, dt_s=1.0 / 30.0)

    assert resolved.x_m == pytest.approx(physx_state.x_m)
    assert resolved.y_m == pytest.approx(physx_state.y_m)
    assert resolved.velocity_x_mps == pytest.approx(physx_state.velocity_x_mps)
    assert resolved.velocity_y_mps == pytest.approx(physx_state.velocity_y_mps)
    assert resolved.yaw_rad == pytest.approx(incoming.yaw_rad)
    assert resolved.yaw_rate_radps == pytest.approx(incoming.yaw_rate_radps)
    expected_speed = float(
        np.dot(
            np.asarray([2.0, -0.5]),
            np.asarray([np.cos(incoming.yaw_rad), np.sin(incoming.yaw_rad)]),
        )
    )
    assert resolved.speed_mps == pytest.approx(expected_speed)
