# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for slope-relative physical roll limits using synthetic road planes."""

import math
from dataclasses import replace

import numpy as np
import pytest
from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.physics import constrain_grounded_roll
from omnidreams_game_engine.math3d import rig_pose_from_state
from omnidreams_game_engine.types import VehicleState

pytestmark = pytest.mark.ci_cpu


def _body_center(state, vehicle):
    pose = rig_pose_from_state(
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        yaw_rad=state.yaw_rad,
        pitch_rad=state.pitch_rad,
        roll_rad=state.roll_rad,
    )
    return pose[:3, 3] + pose[:3, 2] * vehicle.aabb_height_m / 2


@pytest.mark.parametrize("yaw", [0, 0.7, math.pi / 2])
@pytest.mark.parametrize("grade", [-0.15, 0, 0.15])
@pytest.mark.parametrize("side", [-1, 1])
def test_grounded_roll_is_limited_relative_to_road_without_moving_body_center(
    yaw, grade, side
):
    normal = np.asarray([-grade, -0.1, 1.0])
    normal /= np.linalg.norm(normal)
    cy, sy = math.cos(yaw), math.sin(yaw)
    road_roll = -math.asin(-sy * normal[0] + cy * normal[1])
    vehicle = TaxiVehicleConfig()
    state = VehicleState(
        4,
        8,
        2.05,
        yaw,
        10,
        0,
        pitch_rad=math.atan2(cy * normal[0] + sy * normal[1], normal[2]),
        roll_rad=road_roll + side * 0.5,
        angular_velocity_x_radps=side * cy,
        angular_velocity_y_radps=side * sy,
        yaw_rate_radps=0.3,
    )
    result = constrain_grounded_roll(state, vehicle, (2, normal))
    assert result.roll_rad == pytest.approx(
        road_roll + side * vehicle.max_body_roll_rad
    )
    assert result.angular_velocity_x_radps == pytest.approx(0)
    assert result.angular_velocity_y_radps == pytest.approx(0)
    assert result.pitch_rad == state.pitch_rad
    assert result.yaw_rad == state.yaw_rad
    assert result.yaw_rate_radps == state.yaw_rate_radps
    assert result.speed_mps == state.speed_mps
    np.testing.assert_allclose(
        _body_center(result, vehicle), _body_center(state, vehicle), atol=1e-6
    )


def test_roll_limit_leaves_airborne_motion_and_inward_rotation_free():
    vehicle = TaxiVehicleConfig()
    state = VehicleState(0, 0, 0, 0, 0, 0, roll_rad=0.5, angular_velocity_x_radps=-0.4)
    surface = (0.0, np.asarray([0.0, 0.0, 1.0]))
    assert constrain_grounded_roll(state, vehicle, None) is state
    airborne = replace(state, z_m=vehicle.suspension_travel_m + 0.01)
    assert constrain_grounded_roll(airborne, vehicle, surface) is airborne
    within_limit = replace(state, roll_rad=vehicle.max_body_roll_rad / 2)
    assert constrain_grounded_roll(within_limit, vehicle, surface) is within_limit
    assert (
        constrain_grounded_roll(state, vehicle, surface).angular_velocity_x_radps
        == -0.4
    )


def test_contact_roll_limit_is_published_to_the_physical_world(monkeypatch):
    from crazy_robotaxi.physics import TaxiPhysicsWorld
    from omnidreams_game_engine.simulation.game_physics import GamePhysicsWorld
    from omnidreams_game_engine.types import DriverCommand

    state = VehicleState(0, 0, 0, 0, 0, 0, roll_rad=0.5, ragdoll_active=True)
    world = object.__new__(TaxiPhysicsWorld)
    world._taxi_vehicle = TaxiVehicleConfig()
    world.last_step_ground_surface = (0.0, np.asarray([0.0, 0.0, 1.0]))
    published = []
    monkeypatch.setattr(GamePhysicsWorld, "step", lambda *args: (state, ()))
    monkeypatch.setattr(world, "synchronize_ego_state", published.append)
    result, _ = world.step_with_command(state, DriverCommand(), 0, 1 / 120)
    assert published == [result]
    assert result.roll_rad == pytest.approx(world._taxi_vehicle.max_body_roll_rad)
