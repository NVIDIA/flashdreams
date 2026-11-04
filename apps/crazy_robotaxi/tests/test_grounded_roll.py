# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxi activation of the engine ground-relative roll constraint."""

import numpy as np
import pytest
from crazy_robotaxi.dynamics import TaxiVehicleConfig
from omnidreams_game_engine.types import VehicleState

pytestmark = pytest.mark.ci_cpu


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
