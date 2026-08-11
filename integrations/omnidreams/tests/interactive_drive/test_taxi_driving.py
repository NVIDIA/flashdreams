# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for Taxi-only driving policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from omnidreams.interactive_drive import cli
from omnidreams.interactive_drive.cli import build_parser
from omnidreams.interactive_drive.config import AppConfig, VehicleConfig
from omnidreams.interactive_drive.crazy_robotaxi.driving import (
    TaxiVehicleConfig,
    integrate_taxi_vehicle,
)
from omnidreams.interactive_drive.crazy_robotaxi.game import TaxiGameConfig
from omnidreams.interactive_drive.input.keyboard import KeyboardState
from omnidreams.interactive_drive.types import DriverCommand, VehicleState

pytestmark = pytest.mark.ci_cpu


def _stopped_state() -> VehicleState:
    return VehicleState(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        speed_mps=0.0,
        steer_rad=0.0,
    )


def test_taxi_config_does_not_enable_base_game_mode() -> None:
    config = AppConfig(
        scene_path=Path("scene.usdz"),
        taxi_game=TaxiGameConfig(enabled=True),
    )

    assert config.game_mode is False
    assert config.vehicle == VehicleConfig(
        speed_limit_enabled=False,
        actor_collision_enabled=False,
        static_collision_enabled=False,
    )
    assert config.taxi_game.vehicle == TaxiVehicleConfig()


def test_taxi_cli_keeps_base_mode_disabled_and_owns_traffic_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "RasterRenderBackend", lambda **_kwargs: object())

    config, _backend = cli.prepare_config_and_backend(
        build_parser().parse_args(["--taxi-game", "--traffic-density", "0.25"])
    )

    assert config.game_mode is False
    assert config.vehicle.actor_collision_enabled is False
    assert config.visual_flare_enabled is False
    assert config.taxi_game.enabled is True
    assert config.taxi_game.traffic_density == pytest.approx(0.25)
    assert config.taxi_game.vehicle.actor_collision_enabled is True


def test_taxi_brake_enters_reverse_while_base_brake_does_not() -> None:
    command = DriverCommand(brake=1.0, manual_control=True)

    taxi_state = integrate_taxi_vehicle(
        _stopped_state(), command, dt_s=0.1, vehicle=TaxiVehicleConfig()
    )
    from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
        integrate_vehicle,
    )

    base_state = integrate_vehicle(
        _stopped_state(), command, dt_s=0.1, vehicle=VehicleConfig()
    )

    assert taxi_state.speed_mps < 0.0
    assert base_state.speed_mps == 0.0


def test_space_remains_upstream_stop_until_taxi_controls_are_enabled() -> None:
    keyboard = KeyboardState()
    keyboard.set_drive_command(
        DriverCommand(throttle=1.0, steer=0.25, manual_control=True)
    )
    keyboard.set_key("space", True)

    base_command = keyboard.command()

    assert base_command.stop is True
    assert base_command.handbrake is False

    keyboard.enable_taxi_controls()
    taxi_command = keyboard.command()

    assert taxi_command.stop is False
    assert taxi_command.handbrake is True


@pytest.mark.parametrize("density", [0.0, -0.1, 1.1])
def test_taxi_config_rejects_invalid_traffic_density(density: float) -> None:
    with pytest.raises(ValueError, match="traffic_density"):
        TaxiGameConfig(traffic_density=density)
