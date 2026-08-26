# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for Taxi-only driving policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from crazy_robotaxi import runtime_cli as cli
from crazy_robotaxi.app import (
    CrazyRobotaxiApplication,
    taxi_config_from_args,
)
from crazy_robotaxi.driving import (
    TaxiVehicleConfig,
    integrate_taxi_vehicle,
)
from crazy_robotaxi.game import TaxiGameConfig
from crazy_robotaxi.input import (
    CrazyRobotaxiKeyboardState,
)
from crazy_robotaxi.live_edit.config import LiveEditConfig, LiveEditObstacleConfig
from crazy_robotaxi.runtime_cli import build_parser
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.input.keyboard import KeyboardState
from omnidreams_game_engine.types import DriverCommand, VehicleState

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
    config = TaxiGameConfig(enabled=True)

    assert config.vehicle == TaxiVehicleConfig()


def test_taxi_cli_keeps_base_mode_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "RasterRenderBackend", lambda **_kwargs: object())

    config, _backend = cli.prepare_config_and_backend(
        args := build_parser().parse_args(
            [
                "--map",
                "city.robotaxi.yaml",
                "--taxi-game",
            ]
        )
    )
    taxi_config = taxi_config_from_args(args)

    assert config.game_mode is False
    assert config.vehicle.actor_collision_enabled is False
    assert config.visual_flare_enabled is False
    assert taxi_config.enabled is True
    assert taxi_config.vehicle.actor_collision_enabled is True


def test_taxi_alignment_diagnostics_forces_physics_snapshots() -> None:
    args = build_parser().parse_args(
        ["--taxi-game", "--taxi-alignment-diagnostics", "diagnostics"]
    )

    taxi_config = taxi_config_from_args(args)

    assert taxi_config.alignment_diagnostics_enabled is True


def test_taxi_rollout_aligns_model_frame_zero_with_initial_pose() -> None:
    config = TaxiGameConfig(enabled=True)
    application = CrazyRobotaxiApplication(
        config,
        CrazyRobotaxiKeyboardState(),
        presenter_config=None,
    )

    rollout = application.rollout_spec(
        object(),  # type: ignore[arg-type]
        default_vehicle=VehicleConfig(),
        default_visual_flare_enabled=True,
    )

    assert rollout.include_initial_state_in_first_chunk is True


def test_physical_obstacle_controller_is_shared_with_physics_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_physics_world(scene, vehicle, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("crazy_robotaxi.app.TaxiPhysicsWorld", fake_physics_world)
    application = CrazyRobotaxiApplication(
        TaxiGameConfig(enabled=True),
        CrazyRobotaxiKeyboardState(),
        presenter_config=None,
        live_edit=LiveEditConfig(
            obstacle=LiveEditObstacleConfig(enabled=True, physics=True)
        ),
    )
    scene = SimpleNamespace(
        game_map=SimpleNamespace(lanes=()), ground_mesh_vertices=None
    )
    rollout = application.rollout_spec(
        scene,
        default_vehicle=VehicleConfig(),
        default_visual_flare_enabled=True,
    )

    rollout.physics_world_factory(scene, TaxiVehicleConfig())

    actor_controllers = captured["actor_controllers"]
    assert isinstance(actor_controllers, tuple)
    (physics_controller,) = actor_controllers
    assert physics_controller is application._obstacle_ability


def test_taxi_brake_enters_reverse_while_base_brake_does_not() -> None:
    command = DriverCommand(brake=1.0, manual_control=True)

    taxi_state = integrate_taxi_vehicle(
        _stopped_state(), command, dt_s=0.1, vehicle=TaxiVehicleConfig()
    )
    from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
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

    taxi_keyboard = CrazyRobotaxiKeyboardState()
    taxi_keyboard.set_drive_command(
        DriverCommand(throttle=1.0, steer=0.25, manual_control=True)
    )
    taxi_keyboard.set_key("space", True)
    taxi_command = taxi_keyboard.command()

    assert taxi_command.stop is False
    assert taxi_command.handbrake is True
