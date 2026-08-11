# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import pytest
from omnidreams.interactive_drive.config import AppConfig, VehicleConfig

pytestmark = pytest.mark.ci_cpu


def test_app_config_defaults_to_non_game_mode() -> None:
    config = AppConfig(scene_path=Path("scene.usdz"))

    assert config.game_mode is False
    assert config.vehicle.max_speed_mps == pytest.approx(70.0 * 0.44704)
    assert config.vehicle.speed_limit_enabled is False
    assert config.vehicle.actor_collision_enabled is False
    assert config.vehicle.static_collision_enabled is False
    assert config.vehicle.traffic_density == pytest.approx(1.0)
    assert config.visual_flare_enabled is False


def test_game_mode_enables_physics_without_visual_flare() -> None:
    config = AppConfig(scene_path=Path("scene.usdz"), game_mode=True)

    assert config.vehicle.speed_limit_enabled is True
    assert config.vehicle.actor_collision_enabled is True
    assert config.vehicle.static_collision_enabled is True
    assert config.vehicle.traffic_density == pytest.approx(0.4)
    assert config.visual_flare_enabled is False


def test_game_mode_allows_traffic_density_override() -> None:
    config = AppConfig(
        scene_path=Path("scene.usdz"), game_mode=True, game_traffic_density=0.25
    )

    assert config.vehicle.traffic_density == pytest.approx(0.25)


@pytest.mark.parametrize("density", [0.0, -0.1, 1.1])
def test_app_config_rejects_invalid_game_traffic_density(density: float) -> None:
    with pytest.raises(ValueError, match="game_traffic_density"):
        AppConfig(scene_path=Path("scene.usdz"), game_traffic_density=density)


def test_game_mode_allows_visual_flare_override() -> None:
    config = AppConfig(
        scene_path=Path("scene.usdz"),
        game_mode=True,
        visual_flare_enabled=False,
    )

    assert config.visual_flare_enabled is False


def test_app_config_requires_game_mode_for_mode_controlled_physics() -> None:
    config = AppConfig(
        scene_path=Path("scene.usdz"),
        vehicle=VehicleConfig(actor_collision_enabled=True),
    )

    assert config.game_mode is False
    assert config.vehicle.speed_limit_enabled is False
    assert config.vehicle.actor_collision_enabled is False
    assert config.vehicle.static_collision_enabled is False
    assert config.visual_flare_enabled is False
