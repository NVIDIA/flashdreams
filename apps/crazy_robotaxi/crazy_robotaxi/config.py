# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict Crazy Robotaxi gameplay configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from omnidreams_game_engine.config import DriverInputConfig
from omnidreams_game_engine.yaml_config import (
    StrictConfigError,
    load_yaml_mapping,
    require_bool,
    require_exact_keys,
    require_float,
    require_mapping,
    require_version,
)

from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.rules import TaxiGameConfig

_RUNTIME_GAME_FIELDS = {
    "seed",
    "high_scores_path",
}
_RULE_FIELDS = (
    {field.name for field in fields(TaxiGameConfig)}
    - _RUNTIME_GAME_FIELDS
    - {"vehicle"}
)
_VEHICLE_FIELDS = {field.name for field in fields(TaxiVehicleConfig)}
_VEHICLE_BOOL_FIELDS = {
    "speed_limit_enabled",
    "actor_collision_enabled",
    "static_collision_enabled",
}
_INPUT_FIELDS = {field.name for field in fields(DriverInputConfig)}


@dataclass(frozen=True, slots=True)
class TaxiSettings:
    """Game rules, driver input, and vehicle dynamics loaded together."""

    game: TaxiGameConfig
    driver_input: DriverInputConfig


def load_game_settings(path: Path) -> TaxiSettings:
    """Load a complete Crazy Robotaxi game YAML document.

    Args:
        path: Game YAML path.

    Returns:
        Validated game rules, V2 input reduction, and taxi dynamics.
    """
    doc = load_yaml_mapping(path)
    require_exact_keys(
        doc,
        {"schema_version", "rules", "input", "vehicle"},
        "game",
    )
    require_version(doc, "game")
    raw_rules = require_mapping(doc["rules"], "game.rules")
    require_exact_keys(raw_rules, _RULE_FIELDS, "game.rules")
    rules = {
        name: require_float(raw_rules[name], f"game.rules.{name}", minimum=0.0)
        for name in _RULE_FIELDS
    }
    for integer_name in ("base_fare_points", "bonus_points_per_second"):
        value = raw_rules[integer_name]
        if type(value) is not int or value < 0:
            raise StrictConfigError(
                f"game.rules.{integer_name} must be a nonnegative integer"
            )
        rules[integer_name] = value
    if rules["fare_min_route_distance_m"] > rules["fare_max_route_distance_m"]:
        raise StrictConfigError(
            "game.rules.fare_min_route_distance_m must not exceed fare_max_route_distance_m"
        )
    if rules["min_time_s"] > rules["max_time_s"]:
        raise StrictConfigError("game.rules.min_time_s must not exceed max_time_s")

    raw_input = require_mapping(doc["input"], "game.input")
    require_exact_keys(raw_input, _INPUT_FIELDS, "game.input")
    input_values = {
        name: require_float(raw_input[name], f"game.input.{name}", minimum=0.0)
        for name in _INPUT_FIELDS
    }
    if input_values["steering_scale"] <= 0.0:
        raise StrictConfigError("game.input.steering_scale must be positive")

    raw_vehicle = require_mapping(doc["vehicle"], "game.vehicle")
    require_exact_keys(raw_vehicle, _VEHICLE_FIELDS, "game.vehicle")
    vehicle_values = {
        name: require_bool(raw_vehicle[name], f"game.vehicle.{name}")
        if name in _VEHICLE_BOOL_FIELDS
        else require_float(raw_vehicle[name], f"game.vehicle.{name}", minimum=0.0)
        for name in _VEHICLE_FIELDS
    }
    for name in (
        "wheel_base_m",
        "max_speed_mps",
        "aabb_length_m",
        "aabb_width_m",
        "aabb_height_m",
        "speed_taper_knee_fraction",
    ):
        if vehicle_values[name] <= 0.0:
            raise StrictConfigError(f"game.vehicle.{name} must be positive")
    for name in (
        "speed_taper_knee_fraction",
        "speed_taper_low_floor",
        "speed_taper_high_floor",
        "curb_collision_restitution",
        "curb_forward_momentum_retention",
    ):
        if vehicle_values[name] > 1.0:
            raise StrictConfigError(f"game.vehicle.{name} must be at most 1")
    return TaxiSettings(
        game=TaxiGameConfig(
            vehicle=TaxiVehicleConfig(**cast(Any, vehicle_values)),
            **cast(Any, rules),
        ),
        driver_input=DriverInputConfig(**input_values),
    )
