# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layered Crazy Robotaxi gameplay configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Literal

from omnidreams_game_engine.yaml_config import (
    StrictConfigError,
    load_yaml_mapping,
    overlay_dataclass,
    require_mapping,
    require_version,
)

from crazy_robotaxi.dynamics import TaxiVehicleConfig
from crazy_robotaxi.live_edit.config import LiveEditConfig
from crazy_robotaxi.rules import TaxiGameConfig

_RUNTIME_GAME_FIELDS = {"seed", "high_scores_path"}
_RULE_FIELDS = (
    {item.name for item in fields(TaxiGameConfig)} - _RUNTIME_GAME_FIELDS - {"vehicle"}
)
_VEHICLE_FIELDS = {item.name for item in fields(TaxiVehicleConfig)}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "mode",
    "effects",
    "rules",
    "vehicle",
    "taxi",
    "race",
    "live_edit",
    "diagnostics",
}


@dataclass(frozen=True)
class GameEffectsSettings:
    """Game-directed visual effects."""

    visual_flare_enabled: bool = False
    """Whether collisions may trigger the full-screen visual flare."""


@dataclass(frozen=True)
class TaxiSessionSettings:
    """Taxi-mode session and persistence settings."""

    seed: int | None = None
    """Optional deterministic fare-layout seed."""

    high_scores_path: Path | None = None
    """Taxi leaderboard path; ``None`` uses the cache-directory default."""


@dataclass(frozen=True)
class RaceSessionSettings:
    """Race-mode selection and persistence settings."""

    course: str | None = None
    """Race-course identifier; ``None`` selects the map's first course."""

    times_path: Path | None = None
    """Race leaderboard path; ``None`` uses the cache-directory default."""


@dataclass(frozen=True)
class GameDiagnosticsSettings:
    """Optional Crazy Robotaxi diagnostic outputs."""

    alignment_directory: Path | None = None
    """Optional frame-alignment diagnostic output directory."""


@dataclass(frozen=True)
class CrazyRobotaxiSettings:
    """Complete durable game configuration for the V2 application."""

    mode: Literal["taxi", "race"] = "taxi"
    """Gameplay mode selected for the session."""

    effects: GameEffectsSettings = field(default_factory=GameEffectsSettings)
    """Game-directed visual effects."""

    game: TaxiGameConfig = field(default_factory=TaxiGameConfig)
    """Taxi rules and player-vehicle configuration."""

    taxi: TaxiSessionSettings = field(default_factory=TaxiSessionSettings)
    """Taxi session and persistence settings."""

    race: RaceSessionSettings = field(default_factory=RaceSessionSettings)
    """Race selection and persistence settings."""

    live_edit: LiveEditConfig = field(default_factory=LiveEditConfig)
    """Map-context prompting and live-edit ability settings."""

    diagnostics: GameDiagnosticsSettings = field(
        default_factory=GameDiagnosticsSettings
    )
    """Optional diagnostic outputs."""


TaxiSettings = CrazyRobotaxiSettings


def load_game_settings(
    path: Path,
    *,
    base: CrazyRobotaxiSettings | None = None,
) -> CrazyRobotaxiSettings:
    """Overlay a partial game YAML onto typed settings.

    Args:
        path: Game configuration path.
        base: Lower-precedence settings; ``None`` uses typed defaults.

    Returns:
        Resolved game, session, and live-edit settings.

    Raises:
        StrictConfigError: The YAML or merged settings are invalid.
    """
    config_path = path.expanduser().resolve()
    doc = load_yaml_mapping(config_path)
    require_version(doc, "game")
    _reject_unknown(doc, _TOP_LEVEL_FIELDS, "game")
    settings = base or CrazyRobotaxiSettings()
    base_dir = config_path.parent

    if "mode" in doc:
        settings = overlay_dataclass(
            settings, {"mode": doc["mode"]}, "game", base_dir=base_dir
        )
    for yaml_name in ("effects", "taxi", "race", "live_edit", "diagnostics"):
        if yaml_name not in doc:
            continue
        nested = overlay_dataclass(
            getattr(settings, yaml_name),
            require_mapping(doc[yaml_name], f"game.{yaml_name}"),
            f"game.{yaml_name}",
            base_dir=base_dir,
        )
        settings = replace(settings, **{yaml_name: nested})

    game = settings.game
    if "rules" in doc:
        rules = require_mapping(doc["rules"], "game.rules")
        _reject_unknown(rules, _RULE_FIELDS, "game.rules")
        game = overlay_dataclass(game, rules, "game.rules", base_dir=base_dir)
    if "vehicle" in doc:
        vehicle_values = require_mapping(doc["vehicle"], "game.vehicle")
        _reject_unknown(vehicle_values, _VEHICLE_FIELDS, "game.vehicle")
        vehicle = overlay_dataclass(
            game.vehicle, vehicle_values, "game.vehicle", base_dir=base_dir
        )
        game = replace(game, vehicle=vehicle)
    game = replace(
        game,
        seed=settings.taxi.seed,
        **(
            {"high_scores_path": settings.taxi.high_scores_path}
            if settings.taxi.high_scores_path is not None
            else {}
        ),
    )
    settings = replace(settings, game=game)
    _validate_game_settings(settings)
    return settings


def _reject_unknown(values: dict[str, object], allowed: set[str], context: str) -> None:
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise StrictConfigError(f"{context} has unknown keys: {', '.join(unknown)}")


def _validate_game_settings(settings: CrazyRobotaxiSettings) -> None:
    game = settings.game
    for name in _RULE_FIELDS:
        if getattr(game, name) < 0:
            raise StrictConfigError(f"game.rules.{name} must be non-negative")
    for name in _VEHICLE_FIELDS:
        value = getattr(game.vehicle, name)
        if type(value) is not bool and value < 0:
            raise StrictConfigError(f"game.vehicle.{name} must be non-negative")
    if game.fare_min_route_distance_m > game.fare_max_route_distance_m:
        raise StrictConfigError(
            "game.rules.fare_min_route_distance_m must not exceed fare_max_route_distance_m"
        )
    if game.min_time_s > game.max_time_s:
        raise StrictConfigError("game.rules.min_time_s must not exceed max_time_s")
