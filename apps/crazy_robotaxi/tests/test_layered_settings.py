# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for layered Crazy Robotaxi settings."""

from pathlib import Path

import pytest
from crazy_robotaxi.application import CrazyRobotaxiApplication
from crazy_robotaxi.config import load_game_settings
from omnidreams_game_engine.engine_settings import load_engine_settings
from omnidreams_game_engine.yaml_config import StrictConfigError

pytestmark = pytest.mark.ci_cpu

_CONFIGS = Path(__file__).parents[1] / "crazy_robotaxi" / "configs"


def test_shipped_example_configs_load() -> None:
    engine = load_engine_settings(_CONFIGS / "example_engine_config.yaml")
    game = load_game_settings(_CONFIGS / "example_game_config.yaml")

    assert (
        engine.map.path
        == (_CONFIGS.parent / "maps" / "boulevard_district.robotaxi.yaml").resolve()
    )
    assert engine.world_model.model_preset == "standard"
    assert game.game.vehicle.max_steer_rad == pytest.approx(0.69)


def test_partial_game_config_retains_defaults_and_resolves_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "game.yaml"
    config_path.write_text(
        """\
schema_version: 1
rules:
  global_time_s: 90
taxi:
  seed: 17
  high_scores_path: scores.csv
live_edit:
  coins:
    enabled: true
""",
        encoding="utf-8",
    )

    settings = load_game_settings(config_path)

    assert settings.game.global_time_s == pytest.approx(90.0)
    assert settings.game.pickup_radius_m == pytest.approx(5.0)
    assert settings.game.seed == 17
    assert settings.game.high_scores_path == (tmp_path / "scores.csv").resolve()
    assert settings.live_edit.coins.enabled


def test_partial_engine_config_retains_defaults_and_resolves_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "engine.yaml"
    config_path.write_text(
        """\
schema_version: 1
map:
  path: maps/test.robotaxi.yaml
rendering:
  bev:
    height_m: 42
presentation:
  show_fps: true
runtime:
  prewarm_blocks: 2
""",
        encoding="utf-8",
    )

    settings = load_engine_settings(config_path)

    assert settings.map.path == (tmp_path / "maps/test.robotaxi.yaml").resolve()
    assert settings.rendering.raster.width == 1280
    assert settings.rendering.bev.height_m == pytest.approx(42.0)
    assert settings.presentation.show_fps
    assert settings.runtime.prewarm_blocks == 2


def test_explicit_cli_overrides_engine_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "engine.yaml"
    config_path.write_text(
        """\
schema_version: 1
world_model:
  model_preset: perf
  device: cuda
presentation:
  show_fps: true
runtime:
  prewarm_blocks: 1
  profile_world_model: true
  profile_input_latency: false
""",
        encoding="utf-8",
    )
    app = CrazyRobotaxiApplication()

    app.init(
        [
            "--engine-config",
            str(config_path),
            "--model-preset",
            "standard",
            "--device",
            "cpu",
            "--prewarm-blocks",
            "3",
            "--profile-input-latency",
            "--no-show-fps",
        ]
    )

    assert app._config is not None
    assert app._config.model_preset_name == "standard"
    assert app._config.device == "cpu"
    assert app._config.prewarm_blocks == 3
    assert app._config.profile_input_latency
    assert not app._config.show_fps
    assert app._config.pipeline_profiling


def test_unknown_layered_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "game.yaml"
    config_path.write_text(
        "schema_version: 1\nrules:\n  not_a_rule: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(StrictConfigError, match="not_a_rule"):
        load_game_settings(config_path)


def test_game_yaml_flags_and_inactive_race_settings_are_layered(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "game.yaml"
    config_path.write_text(
        """\
schema_version: 1
effects:
  visual_flare_enabled: true
race:
  course: future-course
  times_path: race-times.csv
""",
        encoding="utf-8",
    )
    yaml_app = CrazyRobotaxiApplication()
    cli_app = CrazyRobotaxiApplication()

    yaml_app.init(["--game-config", str(config_path)])
    cli_app.init(["--game-config", str(config_path), "--no-visual-flare"])

    assert yaml_app._config is not None
    assert yaml_app._config.game_mode == "taxi"
    assert yaml_app._config.race_course_id == "future-course"
    assert yaml_app._config.visual_flare_enabled
    assert cli_app._config is not None
    assert not cli_app._config.visual_flare_enabled
