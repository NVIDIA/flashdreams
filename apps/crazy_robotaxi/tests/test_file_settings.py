# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-safe tests for standalone renderer and game YAML settings."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from crazy_robotaxi import runtime_cli
from crazy_robotaxi.game_settings import load_game_settings
from omnidreams_game_engine.renderer_settings import load_renderer_settings
from omnidreams_game_engine.yaml_config import StrictConfigError

pytestmark = pytest.mark.ci_cpu

_CONFIG_ROOT = Path(__file__).parents[1] / "crazy_robotaxi" / "configs"


def test_bundled_renderer_and_game_settings_are_complete() -> None:
    """Load both packaged configuration documents without implicit fields."""
    renderer = load_renderer_settings(_CONFIG_ROOT / "default_renderer.yaml")
    game = load_game_settings(_CONFIG_ROOT / "default_game.yaml")

    assert renderer.raster.line_width_px == pytest.approx(12.0)
    assert renderer.bev.width == 1024
    assert not renderer.visual_flare_enabled
    assert game.traffic_density == pytest.approx(0.4)
    assert game.vehicle.aabb_width_m == pytest.approx(2.0)
    assert game.vehicle.curb_forward_momentum_retention == pytest.approx(0.85)


def test_renderer_settings_reject_missing_and_unknown_keys(tmp_path: Path) -> None:
    """Reject renderer documents that are incomplete or misspelled."""
    source = yaml.safe_load(
        (_CONFIG_ROOT / "default_renderer.yaml").read_text(encoding="utf-8")
    )
    del source["raster"]["line_width_px"]
    source["raster"]["line_wdith_px"] = 12.0
    path = tmp_path / "renderer.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(StrictConfigError, match="missing required keys: line_width_px"):
        load_renderer_settings(path)


def test_game_settings_reject_missing_fields(tmp_path: Path) -> None:
    """Reject gameplay documents that omit a required vehicle property."""
    source = yaml.safe_load(
        (_CONFIG_ROOT / "default_game.yaml").read_text(encoding="utf-8")
    )
    del source["vehicle"]["aabb_width_m"]
    path = tmp_path / "game.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys: aabb_width_m"):
        load_game_settings(path)


def test_visual_cli_values_override_renderer_yaml() -> None:
    """Apply explicit BEV flags after loading the renderer document."""
    args = runtime_cli.build_parser().parse_args(
        [
            "--renderer-config",
            str(_CONFIG_ROOT / "default_renderer.yaml"),
            "--bev-resolution",
            "640x480",
            "--bev-height-m",
            "90",
            "--no-bev",
        ]
    )

    settings = runtime_cli.renderer_settings_from_args(args)

    assert not settings.bev.enabled
    assert (settings.bev.width, settings.bev.height) == (640, 480)
    assert settings.bev.height_m == pytest.approx(90.0)
    assert args.bev is False
    assert args.bev_resolution == "640x480"
    assert args.bev_height_m == pytest.approx(90.0)


def test_default_renderer_populates_legacy_presenter_arguments() -> None:
    """Publish file defaults through the namespace consumed by HUD presenters."""
    args = runtime_cli.build_parser().parse_args([])

    runtime_cli.renderer_settings_from_args(args)

    assert args.bev is True
    assert args.bev_resolution == "1024x1024"
    assert args.bev_height_m == pytest.approx(75.0)
    assert args.bev_fov_deg == pytest.approx(60.0)
    assert args.bev_tilt_deg == pytest.approx(0.0)
