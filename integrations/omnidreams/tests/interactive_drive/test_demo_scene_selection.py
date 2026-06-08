# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from pathlib import Path

from omnidreams.interactive_drive.demo import (
    SceneOption,
    _resolve_scene_variant,
    build_parser,
)


def test_auto_start_flag_and_deprecated_alias() -> None:
    parser = build_parser()

    assert parser.parse_args([]).auto_start is False
    assert parser.parse_args(["--auto-start"]).auto_start is True
    # --autoload-scene is kept as a backward-compatible alias for --auto-start.
    assert parser.parse_args(["--autoload-scene"]).auto_start is True
    assert parser.parse_args(["--no-autoload-scene"]).auto_start is False


def test_resolve_scene_variant_prefers_weather_archive_path_for_default(
    tmp_path: Path,
) -> None:
    scene_uuid = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    base = (tmp_path / f"clipgt-{scene_uuid}.usdz").resolve()
    snow = (tmp_path / f"clipgt-{scene_uuid}-snow.usdz").resolve()
    option = SceneOption(
        label="Quiet Suburban Boulevard",
        path=base,
        variants=("default", "rain", "snow"),
        variant_paths={"default": base, "snow": snow},
    )

    assert _resolve_scene_variant((option,), snow, "default") == "snow"


def test_resolve_scene_variant_keeps_explicit_weather_choice(tmp_path: Path) -> None:
    scene_uuid = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    base = (tmp_path / f"clipgt-{scene_uuid}.usdz").resolve()
    snow = (tmp_path / f"clipgt-{scene_uuid}-snow.usdz").resolve()
    rain = (tmp_path / f"clipgt-{scene_uuid}-rain.usdz").resolve()
    option = SceneOption(
        label="Quiet Suburban Boulevard",
        path=base,
        variants=("default", "rain", "snow"),
        variant_paths={"default": base, "rain": rain, "snow": snow},
    )

    assert _resolve_scene_variant((option,), snow, "rain") == "rain"


def test_resolve_scene_variant_legacy_option_without_variant_paths(
    tmp_path: Path,
) -> None:
    scene = (tmp_path / "legacy.usdz").resolve()
    option = SceneOption(label="legacy", path=scene, variants=("1", "2"))

    assert _resolve_scene_variant((option,), scene, "default") == "1"
    assert _resolve_scene_variant((option,), scene, "2") == "2"
