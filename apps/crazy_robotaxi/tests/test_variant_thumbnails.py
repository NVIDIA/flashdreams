# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Map thumbnails feeding the HUD variant dropdown."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from crazy_robotaxi import cli as crazy_cli
from omnidreams_game_engine import demo as engine_demo

pytestmark = pytest.mark.ci_cpu

_MAP = (
    Path(__file__).parents[1] / "crazy_robotaxi" / "maps" / "minimal_loop.robotaxi.yaml"
)


@pytest.mark.parametrize(
    "build_option",
    [crazy_cli._scene_option_for_game_map, engine_demo._scene_option_for_game_map],
)
def test_map_option_uses_authored_seed_image(
    build_option: Callable[[Path], object],
) -> None:
    option = build_option(_MAP)

    assert option.label == "Minimal Loop and Parking Lot"
    assert option.variants == ("default",)
    assert option.thumbnail is not None
    assert option.thumbnail.size == crazy_cli.SCENE_THUMB_SIZE
    assert set(option.variant_thumbnails) == {"default"}
    assert option.variant_paths == {"default": _MAP}


@pytest.mark.parametrize(
    "build_option",
    [crazy_cli._scene_option_for_game_map, engine_demo._scene_option_for_game_map],
)
def test_map_option_generates_thumbnail_when_seed_image_is_omitted(
    build_option: Callable[[Path], object], tmp_path: Path
) -> None:
    document = yaml.safe_load(_MAP.read_text(encoding="utf-8"))
    del document["spawns"][0]["variants"]["default"]["image"]
    path = tmp_path / "fallback.robotaxi.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    option = build_option(path)

    assert option.thumbnail is not None
    assert option.thumbnail.size == crazy_cli.SCENE_THUMB_SIZE
    assert set(option.variant_thumbnails) == {"default"}


@pytest.mark.parametrize(
    "discover",
    [crazy_cli._discover_scene_options, engine_demo._discover_scene_options],
)
def test_map_discovery_ignores_archives(
    discover: Callable[[Path, Path], tuple[object, ...]], tmp_path: Path
) -> None:
    (tmp_path / "recorded.usdz").write_bytes(b"not a map")

    options = discover(tmp_path, _MAP)

    assert options
    assert all(option.path.name.endswith(".robotaxi.yaml") for option in options)
    assert _MAP.resolve() in {option.path for option in options}
