# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored-map compilation and immutable scene preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.game_map import compile_game_map
from omnidreams_game_engine.scene_loader import load_scene_bundle
from omnidreams_game_engine.types import SceneDefinition


@dataclass(frozen=True, slots=True)
class SceneRequest:
    """Inputs selecting one immutable game scene."""

    map_path: Path
    camera_name: str = "camera_front_wide_120fov"
    use_prompt_context: bool = False
    force_recompile: bool = False
    spawn_id: str | None = None
    """Authored spawn to load; ``None`` selects the map's first spawn."""


def load_scene(request: SceneRequest, raster: RasterConfig) -> SceneDefinition:
    """Compile an authored map if necessary and load its runtime scene."""
    compiled = compile_game_map(
        request.map_path,
        spawn_id=request.spawn_id,
        use_prompt_context=request.use_prompt_context,
        force=request.force_recompile,
    )
    return load_scene_bundle(
        scene_path=compiled.archive_path,
        camera_name=request.camera_name,
        raster=raster,
    )
