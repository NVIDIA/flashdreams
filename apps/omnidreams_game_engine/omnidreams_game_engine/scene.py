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
    variant: str = "default"
    prompt: str | None = None
    use_prompt_context: bool = False
    force_recompile: bool = False


def load_scene(request: SceneRequest, raster: RasterConfig) -> SceneDefinition:
    """Compile an authored map if necessary and load its runtime scene."""
    compiled = compile_game_map(request.map_path, force=request.force_recompile)
    prompt_override = request.prompt
    if prompt_override is None and request.use_prompt_context:
        variants = compiled.game_map.default_spawn.variants
        selected = next(
            (item for item in variants if item.name == request.variant), variants[0]
        )
        prompt_override = selected.prompt_context or selected.prompt
    return load_scene_bundle(
        scene_path=compiled.archive_path,
        camera_name=request.camera_name,
        variant=request.variant,
        prompt_override=prompt_override,
        raster=raster,
    )
