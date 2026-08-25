# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Semantic game-map loading, compilation, and previews."""

from omnidreams_game_engine.game_map._schema import (
    GAME_MAP_SUFFIX,
    GameMapError,
    GameMapHeader,
    load_game_map_header,
    resolve_seed_asset,
)
from omnidreams_game_engine.game_map.compiler import (
    CompiledGameMap,
    compile_game_map,
)
from omnidreams_game_engine.game_map.loader import load_game_map
from omnidreams_game_engine.game_map.preview import write_game_map_preview
from omnidreams_game_engine.game_map.spawn_render import (
    SPAWN_RENDERER_VERSION,
    render_spawn_first_frame,
    write_spawn_first_frame_preview,
)
from omnidreams_game_engine.game_map.types import (
    GameMapBoundaryAttributes,
    GameMapCurb,
    GameMapElement,
    GameMapLane,
    GameMapLaneDivider,
    GameMapLinearAttributes,
    GameMapLineMarking,
    GameMapNode,
    GameMapParkingAccess,
    GameMapRaceCourse,
    GameMapRoad,
    GameMapRoadBoundary,
    GameMapSpawn,
    GameMapTopology,
    GameMapTrafficVehicle,
    ResolvedGameMap,
)
from omnidreams_game_engine.game_map.vicinity import (
    GameMapVicinity,
    GameMapVicinityResolver,
)

__all__ = [
    "CompiledGameMap",
    "GAME_MAP_SUFFIX",
    "GameMapError",
    "GameMapBoundaryAttributes",
    "GameMapCurb",
    "GameMapElement",
    "GameMapHeader",
    "GameMapLane",
    "GameMapLaneDivider",
    "GameMapLinearAttributes",
    "GameMapLineMarking",
    "GameMapNode",
    "GameMapParkingAccess",
    "GameMapRaceCourse",
    "GameMapRoad",
    "GameMapRoadBoundary",
    "GameMapSpawn",
    "GameMapTrafficVehicle",
    "GameMapTopology",
    "GameMapVicinity",
    "GameMapVicinityResolver",
    "ResolvedGameMap",
    "SPAWN_RENDERER_VERSION",
    "compile_game_map",
    "load_game_map",
    "load_game_map_header",
    "resolve_seed_asset",
    "render_spawn_first_frame",
    "write_game_map_preview",
    "write_spawn_first_frame_preview",
]
