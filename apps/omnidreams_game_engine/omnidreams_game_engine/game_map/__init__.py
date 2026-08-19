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
    GameMapRoad,
    GameMapRoadBoundary,
    GameMapSpawn,
    GameMapTopology,
    ResolvedGameMap,
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
    "GameMapRoad",
    "GameMapRoadBoundary",
    "GameMapSpawn",
    "GameMapTopology",
    "ResolvedGameMap",
    "compile_game_map",
    "load_game_map",
    "load_game_map_header",
    "resolve_seed_asset",
    "write_game_map_preview",
]
