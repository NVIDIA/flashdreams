# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Low-level rendering operations for Ludus renderer.

This module re-exports all symbols from the split submodules:
- _plugin: JIT compilation
- primitives: Data classes and packing functions
- context: ``LudusCudaTimestampedContext`` rendering context
"""

# JIT compilation
from ._plugin import _get_plugin, get_log_level, set_log_level

# Ludus rendering context.
from .context import LudusCudaTimestampedContext

# Primitive data types and packing
from .primitives import (
    CAMERA_TYPE_BEV,
    CAMERA_TYPE_REGULAR,
    CUBE_FLAG_WIREFRAME,
    PRIM_BUFFER_ZONE,
    PRIM_CROSSWALK,
    PRIM_DOT_WHITE,
    PRIM_DOT_YELLOW,
    PRIM_EGO_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_INTERSECTION,
    PRIM_LANE_BOUNDARY,
    PRIM_LANE_LINE,
    PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_OBSTACLE,
    PRIM_POLE,
    # Constants
    PRIM_ROAD_BOUNDARY,
    PRIM_ROAD_ISLAND,
    PRIM_ROAD_MARKING,
    PRIM_STATIC_OBSTACLE,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_TYPE_COUNT,
    PRIM_WAIT_LINE,
    # Data classes
    CapStyle,
    Cube,
    CubePool,
    FThetaCamera,
    ObstaclePool,
    Polygon,
    Polyline,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
    _pack_cameras,
    # Packing functions (internal)
    _pack_cubes,
    _pack_polygons,
    _pack_polylines,
    _triangulate_polygon_ear_clipping,
)

__all__ = [
    # Plugin
    "_get_plugin",
    "get_log_level",
    "set_log_level",
    # Constants
    "PRIM_ROAD_BOUNDARY",
    "PRIM_LANE_LINE",
    "PRIM_CROSSWALK",
    "PRIM_STATIC_OBSTACLE",
    "PRIM_EGO_TRAJECTORY",
    "PRIM_OBSTACLE",
    "PRIM_EGO_OBSTACLE",
    "PRIM_WAIT_LINE",
    "PRIM_POLE",
    "PRIM_ROAD_MARKING",
    "PRIM_LANE_BOUNDARY",
    "PRIM_TRAFFIC_LIGHT",
    "PRIM_TRAFFIC_SIGN",
    "PRIM_INTERSECTION",
    "PRIM_ROAD_ISLAND",
    "PRIM_BUFFER_ZONE",
    "PRIM_LANE_LINE_WHITE_SOLID",
    "PRIM_LANE_LINE_WHITE_DASHED",
    "PRIM_LANE_LINE_YELLOW_SOLID",
    "PRIM_LANE_LINE_YELLOW_DASHED",
    "PRIM_DOT_YELLOW",
    "PRIM_DOT_WHITE",
    "PRIM_TYPE_COUNT",
    "CAMERA_TYPE_REGULAR",
    "CAMERA_TYPE_BEV",
    "CUBE_FLAG_WIREFRAME",
    # Data classes
    "CapStyle",
    "Polyline",
    "Polygon",
    "Cube",
    "FThetaCamera",
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "CubePool",
    "ObstaclePool",
    "TimestampedScene",
    # Ludus rendering context
    "LudusCudaTimestampedContext",
]
