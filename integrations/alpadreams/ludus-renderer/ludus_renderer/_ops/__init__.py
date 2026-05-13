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
- context: Ludus rendering contexts (LudusGLContext, LudusTimestampedContext)
"""

# JIT compilation
from ._plugin import _get_plugin, get_log_level, set_log_level

# Primitive data types and packing
from .primitives import (
    # Constants
    PRIM_ROAD_BOUNDARY,
    PRIM_LANE_LINE,
    PRIM_CROSSWALK,
    PRIM_STATIC_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_OBSTACLE,
    PRIM_EGO_OBSTACLE,
    PRIM_WAIT_LINE,
    PRIM_POLE,
    PRIM_ROAD_MARKING,
    PRIM_LANE_BOUNDARY,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_INTERSECTION,
    PRIM_ROAD_ISLAND,
    PRIM_BUFFER_ZONE,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED,
    PRIM_DOT_YELLOW,
    PRIM_DOT_WHITE,
    PRIM_TYPE_COUNT,
    CAMERA_TYPE_REGULAR,
    CAMERA_TYPE_BEV,
    CUBE_FLAG_WIREFRAME,
    # Data classes
    CapStyle,
    Polyline,
    Polygon,
    Cube,
    FThetaCamera,
    TimestampedPolylinePool,
    TimestampedPolygonPool,
    CubePool,
    ObstaclePool,
    TimestampedScene,
    # Packing functions (internal)
    _pack_cubes,
    _pack_polylines,
    _pack_polygons,
    _pack_cameras,
    _triangulate_polygon_ear_clipping,
)

# Ludus context and rendering will be imported from context.py once created
# For now, we'll import from the old ops.py path until migration is complete

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
    # Ludus rendering (will be added)
    "LudusGLContext",
    "ludus_render",
    "LudusTimestampedContext",
]

# Import Ludus context classes (these remain in context.py for now,
# we'll create this file next)
try:
    from .context import LudusGLContext, ludus_render, LudusTimestampedContext
except ImportError:
    # Fallback during migration - import from old location
    pass
