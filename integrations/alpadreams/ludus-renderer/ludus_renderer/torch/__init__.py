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
Backward compatibility layer for ludus_renderer.torch imports.

New code should import directly from ludus_renderer:
    from ludus_renderer import LudusRenderer, LudusTimestampedContext

This module re-exports the same symbols for backward compatibility:
    from ludus_renderer.torch import LudusTimestampedContext  # Still works
"""

# Re-export from _ops for backward compatibility
from .._ops import (
    # Contexts
    LudusGLContext,
    LudusTimestampedContext,
    ludus_render,
    # Primitives
    Polyline,
    Polygon,
    Cube,
    FThetaCamera,
    CapStyle,
    # Timestamped pools
    TimestampedPolylinePool,
    TimestampedPolygonPool,
    CubePool,
    ObstaclePool,
    TimestampedScene,
    # Constants
    PRIM_ROAD_BOUNDARY,
    PRIM_LANE_LINE,
    PRIM_CROSSWALK,
    PRIM_STATIC_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_OBSTACLE,
    PRIM_EGO_OBSTACLE,
    PRIM_TYPE_COUNT,
    CAMERA_TYPE_REGULAR,
    CAMERA_TYPE_BEV,
    CUBE_FLAG_WIREFRAME,
)

__all__ = [
    # Core
    "LudusGLContext",
    "LudusTimestampedContext",
    "ludus_render",
    # Primitives
    "Polyline",
    "Polygon",
    "Cube",
    "FThetaCamera",
    "CapStyle",
    # Timestamped
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "CubePool",
    "ObstaclePool",
    "TimestampedScene",
    "CUBE_FLAG_WIREFRAME",
    # Primitive Type IDs
    "PRIM_ROAD_BOUNDARY",
    "PRIM_LANE_LINE",
    "PRIM_CROSSWALK",
    "PRIM_STATIC_OBSTACLE",
    "PRIM_EGO_TRAJECTORY",
    "PRIM_OBSTACLE",
    "PRIM_EGO_OBSTACLE",
    "PRIM_TYPE_COUNT",
    "CAMERA_TYPE_REGULAR",
    "CAMERA_TYPE_BEV",
]
