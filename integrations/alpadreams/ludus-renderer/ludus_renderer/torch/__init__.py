# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
