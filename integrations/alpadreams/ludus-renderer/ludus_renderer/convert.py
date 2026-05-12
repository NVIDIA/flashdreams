"""
Conversion utilities for ludus_renderer.

Note: Direct scene loading via clipgt.py is preferred. The clipgt module 
handles conversion internally and provides ready-to-use TimestampedScene objects.

For custom scene building, see the primitives in ludus_renderer._ops:
- TimestampedPolylinePool
- TimestampedPolygonPool  
- ObstaclePool
- TimestampedScene
"""

# Re-export useful types for custom scene building
from ._ops import (
    TimestampedPolylinePool,
    TimestampedPolygonPool,
    ObstaclePool,
    TimestampedScene,
    FThetaCamera,
    CapStyle,
)

__all__ = [
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "ObstaclePool",
    "TimestampedScene",
    "FThetaCamera",
    "CapStyle",
]
