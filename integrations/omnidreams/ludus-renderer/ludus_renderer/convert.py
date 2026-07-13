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
    CapStyle,
    FThetaCamera,
    ObstaclePool,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
)

__all__ = [
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "ObstaclePool",
    "TimestampedScene",
    "FThetaCamera",
    "CapStyle",
]
