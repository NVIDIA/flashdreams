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

"""SceneBundle and layer dataclasses consumed by the SlangPy rasterizer.

Adapted from ``roaddreams.types``; the Bundle is the static, world-space view
of an HD map that the Slang kernels expect. It does not carry any timestamp
dimension because flashdreams only uses static HD-map conditioning today.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class WorldLineSegments:
    """A pool of world-space line segments sharing one color and width."""

    segments_world: FloatArray  # shape (N, 2, 3)
    color_rgba: tuple[float, float, float, float]
    width_px: float
    layer_name: str


@dataclass(frozen=True)
class WorldTriangleList:
    """A pool of world-space triangles sharing one color."""

    triangles_world: FloatArray  # shape (N, 3, 3)
    color_rgba: tuple[float, float, float, float]
    layer_name: str


@dataclass(frozen=True)
class WorldPolygonList:
    """A pool of world-space planar polygons sharing one color.

    Each polygon is an ``(M, 3)`` array of vertices; ``M`` may differ between
    polygons. The rasterizer triangulates polygons on the CPU before upload.
    """

    polygons_world: tuple[FloatArray, ...]
    color_rgba: tuple[float, float, float, float]
    layer_name: str


@dataclass(frozen=True)
class SceneBundle:
    """Container for the world-space HD-map geometry the rasterizer consumes."""

    line_layers: tuple[WorldLineSegments, ...] = ()
    triangle_layers: tuple[WorldTriangleList, ...] = ()
    polygon_layers: tuple[WorldPolygonList, ...] = ()
