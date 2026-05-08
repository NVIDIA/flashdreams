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

"""SlangPy-based HD-map rasterizer that replaces the ludus-renderer CUDA path.

The Slang shader code, scene-bundle representation, and per-pass orchestration
in this subpackage are adapted from the roaddreams sample
(``references/omni-dreams-slangpy/samples/roaddreams``). The public surface
exposed here is what :mod:`alpadreams.conditioning.renderer` consumes; nothing
else in flashdreams should depend on this subpackage directly.
"""

from alpadreams.conditioning.slang_renderer.config import RasterConfig
from alpadreams.conditioning.slang_renderer.mirror_augment import mirror_augment_bundle
from alpadreams.conditioning.slang_renderer.rasterizer import SlangConditionRasterizer
from alpadreams.conditioning.slang_renderer.scene_loader import scene_data_to_bundle
from alpadreams.conditioning.slang_renderer.types import (
    SceneBundle,
    WorldLineSegments,
    WorldPolygonList,
    WorldTriangleList,
)

__all__ = [
    "RasterConfig",
    "SceneBundle",
    "SlangConditionRasterizer",
    "WorldLineSegments",
    "WorldPolygonList",
    "WorldTriangleList",
    "mirror_augment_bundle",
    "scene_data_to_bundle",
]
