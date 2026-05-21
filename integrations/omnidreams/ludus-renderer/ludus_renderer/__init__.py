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
Ludus Renderer - GPU-native F-theta fisheye renderer with two backends:

- ``LudusCudaTimestampedContext`` (always available): CUDA software
  rasterizer based on the HPG 2011 CudaRaster triangle pipeline.
- ``LudusTimestampedContext`` (requires Vulkan 1.3 + VK_EXT_mesh_shader):
  hardware mesh-shader pipeline with CUDA-Vulkan external-memory interop.

High-level API::

    from ludus_renderer import LudusCudaTimestampedContext, load_clipgt_scene

    scene = load_clipgt_scene("/path/to/clipgt/scene", device="cuda")
    ctx = LudusCudaTimestampedContext(device="cuda")
    ctx.upload_cameras(cameras)
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    images = ctx.render(scene_ids, camera_ids, timestamps_us, camera_type_ids,
                        camera_poses, resolution=(H, W))

The Vulkan backend exposes the same public method surface so it is a
near drop-in replacement::

    from ludus_renderer import LudusTimestampedContext
    ctx = LudusTimestampedContext(device="cuda")
    # ... same upload_cameras / upload_scene / render API ...

NVJPEG Encoding (GPU-accelerated JPEG)::

    from ludus_renderer import nvjpeg

    images = torch.randint(0, 255, (4, 3, 480, 640), dtype=torch.uint8, device='cuda')
    jpeg_bytes_list = nvjpeg.encode(images, quality=85)
"""

__version__ = "0.1.0"

# NVJPEG GPU encoding (lazy import)
from . import nvjpeg

# Low-level API (from _ops)
from ._ops import (
    CAMERA_TYPE_BEV,
    CAMERA_TYPE_REGULAR,
    CUBE_FLAG_WIREFRAME,
    PRIM_CROSSWALK,
    PRIM_EGO_OBSTACLE,
    PRIM_EGO_TRAJECTORY,
    PRIM_LANE_LINE,
    PRIM_OBSTACLE,
    # Constants
    PRIM_ROAD_BOUNDARY,
    PRIM_STATIC_OBSTACLE,
    PRIM_TYPE_COUNT,
    CapStyle,
    Cube,
    CubePool,
    FThetaCamera,
    # Contexts
    LudusCudaTimestampedContext,
    LudusTimestampedContext,
    ObstaclePool,
    Polygon,
    # Primitives
    Polyline,
    TimestampedPolygonPool,
    # Timestamped pools
    TimestampedPolylinePool,
    TimestampedScene,
)
from .augmentation import mirror_augment_scene
from .clipgt import ClipgtGpuScene, EgoTrackData, load_av2_scene, load_clipgt_scene

# Scene cache
from .scene_cache import (
    LMDBSceneStore,
    PackedSceneBuffers,
    SceneDatabase,
    pack_scene,
    scene_bytes,
    scene_key,
    scene_to_device,
)

# Utilities
from .util import (
    FLU_TO_OPENCV_MATRIX,
    FLU_TO_OPENGL_MATRIX,
    OPENCV_TO_OPENGL_MATRIX,
    projection_matrix,
    resample_timestamps,
    rgb,
)

__all__ = [
    # Version
    "__version__",
    # High-level API
    "ClipgtGpuScene",
    "load_clipgt_scene",
    "load_av2_scene",
    "EgoTrackData",
    "mirror_augment_scene",
    # Rendering contexts
    "LudusCudaTimestampedContext",
    "LudusTimestampedContext",
    # Primitives
    "Polyline",
    "Polygon",
    "Cube",
    "FThetaCamera",
    "CapStyle",
    # Timestamped pools
    "TimestampedPolylinePool",
    "TimestampedPolygonPool",
    "CubePool",
    "ObstaclePool",
    "TimestampedScene",
    # Constants
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
    "CUBE_FLAG_WIREFRAME",
    # Utilities
    "rgb",
    "resample_timestamps",
    "projection_matrix",
    "FLU_TO_OPENCV_MATRIX",
    "OPENCV_TO_OPENGL_MATRIX",
    "FLU_TO_OPENGL_MATRIX",
    # Scene cache
    "SceneDatabase",
    "LMDBSceneStore",
    "PackedSceneBuffers",
    "pack_scene",
    "scene_key",
    "scene_bytes",
    "scene_to_device",
    # NVJPEG GPU encoding
    "nvjpeg",
]
