# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scene loading utilities for ludus_renderer.

Use load_scene() from clipgt.py to load ClipGT or AV2 scenes. It provides
a ClipgtGpuScene with:
- timestamped_scene: TimestampedScene ready for GPU upload
- cameras: List of FThetaCamera intrinsics
- ego_track: EgoTrackData for pose computation

Example:
    from ludus_renderer import load_scene

    scene = load_scene("/path/to/scene", device="cuda")
    renderer.upload_scene(scene.timestamped_scene)
"""

# Re-export from clipgt for convenience
from .clipgt import (
    ClipgtGpuScene,
    EgoTrackData,
    is_clipgt,
    load_av2_scene,
    load_clipgt_scene,
    load_scene,
)

__all__ = [
    "ClipgtGpuScene",
    "load_scene",
    "is_clipgt",
    "load_clipgt_scene",
    "load_av2_scene",
    "EgoTrackData",
]
