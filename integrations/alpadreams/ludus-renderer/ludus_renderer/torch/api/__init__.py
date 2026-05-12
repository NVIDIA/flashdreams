# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Backward compatibility layer for ludus_renderer.torch.api imports.

New code should import directly from ludus_renderer:
    from ludus_renderer import LudusRenderer, load_av2_scene

This module re-exports the same symbols for backward compatibility:
    from ludus_renderer.torch.api import LudusRenderer  # Still works
"""

# Re-export from new top-level locations
from ...renderer import LudusRenderer
from ...scene import Av2GpuScene, load_av2_scene, load_clipgt_scene
from ...clipgt import ClipgtGpuScene, load_clipgt_scene
from ...convert import convert_cameras, convert_timestamped_scene

__all__ = [
    "LudusRenderer",
    "Av2GpuScene",
    "load_av2_scene",
    "load_clipgt_scene",
    "ClipgtGpuScene",
    "load_clipgt_scene",
    "convert_cameras",
    "convert_timestamped_scene",
]
