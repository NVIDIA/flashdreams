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
