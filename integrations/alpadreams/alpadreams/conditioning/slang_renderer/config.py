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

"""Tunable knobs for the SlangPy HD-map rasterizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComputeDeviceName = Literal["automatic", "cuda", "vulkan"]


@dataclass(frozen=True)
class RasterConfig:
    """Per-camera rasterizer configuration.

    The defaults are the same as the ``RasterConfig`` in roaddreams: tuned for
    HD-map conditioning frames at 1280x704. ``width``/``height`` are filled in
    by :class:`alpadreams.conditioning.renderer.LudusRenderer` from the camera
    resolution at construction time.
    """

    width: int = 1280
    height: int = 704
    compute_device: ComputeDeviceName = "cuda"
    sync_gpu_timing: bool = False
    near_plane_m: float = 0.1
    far_plane_m: float = 200.0
    fog_start_m: float = 40.0
    fog_end_m: float = 140.0
    fog_power: float = 1.5
    triangle_raytrace_distance_m: float = 25.0
    lane_segment_interval_m: float = 0.05
    polyline_segment_interval_m: float = 0.8
    line_width_px: float = 12.0
    pole_width_px: float = 5.0
    dual_line_offset_m: float = 0.10
    depth_clear_m: float = 1.0e6
