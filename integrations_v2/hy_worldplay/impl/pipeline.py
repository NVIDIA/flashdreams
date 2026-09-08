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

"""HY-WorldPlay pipeline configuration for camera-controlled inference."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineConfig,
)


@dataclass(kw_only=True)
class HyWorldPlayPipelineConfig(WanInferencePipelineConfig):
    """Model and memory-selection configuration for HY-WorldPlay."""

    _target: type[WanInferencePipeline] = field(
        default_factory=lambda: WanInferencePipeline
    )

    memory_seed: int = 0
    """Seed used to sample the memory-selection point cloud."""

    context_window_length: int = 16
    """Frame count below which FOV-based memory selection is bypassed."""

    memory_frames: int = 16
    """Total memory-frame budget for each autoregressive step."""

    temporal_context_size: int = 12
    """Recent-frame portion retained unconditionally."""

    memory_pred_latent_size: int = 4
    """Query clip size used by the FOV-overlap scorer."""

    memory_fov_h_deg: float = 60.0
    """Horizontal field of view used for memory selection, in degrees."""

    memory_fov_v_deg: float = 35.0
    """Vertical field of view used for memory selection, in degrees."""

    memory_points_count: int = 50_000
    """Number of Monte Carlo points used by the overlap scorer."""

    memory_points_radius: float = 8.0
    """Radius of the memory-selection point cloud."""


__all__ = ["HyWorldPlayPipelineConfig"]
