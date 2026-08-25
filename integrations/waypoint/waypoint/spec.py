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

"""Waypoint 1.5 checkpoint configuration contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class WaypointModelSpec:
    """Architecture and rollout invariants for a published Waypoint checkpoint."""

    model_id: str
    """Published Hugging Face checkpoint identifier."""
    channels: int
    """Number of channels in a single latent frame."""
    latent_height: int
    """Pre-patchify latent-frame height."""
    latent_width: int
    """Pre-patchify latent-frame width."""
    patch_height: int
    """Spatial patch height."""
    patch_width: int
    """Spatial patch width."""
    temporal_compression: int
    """Presented RGB frames emitted per autoregressive latent frame."""
    inference_fps: int
    """Presented RGB frame rate."""
    base_fps: int
    """Timestamp base rate expected by the checkpoint."""
    n_layers: int
    """Number of transformer blocks."""
    d_model: int
    """Transformer channel width."""
    n_heads: int
    """Number of query-attention heads."""
    n_kv_heads: int
    """Number of key/value attention heads."""
    mlp_ratio: int
    """Hidden-width multiplier of each transformer feed-forward network."""
    n_buttons: int
    """Size of the multi-hot button-control vocabulary."""
    local_window: int
    """Recent latent-frame capacity of local attention layers."""
    global_window: int
    """Latent-frame horizon of global attention layers."""
    global_pinned_dilation: int
    """Temporal stride of pinned history in global-attention layers."""
    global_attention_period: int
    """Stride between global-attention transformer blocks."""
    global_attention_offset: int
    """Offset used to select global-attention transformer blocks."""
    controller_conditioning_period: int
    """Stride between transformer blocks with controller fusion weights."""
    value_residual: bool
    """Whether attention values carry a residual stream across blocks."""
    gated_attention: bool
    """Whether attention outputs use an additional learned gate."""
    noise_conditioning: str
    """Published noise-conditioning family used by the checkpoint."""
    rope_theta: float
    """Base frequency of the geometric temporal rotary spectrum."""
    rope_nyquist_fraction: float
    """Fraction of the spatial Nyquist limit used by rotary features."""
    scheduler_sigmas: tuple[float, ...]
    """Fixed rectified-flow Euler schedule, including the terminal sigma."""
    text_conditioning: bool
    """Whether the checkpoint provides a text-conditioning input."""

    @property
    def head_dim(self) -> int:
        """Return the channel dimension of an attention head."""
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def tokens_per_latent_frame(self) -> int:
        """Return the number of spatial tokens generated for one action."""
        assert self.latent_height % self.patch_height == 0
        assert self.latent_width % self.patch_width == 0
        return (self.latent_height // self.patch_height) * (
            self.latent_width // self.patch_width
        )

    @property
    def patch_grid_height(self) -> int:
        """Return the number of patch tokens along the latent-image height."""
        return self.latent_height // self.patch_height

    @property
    def patch_grid_width(self) -> int:
        """Return the number of patch tokens along the latent-image width."""
        return self.latent_width // self.patch_width

    @property
    def frames_per_action(self) -> int:
        """Return the number of presented RGB frames decoded from one latent frame."""
        return self.temporal_compression

    @property
    def num_denoising_steps(self) -> int:
        """Return the number of Euler velocity evaluations in the fixed schedule."""
        return len(self.scheduler_sigmas) - 1

    @property
    def latent_fps(self) -> int:
        """Return the autoregressive latent-frame rate."""
        assert self.inference_fps % self.temporal_compression == 0
        return self.inference_fps // self.temporal_compression

    @property
    def frame_timestamp_stride(self) -> int:
        """Return the model timestamp increment per latent frame."""
        assert self.base_fps % self.latent_fps == 0
        return self.base_fps // self.latent_fps

    @property
    def global_attention_layers(self) -> tuple[int, ...]:
        """Return zero-indexed transformer layers that use global cache policy."""
        return tuple(
            index
            for index in range(self.n_layers)
            if (index - self.global_attention_offset) % self.global_attention_period
            == 0
        )

    def latent_shape(self, batch_size: int = 1) -> tuple[int, int, int, int, int]:
        """Return the pre-patchify DiT shape for one autoregressive action."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return (batch_size, 1, self.channels, self.latent_height, self.latent_width)


WAYPOINT_1_5 = WaypointModelSpec(
    model_id="Overworld/Waypoint-1.5-1B",
    channels=32,
    latent_height=32,
    latent_width=64,
    patch_height=2,
    patch_width=2,
    temporal_compression=4,
    inference_fps=60,
    base_fps=15,
    n_layers=24,
    d_model=2048,
    n_heads=32,
    n_kv_heads=16,
    mlp_ratio=4,
    n_buttons=256,
    local_window=16,
    global_window=128,
    global_pinned_dilation=8,
    global_attention_period=4,
    global_attention_offset=-1,
    controller_conditioning_period=3,
    value_residual=True,
    gated_attention=False,
    noise_conditioning="wan",
    rope_theta=10_000.0,
    rope_nyquist_fraction=0.8,
    scheduler_sigmas=(1.0, 0.9, 0.75, 0.3, 0.0),
    text_conditioning=False,
)
"""Static contract for the published ``Overworld/Waypoint-1.5-1B`` checkpoint."""
