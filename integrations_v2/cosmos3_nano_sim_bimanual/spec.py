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

"""Checkpoint geometry for the AgiBot mixture-of-transformers network."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMSpec:
    """Qwen3-VL-8B text-tower dims. Each layer is doubled into a MoT (und + gen)."""

    hidden_size: int = 4096
    num_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # grouped-query attention
    head_dim: int = 128
    intermediate_size: int = 12288  # SwiGLU gate/up/down
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    attention_bias: bool = False

    def __post_init__(self) -> None:
        assert self.num_attention_heads * self.head_dim == self.hidden_size, (
            f"heads*head_dim ({self.num_attention_heads}*{self.head_dim}) "
            f"!= hidden_size ({self.hidden_size})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads (GQA)"
        )
        assert sum(self.mrope_section) * 2 == self.head_dim, (
            f"2*sum(mrope_section) ({2 * sum(self.mrope_section)}) must equal "
            f"head_dim ({self.head_dim})"
        )


@dataclass(frozen=True)
class VisionSpec:
    """Qwen3-VL ViT tower dims (used for image-to-video conditioning only)."""

    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096  # merges to the LLM hidden size
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)


@dataclass(frozen=True)
class VFMSpec:
    """Video-foundation-model adapter and mRoPE dimensions."""

    # Wan 2.2 VAE (4x16x16): temporal /4, spatial /16.
    vae_latent_channels: int = 48  # exported config.json ``state_ch``
    latent_patch_size: int = 2  # DiT ``patch_spatial``

    # Diffusion timestep embedder: sinusoidal freq dim -> MLP into hidden.
    # Verified from the checkpoint: time_embedder.mlp.0 is Linear(256 -> 4096),
    # .2 is Linear(4096 -> 4096).
    time_embed_freq_dim: int = 256
    # Raw diffusion timesteps are multiplied by this before the sinusoidal embed.
    timestep_scale: float = 0.001

    # 3D mRoPE conventions (exported config.json ``diffusion_expert_config``).
    # The margin is added to the temporal offset at the text->vision boundary so
    # the two modalities occupy disjoint temporal bands: text temporal 0..N_text-1,
    # vision temporal N_text + margin + frame.
    mrope_temporal_modality_margin: int = 15000
    mrope_reset_spatial_ids: bool = True
    enable_fps_modulation: bool = True
    base_fps: float = 24.0

    @property
    def patch_latent_dim(self) -> int:
        """Post-patchify per-token channel width feeding ``vae2llm``."""
        return self.latent_patch_size**2 * self.vae_latent_channels


@dataclass(frozen=True)
class Cosmos3InteractiveSpec:
    """Architectural dimensions for the MoT-on-Qwen3-VL network."""

    llm: LLMSpec = field(default_factory=LLMSpec)
    vision: VisionSpec = field(default_factory=VisionSpec)
    vfm: VFMSpec = field(default_factory=VFMSpec)


SPEC = Cosmos3InteractiveSpec()
"""Module-level singleton — import this."""
