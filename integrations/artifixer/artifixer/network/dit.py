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

"""ArtiFixer DiT network: ``WanDiTNetwork`` with opacity / camera blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from artifixer.network.block import ArtifixerBlock
from torch import Tensor

from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetwork1pt3BConfig,
    WanDiTNetworkConfig,
)

# Wan2.1 VAE downsampling factors. Identical for T2V-1.3B and T2V-14B.
# Used to compute the per-block opacity / camera input dims.
WAN_VAE_SCALE_FACTOR_SPATIAL = 8
WAN_VAE_SCALE_FACTOR_TEMPORAL = 4


def artifixer_embedding_dims(
    patch_size: tuple[int, int, int],
    vae_scale_factor_spatial: int = WAN_VAE_SCALE_FACTOR_SPATIAL,
    vae_scale_factor_temporal: int = WAN_VAE_SCALE_FACTOR_TEMPORAL,
) -> tuple[int, int]:
    """Compute (opacity_embedding_dim, camera_embedding_dim) for ArtiFixer.

      opacity_embedding_dim = vae_t * vae_s * ph * vae_s * pw
      camera_embedding_dim  = vae_s * ph * vae_s * pw * 6   (6 = Plucker channels)

    For Wan 2.1 default ``patch_size = (1, 2, 2)`` with VAE strides
    ``(4, 8, 8)`` this gives ``(1024, 1536)``.
    """
    _, ph, pw = patch_size
    opacity_dim = (
        vae_scale_factor_temporal
        * vae_scale_factor_spatial
        * ph
        * vae_scale_factor_spatial
        * pw
    )
    camera_dim = vae_scale_factor_spatial * ph * vae_scale_factor_spatial * pw * 6
    return opacity_dim, camera_dim


@dataclass
class ArtifixerDiTNetworkConfig(WanDiTNetworkConfig):
    """ArtiFixer DiT network config.

    Inherits every field from ``WanDiTNetworkConfig`` and overrides
    ``_target`` to construct an :class:`ArtifixerDiTNetwork`. The
    opacity / camera input dims are derived from the VAE strides + patch
    size via :func:`artifixer_embedding_dims`, not stored on the config,
    so they cannot drift from the model that produced the conditioning
    tensors.
    """

    _target: type = field(default_factory=lambda: ArtifixerDiTNetwork)


@dataclass
class ArtifixerDiTNetwork1pt3BConfig(WanDiTNetwork1pt3BConfig):
    """ArtiFixer 1.3B variant (matches the ArtiFixer reference model size)."""

    _target: type = field(default_factory=lambda: ArtifixerDiTNetwork)


class ArtifixerDiTNetwork(WanDiTNetwork):
    """``WanDiTNetwork`` whose blocks are :class:`ArtifixerBlock`."""

    def _build_block(self, layer_idx: int) -> ArtifixerBlock:
        opacity_dim, camera_dim = artifixer_embedding_dims(self.patch_size)
        return ArtifixerBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            opacity_embedding_dim=opacity_dim,
            camera_embedding_dim=camera_dim,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            i2v=self.cross_attn_enable_img,
        )

    def initialize_neighbor_kv_caches(self, context: Tensor | None) -> None:
        """Push neighbor context into every block's cross-attention.

        Call once per rollout, after :meth:`initialize_cache`. ``context=None``
        clears the per-module caches so subsequent forward passes skip the
        PRoPE neighbor branch (vanilla T2V behavior).

        Args:
            context: Neighbor latent context shape
                ``[..., L_neighbor, dim]``. Same dim as the transformer
                hidden size; provided by the pipeline after VAE-encoding
                the neighbor frames and projecting them through
                ``patch_embedding``.
        """
        for block in self.blocks:
            assert isinstance(block, ArtifixerBlock)
            block.cross_attn.initialize_neighbor_cache(context)
