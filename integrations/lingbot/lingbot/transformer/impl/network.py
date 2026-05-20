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

"""Lingbot World DiT network: Wan 2.1 backbone with per-block camera control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)

from .modules import CamCtrlBlock


@dataclass
class LingbotWorldDiTNetworkCache(WanDiTNetworkCache):
    """Cache container for all transformer blocks."""


@dataclass
class LingbotWorldDiTNetworkConfig(WanDiTNetworkConfig):
    """Wan-sized hyperparameters plus Lingbot camera / action control."""

    _target: type["LingbotWorldDiTNetwork"] = field(
        default_factory=lambda: LingbotWorldDiTNetwork
    )
    control_type: Literal["cam", "act"] = "cam"


@dataclass
class LingbotWorldDiTNetwork1pt3BConfig(LingbotWorldDiTNetworkConfig):
    """Configuration for the 1.3B Lingbot World DiT network."""

    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30


@dataclass
class LingbotWorldDiTNetwork14BConfig(LingbotWorldDiTNetworkConfig):
    """Configuration for the 14B Lingbot World DiT network."""

    dim: int = 5120
    ffn_dim: int = 13824
    num_heads: int = 40
    num_layers: int = 40


class LingbotWorldDiTNetwork(WanDiTNetwork):
    """Lingbot World DiT diffusion backbone for text-to-video and image-to-video."""

    def __init__(self, config: LingbotWorldDiTNetworkConfig) -> None:
        super().__init__(config)

        if config.control_type == "cam":
            control_dim = 6
        elif config.control_type == "act":
            control_dim = 7
        else:
            raise ValueError(f"Invalid control type: {config.control_type}")
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim
            * 64
            * self.patch_size[0]
            * self.patch_size[1]
            * self.patch_size[2],
            self.dim,
        )
        self.c2ws_hidden_states_layer1 = nn.Linear(self.dim, self.dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(self.dim, self.dim)

    def _build_block(self, layer_idx: int) -> CamCtrlBlock:
        return CamCtrlBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
        )

    def forward(
        self,
        plucker: Tensor,
        x: Tensor,
        timesteps: Tensor,
        cache: LingbotWorldDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
    ) -> Tensor:
        """Run one denoising forward pass.

        Args:
            plucker: Camera-control Plücker embedding of shape
                ``[..., L, D_p]`` after patchify + CP.
            x: Input tokens of shape ``[..., L, D_in]`` after patchify
                + CP. Layout ``"... (t h w) (c kt kh kw)"``.
            timesteps: Diffusion timesteps broadcastable to ``[...]``.
            cache: Per-block KV caches.
            rope_freqs: RoPE frequencies of shape
                ``[L, 1, 1, head_dim // 2]`` after CP.
            current_chunk_idx: Current chunk index for streaming cache update.
            eager_mode: If True, run cache before/after update hooks.

        Returns:
            Network output, shape ``[..., L, prod(patch_size) * out_dim]``.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )

        plucker_embedding = self.patch_embedding_wancamctrl(plucker)
        plucker_hidden_states = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(plucker_embedding))
        )
        plucker_embedding = plucker_embedding + plucker_hidden_states

        return super().forward(
            x=x,
            timesteps=timesteps,
            cache=cache,
            rope_freqs=rope_freqs,
            current_chunk_idx=current_chunk_idx,
            eager_mode=eager_mode,
            block_extra_kwargs={"plucker_embedding": plucker_embedding},
        )
