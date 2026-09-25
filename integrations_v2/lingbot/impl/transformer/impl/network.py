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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)

from .modules import (
    CamCtrlBlock,
    CamCtrlBlockCache,
    OptimizedSelfAttention,
    _rowwise_fp8_linear,
)


class _DynamicFP8Linear(nn.Module):
    """Apply rowwise dynamic FP8 inference through FlashDreams Accelerated."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.projection = QuantizedNonPersistentLinear(
            linear.weight,
            linear.bias,
            WeightGranularity.PER_OUT_CHANNEL,
            torch.float8_e4m3fn,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project ``x`` with rowwise FP8 activations and weights."""
        return _rowwise_fp8_linear(self.projection, x)


def _replace_large_block_linears_with_fp8(module: nn.Module) -> None:
    """Replace aligned large linear descendants with dynamic FP8 projections."""
    for parent in list(module.modules()):
        if isinstance(parent, OptimizedSelfAttention):
            continue
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if (
                child.in_features < 1024
                or child.out_features < 1024
                or child.in_features % 16
                or child.out_features % 16
            ):
                continue
            setattr(parent, name, _DynamicFP8Linear(child))


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
    linear_backend: Literal["torch", "rowwise_fp8"] = "torch"
    """Backend for large transformer-block linear projections."""

    self_attention_backend: Literal["wan", "fp8_tma", "scaled_fp8"] = "wan"
    """Self-attention implementation used by each transformer block."""

    self_attention_use_tma: bool = True
    """Prefer TMA when the selected FP8 attention backend supports it."""


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
        self.linear_backend = config.linear_backend
        self.self_attention_backend = config.self_attention_backend
        self.self_attention_use_tma = config.self_attention_use_tma
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

    def update_parameters_after_loading_checkpoint(self) -> None:
        """Finalize checkpoint parameters and derive optional FP8 block weights."""
        if self._parameters_updated_after_loading_checkpoint:
            return
        super().update_parameters_after_loading_checkpoint()
        if self.linear_backend == "rowwise_fp8":
            _replace_large_block_linears_with_fp8(self.blocks)

    def _build_block(self, layer_idx: int) -> CamCtrlBlock:
        return CamCtrlBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            cp_method=self.cp_method,
            self_attention_backend=self.self_attention_backend,
            self_attention_use_tma=self.self_attention_use_tma,
        )

    @torch.no_grad()
    def prepare_camera_cache(
        self,
        plucker: Tensor,
        cache: LingbotWorldDiTNetworkCache,
        shared_cache: LingbotWorldDiTNetworkCache | None = None,
    ) -> None:
        """Cache per-block camera modulation for one autoregressive chunk.

        Args:
            plucker: Patchified camera-control tensor shaped ``[..., L, D_p]``.
            cache: Conditional network cache to refresh.
            shared_cache: Optional CFG-unconditional cache that shares the
                resulting immutable camera tensors.
        """
        plucker_embedding = self.patch_embedding_wancamctrl(plucker)
        plucker_hidden_states = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(plucker_embedding))
        )
        plucker_embedding = plucker_embedding + plucker_hidden_states

        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, CamCtrlBlock)
            block_cache = cache[block_idx]
            assert isinstance(block_cache, CamCtrlBlockCache)
            block.prepare_camera_cache(plucker_embedding, block_cache)
            if shared_cache is not None:
                shared_block_cache = shared_cache[block_idx]
                assert isinstance(shared_block_cache, CamCtrlBlockCache)
                shared_block_cache.camera_scale = block_cache.camera_scale
                shared_block_cache.camera_shift = block_cache.camera_shift

    def replace_text_embeddings(
        self,
        cache: LingbotWorldDiTNetworkCache,
        text_embeddings: Tensor,
    ) -> None:
        """Replace cached cross-attention text K/V for all blocks.

        Text events are represented as alternate UMT5 embeddings. The
        self-attention KV cache stays intact so the rollout horizon is
        preserved; only the static cross-attention text context changes.
        """
        context_text = self.text_embedding(text_embeddings)
        for block, block_cache in zip(self.blocks, cache.block_caches):
            assert isinstance(block, CamCtrlBlock)
            block_cache.cross_attn.text = block.cross_attn.compute_kv(context_text)
