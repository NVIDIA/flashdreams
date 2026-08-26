# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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
"""WanVADiTNetwork — native flashdreams DiT for LingBot-VA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.recipes.wan.transformer.impl.modules import (
    Head,
    sinusoidal_embedding_1d,
)
from lingbot_va.transformer.impl.kvcache import VAKVCache
from lingbot_va.transformer.impl.modules import VABlock, VABlockCache

VideoKV = tuple[tuple[Tensor, Tensor], ...]
"""Per-block video keys and values retained for the matching action branch."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WanVADiTNetworkConfig:
    """Network config for the LingBot-VA video-action DiT."""

    action_dim: int = 30
    action_per_frame: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 48
    out_dim: int = 48
    dim: int = 3072
    ffn_dim: int = 14336
    num_heads: int = 24
    num_layers: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    text_len: int = 512
    cross_attn_norm: bool = True
    eps: float = 1e-6
    apply_rope_before_kvcache: bool = True
    cp_method: Literal["ring", "ulysses"] = "ring"


# ---------------------------------------------------------------------------
# Network cache
# ---------------------------------------------------------------------------


@dataclass
class WanVADiTNetworkCache:
    """Per-block caches for the entire network."""

    block_caches: list[VABlockCache]

    def __getitem__(self, index: int) -> VABlockCache:
        return self.block_caches[index]


# ---------------------------------------------------------------------------
# RoPE helper
# ---------------------------------------------------------------------------


def compute_rope_freqs_from_grid(
    grid_id: Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> Tensor:
    """Compute RoPE frequencies from grid_id ``[3, L]`` (f, h, w).

    Returns ``[L, 1, 1, head_dim]`` for ``apply_rope_freqs(..., interleaved=True)``.
    """
    f_dim = head_dim - 2 * (head_dim // 3)
    h_dim = head_dim // 3
    w_dim = head_dim // 3
    device = grid_id.device

    f_base = 1.0 / (
        theta
        ** (
            torch.arange(0, f_dim, 2, device=device, dtype=torch.float64)[: f_dim // 2]
            / f_dim
        )
    )
    h_base = 1.0 / (
        theta
        ** (
            torch.arange(0, h_dim, 2, device=device, dtype=torch.float64)[: h_dim // 2]
            / h_dim
        )
    )
    w_base = 1.0 / (
        theta
        ** (
            torch.arange(0, w_dim, 2, device=device, dtype=torch.float64)[: w_dim // 2]
            / w_dim
        )
    )

    f_angles = grid_id[0].to(torch.float64).unsqueeze(-1) * f_base.unsqueeze(0)
    h_angles = grid_id[1].to(torch.float64).unsqueeze(-1) * h_base.unsqueeze(0)
    w_angles = grid_id[2].to(torch.float64).unsqueeze(-1) * w_base.unsqueeze(0)

    # Interleaved format: each angle duplicated as [theta_0, theta_0, theta_1, theta_1, ...]
    # to match RotaryPositionEmbedding3D._cat_freqs(interleaved=True)
    freqs = torch.cat(
        [
            f_angles.repeat_interleave(2, dim=-1),
            h_angles.repeat_interleave(2, dim=-1),
            w_angles.repeat_interleave(2, dim=-1),
        ],
        dim=-1,
    ).float()

    return freqs.unsqueeze(1).unsqueeze(1)  # [L, 1, 1, head_dim]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class WanVADiTNetwork(nn.Module):
    """Video-Action DiT network with native flashdreams building blocks."""

    def __init__(self, config: WanVADiTNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.out_dim = config.out_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.patch_size = config.patch_size
        self.eps = config.eps
        self.text_len = config.text_len
        self.head_dim = config.dim // config.num_heads

        # Video path embeddings
        in_channels = config.in_dim * math.prod(config.patch_size)
        self.patch_embedding = nn.Linear(in_channels, self.dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dim, self.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6),
        )

        # Action path embeddings
        self.action_embedder = nn.Linear(config.action_dim, self.dim)
        self.action_time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.action_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6),
        )
        self.action_text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dim, self.dim),
        )

        # Shared transformer blocks
        self.blocks = nn.ModuleList(
            [
                VABlock(
                    dim=self.dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_heads,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    apply_rope_before_kvcache=config.apply_rope_before_kvcache,
                    cp_method=config.cp_method,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Video output head
        self.head = Head(self.dim, config.out_dim, config.patch_size, config.eps)

        # Action output head (shares head.modulation for norm, separate projection)
        self.action_head = nn.Linear(self.dim, config.action_dim)

        self._parameters_updated_after_loading_checkpoint = False

    def update_parameters_after_loading_checkpoint(self) -> None:
        if self._parameters_updated_after_loading_checkpoint:
            return
        self._fuse_shuffle_op_into_last_layer()
        for block in self.blocks:
            assert isinstance(block, VABlock)
            block.update_parameters_after_loading_checkpoint()
        self.head.update_parameters_after_loading_checkpoint()
        self._parameters_updated_after_loading_checkpoint = True

    def _fuse_shuffle_op_into_last_layer(self) -> None:
        """Fuse channel shuffle into head.head weights (same as WanDiTNetwork)."""
        from einops import rearrange

        kt, kh, kw = self.patch_size
        self.head.head.weight.data = rearrange(
            self.head.head.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=kt,
            kh=kh,
            kw=kw,
            c=self.out_dim,
        ).contiguous()
        if self.head.head.bias is not None:
            self.head.head.bias.data = rearrange(
                self.head.head.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=kt,
                kh=kh,
                kw=kw,
                c=self.out_dim,
            ).contiguous()

    def initialize_cache(
        self,
        *,
        text_embeddings: Tensor,
        video_chunk: int,
        action_chunk: int,
        window_slots: int,
        batch_size: int,
    ) -> WanVADiTNetworkCache:
        """Build per-block caches."""
        context_text = self.text_embedding(text_embeddings)
        block_caches: list[VABlockCache] = []
        for block in self.blocks:
            assert isinstance(block, VABlock)
            self_attn_cache = VAKVCache.create(
                video_chunk=video_chunk,
                action_chunk=action_chunk,
                window_slots=window_slots,
                batch_size=batch_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                device=text_embeddings.device,
                dtype=text_embeddings.dtype,
            )
            cross_attn_cache = block.cross_attn.initialize_cache(context_text)
            block_caches.append(
                VABlockCache(
                    self_attn=self_attn_cache,
                    cross_attn=cross_attn_cache,
                )
            )
        return WanVADiTNetworkCache(block_caches=block_caches)

    def _extract_cache_tensors(
        self, cache: WanVADiTNetworkCache
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract committed KV and cross-attn text KV as plain tensors.

        Called outside the compiled graph to avoid BlockKVCache graph breaks.

        Returns:
            (committed_k_stack, committed_v_stack, cross_k_stack, cross_v_stack)
            All shapes: [num_layers, batch, seq_len, heads, head_dim]
        """
        committed_k = torch.stack(
            [
                bc.self_attn.kv_cache._k[:, : bc.self_attn.n_committed_tokens]
                for bc in cache.block_caches
            ]
        )
        committed_v = torch.stack(
            [
                bc.self_attn.kv_cache._v[:, : bc.self_attn.n_committed_tokens]
                for bc in cache.block_caches
            ]
        )
        cross_k = torch.stack(
            [
                bc.cross_attn.text._k[:, : bc.cross_attn.text._n_cached]
                for bc in cache.block_caches
            ]
        )
        cross_v = torch.stack(
            [
                bc.cross_attn.text._v[:, : bc.cross_attn.text._n_cached]
                for bc in cache.block_caches
            ]
        )
        return committed_k, committed_v, cross_k, cross_v

    def _forward_blocks_video(
        self,
        x: Tensor,
        timesteps: Tensor,
        committed_k: Tensor,
        committed_v: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Pure-tensor video forward through all blocks. Compile-friendly.

        Args:
            committed_k: [num_layers, batch, N, heads, head_dim]
            committed_v: same
            cross_k: [num_layers, batch, T, heads, head_dim]
            cross_v: same

        Returns:
            (head_output, k_fresh_list, v_fresh_list)
        """
        x = self.patch_embedding(x)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
        block_e = e0
        head_e = e.unsqueeze(-2)

        k_list: list[Tensor] = []
        v_list: list[Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            x, k_fresh, v_fresh = block(
                x,
                block_e,
                committed_k[block_idx],
                committed_v[block_idx],
                cross_k[block_idx],
                cross_v[block_idx],
                rope_freqs,
            )
            k_list.append(k_fresh)
            v_list.append(v_fresh)

        return self.head(x, head_e), k_list, v_list

    def _forward_blocks_action(
        self,
        x: Tensor,
        timesteps: Tensor,
        committed_k: Tensor,
        committed_v: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Pure-tensor action forward through all blocks. Compile-friendly.

        Returns:
            (action_output, k_fresh_list, v_fresh_list)
        """
        x = self.action_embedder(x)
        e = self.action_time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        e0 = self.action_time_projection(e).unflatten(-1, (6, self.dim))
        block_e = e0
        head_e = e.unsqueeze(-2)

        k_list: list[Tensor] = []
        v_list: list[Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            x, k_fresh, v_fresh = block(
                x,
                block_e,
                committed_k[block_idx],
                committed_v[block_idx],
                cross_k[block_idx],
                cross_v[block_idx],
                rope_freqs,
            )
            k_list.append(k_fresh)
            v_list.append(v_fresh)

        # Action output: shared modulation + separate projection
        e_chunks = [
            c.squeeze(-2) for c in (self.head.modulation + head_e).chunk(2, dim=-2)
        ]
        x = self.head.norm(x) * (1 + e_chunks[1]) + e_chunks[0]
        return self.action_head(x), k_list, v_list

    def forward_video(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanVADiTNetworkCache,
        rope_freqs: Tensor,
        persist: bool = False,
    ) -> tuple[Tensor, VideoKV | None]:
        """Video-mode forward.

        Cache extraction and writes happen outside the compiled block loop.

        Returns:
            Video flow ``[batch, L, prod(patch_size) * out_dim]`` and the
            per-block video KV written by a persistent call.
        """
        assert self._parameters_updated_after_loading_checkpoint

        # Extract cache tensors (outside compile boundary)
        committed_k, committed_v, cross_k, cross_v = self._extract_cache_tensors(cache)

        # Compiled block loop (pure tensors, no cache access)
        output, k_list, v_list = self._forward_blocks_video(
            x,
            timesteps,
            committed_k,
            committed_v,
            cross_k,
            cross_v,
            rope_freqs,
        )

        # Cache write (outside compile boundary)
        video_kv: VideoKV | None = None
        if persist:
            for block_idx, (k, v) in enumerate(zip(k_list, v_list)):
                cache[block_idx].self_attn.write_video(k, v)
            video_kv = tuple(zip(k_list, v_list))

        return output, video_kv

    def forward_action(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanVADiTNetworkCache,
        rope_freqs: Tensor,
        video_kv: VideoKV | None = None,
        persist: bool = False,
    ) -> Tensor:
        """Action-mode forward.

        Cache extraction, current-video concatenation, and writes happen outside
        the compiled block loop. Every action denoise pass attends to committed
        prior chunks, the matching branch's current video, and its fresh action
        tokens, in that order.

        Returns:
            Action flow ``[batch, L, action_dim]``.
        """
        assert self._parameters_updated_after_loading_checkpoint
        if video_kv is None:
            raise ValueError("Action attention requires current-chunk video KV.")
        if len(video_kv) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} video KV pairs, got {len(video_kv)}."
            )

        # Extract cache tensors (outside compile boundary)
        committed_k, committed_v, cross_k, cross_v = self._extract_cache_tensors(cache)
        video_k = torch.stack([key for key, _ in video_kv])
        video_v = torch.stack([value for _, value in video_kv])
        committed_k = torch.cat([committed_k, video_k], dim=2)
        committed_v = torch.cat([committed_v, video_v], dim=2)

        # Compiled block loop (pure tensors, no cache access)
        output, k_list, v_list = self._forward_blocks_action(
            x,
            timesteps,
            committed_k,
            committed_v,
            cross_k,
            cross_v,
            rope_freqs,
        )

        # Cache write (outside compile boundary)
        if persist:
            for block_idx, (k, v, branch_video_kv) in enumerate(
                zip(k_list, v_list, video_kv)
            ):
                vk, vv = branch_video_kv
                cache[block_idx].self_attn.write_action(k, v, vk, vv)

        return output
