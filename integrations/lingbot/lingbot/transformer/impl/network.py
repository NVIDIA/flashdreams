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
from math import prod
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    sinusoidal_embedding_1d,
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


def pipeline_partition_bounds(
    num_layers: int,
    *,
    stage_index: int,
    stage_count: int,
) -> tuple[int, int]:
    """Return the balanced half-open layer range for one pipeline stage.

    Args:
        num_layers: Total transformer-block count.
        stage_index: Zero-based pipeline-stage index.
        stage_count: Number of pipeline stages.

    Returns:
        Global ``(start, end)`` layer indices assigned to the stage.

    Raises:
        ValueError: The stage layout is empty or outside the valid range.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    if stage_count < 1 or stage_count > num_layers:
        raise ValueError(
            f"stage_count must be in [1, {num_layers}], got {stage_count}."
        )
    if stage_index < 0 or stage_index >= stage_count:
        raise ValueError(
            f"stage_index must be in [0, {stage_count}), got {stage_index}."
        )

    base, remainder = divmod(num_layers, stage_count)
    start = stage_index * base + min(stage_index, remainder)
    end = start + base + (1 if stage_index < remainder else 0)
    return start, end


class LingbotWorldDiTNetwork(WanDiTNetwork):
    """Lingbot World DiT diffusion backbone for text-to-video and image-to-video."""

    def __init__(self, config: LingbotWorldDiTNetworkConfig) -> None:
        super().__init__(config)

        self._pipeline_stage_index: int | None = None
        self._pipeline_stage_count = 1
        self._pipeline_group: ProcessGroup | None = None
        self._pipeline_ranks: tuple[int, ...] = ()
        self._global_layer_range = (0, config.num_layers)

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

    def configure_pipeline_parallel(
        self,
        *,
        stage_index: int,
        stage_count: int,
        group: ProcessGroup,
        ranks: tuple[int, ...],
    ) -> None:
        """Partition resident layers and bind the NCCL pipeline group.

        The checkpoint is loaded on CPU before this method runs. Unowned blocks
        and endpoint-only modules are removed before the stage is moved to its
        GPU, so peak device memory reflects the local partition.

        Args:
            stage_index: Zero-based position in the pipeline group.
            stage_count: Number of ranks in the pipeline group.
            group: NCCL process group spanning the pipeline ranks.
            ranks: Global ranks in pipeline order.

        Raises:
            ValueError: The rank layout does not describe a two-stage group.
            RuntimeError: Pipeline parallelism was already configured.
        """
        if self._pipeline_stage_index is not None:
            raise RuntimeError("Pipeline parallelism is already configured.")
        if stage_count != 2 or len(ranks) != stage_count:
            raise ValueError(
                "LingBot pipeline parallelism currently requires two ranks."
            )
        if torch.distributed.get_rank() != ranks[stage_index]:
            raise ValueError(
                f"Global rank {torch.distributed.get_rank()} does not match "
                f"stage_index={stage_index} in ranks={ranks}."
            )

        global_num_layers = len(self.blocks)
        start, end = pipeline_partition_bounds(
            global_num_layers,
            stage_index=stage_index,
            stage_count=stage_count,
        )
        self.blocks = nn.ModuleList(list(self.blocks[start:end]))
        self.num_layers = len(self.blocks)
        self._global_layer_range = (start, end)
        self._pipeline_stage_index = stage_index
        self._pipeline_stage_count = stage_count
        self._pipeline_group = group
        self._pipeline_ranks = ranks

        if stage_index == 0:
            del self.head
        else:
            del self.patch_embedding

    def _build_block(self, layer_idx: int) -> CamCtrlBlock:
        return CamCtrlBlock(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            cp_method=self.cp_method,
        )

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

    def _pipeline_embeddings(
        self,
        *,
        x: Tensor,
        plucker: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build local camera and timestep embeddings for a pipeline stage."""
        batch_shape = x.shape[:-2]
        token_count = x.shape[-2]
        per_token_timestep = (
            timesteps.ndim > len(batch_shape)
            and timesteps.shape[-1] == token_count
        )
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
        if per_token_timestep:
            block_e_shape = batch_shape + (token_count, 6, self.dim)
            head_e = torch.broadcast_to(
                e,
                batch_shape + (token_count, self.dim),
            ).unsqueeze(-2)
        else:
            block_e_shape = batch_shape + (6, self.dim)
            head_e = torch.broadcast_to(
                e,
                batch_shape + (self.dim,),
            ).unsqueeze(-2).unsqueeze(-2)
        block_e = torch.broadcast_to(e0, block_e_shape)

        plucker_embedding = self.patch_embedding_wancamctrl(plucker)
        plucker_hidden_states = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(plucker_embedding))
        )
        return plucker_embedding + plucker_hidden_states, block_e, head_e

    def _forward_pipeline(
        self,
        *,
        plucker: Tensor,
        x: Tensor,
        timesteps: Tensor,
        cache: LingbotWorldDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int,
        eager_mode: bool,
    ) -> Tensor:
        """Run one local layer partition and exchange boundary activations."""
        assert self._pipeline_stage_index is not None
        assert self._pipeline_group is not None
        peer_rank = self._pipeline_ranks[1 - self._pipeline_stage_index]

        plucker_embedding, block_e, head_e = self._pipeline_embeddings(
            x=x,
            plucker=plucker,
            timesteps=timesteps,
        )
        if self._pipeline_stage_index == 0:
            if self.patch_embedding_type == "linear":
                hidden = self.patch_embedding(x)
            elif self.patch_embedding_type == "conv3d":
                weight = self.patch_embedding.weight.reshape(self.dim, -1)
                hidden = F.linear(x, weight, self.patch_embedding.bias)
            else:
                raise ValueError(
                    f"Invalid patch embedding type: {self.patch_embedding_type}"
                )
        else:
            hidden = torch.empty(
                *x.shape[:-1],
                self.dim,
                dtype=x.dtype,
                device=x.device,
            )
            torch.distributed.recv(
                hidden,
                src=peer_rank,
                group=self._pipeline_group,
            )

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            hidden = block(
                x=hidden,
                e=block_e,
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                plucker_embedding=plucker_embedding,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        if self._pipeline_stage_index == 0:
            torch.distributed.send(
                hidden.contiguous(),
                dst=peer_rank,
                group=self._pipeline_group,
            )
            output = torch.empty(
                *x.shape[:-1],
                self.out_dim * prod(self.patch_size),
                dtype=x.dtype,
                device=x.device,
            )
            torch.distributed.recv(
                output,
                src=peer_rank,
                group=self._pipeline_group,
            )
            return output

        output = self.head(hidden, head_e)
        torch.distributed.send(
            output.contiguous(),
            dst=peer_rank,
            group=self._pipeline_group,
        )
        return output

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

        if self._pipeline_stage_index is not None:
            return self._forward_pipeline(
                plucker=plucker,
                x=x,
                timesteps=timesteps,
                cache=cache,
                rope_freqs=rope_freqs,
                current_chunk_idx=current_chunk_idx,
                eager_mode=eager_mode,
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
