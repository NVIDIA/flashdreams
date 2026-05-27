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

"""Cosmos DiT network for streaming omnidreams inference."""

import re
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)
from flashdreams.infra.config import InstantiateConfig

from .modules import (
    Block,
    BlockCache,
    FinalLayer,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
)

# Cosmos-Predict2 training-side checkpoints wrap the network in a ``net``
# attribute, prefixing every key with ``net.``. Strip that prefix to land
# in the bare ``CosmosDiTNetwork.state_dict()`` layout.
CHECKPOINT_KEY_MAPPING: dict[str, str] = {
    r"^net\.(.*)$": r"\1",
}

# Keys that exist only in the training-side checkpoint and have no
# inference counterpart: Transformer-Engine internal state, learned
# positional embeddings now derived at runtime from RoPE, and the
# training-loop bookkeeping accumulators.
CHECKPOINT_KEY_DROP_PATTERNS: tuple[str, ...] = (
    r".*_extra_state.*",
    r".*pos_embedder\..*",
    r"^accum_video_sample_counter$",
    r"^accum_image_sample_counter$",
    r"^accum_iteration$",
    r"^accum_train_in_hours$",
)


def state_dict_transform(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remap a Cosmos-Predict2 training-side state-dict to the inference layout."""
    remapped = remap_checkpoint_keys(state_dict, CHECKPOINT_KEY_MAPPING)
    return {
        k: v
        for k, v in remapped.items()
        if not any(re.match(p, k) for p in CHECKPOINT_KEY_DROP_PATTERNS)
    }


@dataclass
class CosmosDiTNetworkCache:
    """Cache container for all transformer blocks."""

    block_caches: list[BlockCache]
    """Per-transformer-block self-attn KV + cross-attn KV cache, indexed by block position."""

    def __getitem__(self, index: int) -> BlockCache:
        """Get cache for a specific block."""
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        """Run pre-update hooks for all block caches."""
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run post-update hooks for all block caches."""
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


@dataclass
class CosmosDiTNetworkConfig(InstantiateConfig):
    """Configuration for the Cosmos DiT network."""

    _target: type["CosmosDiTNetwork"] = field(default_factory=lambda: CosmosDiTNetwork)

    in_channels: int = 17
    """Number of input latent channels before patch embedding. (16 + 1 for the condition mask)"""

    out_channels: int = 16
    """Output latent channels after the final layer."""

    patch_size: tuple[int, int, int] = (1, 2, 2)
    """Patch size for the input tensor."""

    model_channels: int = 2048
    """Transformer hidden size (width)."""

    num_blocks: int = 28
    """Number of transformer blocks."""

    num_heads: int = 16
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """FFN inner-dim multiplier relative to ``model_channels``."""

    concat_padding_mask: bool = True
    """If ``True``, expect a padding mask channel concatenated to the input at training."""

    use_adaln_lora: bool = True
    """If ``True``, factorize AdaLN modulation through a low-rank LoRA path."""

    adaln_lora_dim: int = 256
    """Rank of the AdaLN LoRA factorization when ``use_adaln_lora`` is ``True``."""

    use_crossattn_projection: bool = True
    """If ``True``, project text embeddings through a linear before cross-attention."""

    crossattn_proj_in_channels: int = 100352
    """Input dimension of the optional cross-attention projection."""

    crossattn_emb_channels: int = 1024
    """Cross-attention key/value dimension."""

    timestep_scale: float = 0.001
    """Multiplier applied to raw timestep values before sinusoidal embedding."""
    cp_method: Literal["ring", "ulysses"] = "ring"
    """Context-parallel attention method for transformer attention ops."""


class CosmosDiTNetwork(nn.Module):
    """DiT for video generation with block-causal attention and KV-caching.

    Combines the Cosmos DiT architecture with causal attention masking for
    autoregressive video generation.
    """

    def __init__(self, config: CosmosDiTNetworkConfig):
        super().__init__()
        self.config = config

        # add 1 for the condition mask
        in_channels = config.in_channels
        # optionally add 1 for the padding mask
        if self.config.concat_padding_mask:
            in_channels += 1

        assert config.patch_size[1] == config.patch_size[2], (
            "Spatial patch size must be the same for height and width"
        )
        self.patch_temporal = config.patch_size[0]
        self.patch_spatial = config.patch_size[1]

        # Build embeddings
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.config.model_channels,
        )

        # Time embeddings
        self.t_embedder = nn.Sequential(
            Timesteps(self.config.model_channels),
            TimestepEmbedding(
                self.config.model_channels,
                self.config.model_channels,
                use_adaln_lora=self.config.use_adaln_lora,
            ),
        )
        self.t_embedding_norm = nn.RMSNorm(self.config.model_channels, eps=1e-6)

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=self.config.model_channels,
                    context_dim=self.config.crossattn_emb_channels,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    use_adaln_lora=self.config.use_adaln_lora,
                    adaln_lora_dim=self.config.adaln_lora_dim,
                    cp_method=self.config.cp_method,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=self.config.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.config.out_channels,
            use_adaln_lora=self.config.use_adaln_lora,
            adaln_lora_dim=self.config.adaln_lora_dim,
        )

        if self.config.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(
                    self.config.crossattn_proj_in_channels,
                    self.config.crossattn_emb_channels,
                    bias=True,
                ),
                nn.GELU(),
            )

        self._is_shuffle_op_fused = False
        self._is_padding_mask_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def set_context_parallel_group(self, cp_group: ProcessGroup | None = None) -> None:
        for block in self.blocks:
            assert isinstance(block, Block)
            block.set_context_parallel_group(cp_group)

    def _fuse_shuffle_op_into_last_layer(self):
        """Fuse the channel-shuffle that follows the last linear into its weights.

        In the Cosmos model the patchify pattern is
        ``b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)`` while the
        unpatchify pattern is
        ``b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)``. This mismatch
        (likely a Cosmos bug) means the last dimension must be shuffled after
        the network. Folding that shuffle into ``final_layer.linear`` removes
        the explicit ``rearrange`` from the inference path.

        Calling this once is equivalent to running the following after the
        last layer::

            x = rearrange(
                x,
                "... (kt kh kw c) -> ... (c kt kh kw)",
                kt=self.patch_temporal,
                kh=self.patch_spatial,
                kw=self.patch_spatial,
                c=self.out_channels,
            )
        """
        if self._is_shuffle_op_fused:
            return

        self.final_layer.linear.weight.data = rearrange(
            self.final_layer.linear.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
            c=self.config.out_channels,
        ).contiguous()
        if self.final_layer.linear.bias is not None:
            self.final_layer.linear.bias.data = rearrange(
                self.final_layer.linear.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.patch_temporal,
                kh=self.patch_spatial,
                kw=self.patch_spatial,
                c=self.config.out_channels,
            ).contiguous()

        self._is_shuffle_op_fused = True
        return

    def _fuse_padding_mask_into_patch_embed(self) -> None:
        """Fold the always-zero inference padding mask into ``x_embedder`` in place.

        When ``self.concat_padding_mask`` is ``True`` training concatenates a
        ``[B, 1, T, H, W]`` padding mask to the input on the channel dimension
        before ``x_embedder`` (``1`` marks padded regions for variable spatial
        resolutions). At inference the mask is always zero, so the matching
        input channels of ``x_embedder`` can simply be dropped.

        Calling this once is equivalent to running the following before
        ``x_embedder``::

            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)
        """
        if not self.config.concat_padding_mask:
            return

        if self._is_padding_mask_fused:
            return

        self.x_embedder.in_channels -= 1
        in_channels_to_keep = self.x_embedder.get_linear_in_channels()
        proj_linear = self.x_embedder.proj[1]
        assert isinstance(proj_linear, nn.Linear)
        proj_linear.weight.data = proj_linear.weight.data[
            :, :in_channels_to_keep
        ].contiguous()
        if proj_linear.bias is not None:
            proj_linear.bias.data = proj_linear.bias.data[
                :in_channels_to_keep
            ].contiguous()

        self._is_padding_mask_fused = True
        return

    def update_parameters_after_loading_checkpoint(self) -> None:
        """Fuse load-time-known ops into weights; call once after loading the checkpoint."""
        if self._parameters_updated_after_loading_checkpoint:
            return

        self._fuse_padding_mask_into_patch_embed()
        self._fuse_shuffle_op_into_last_layer()
        self._parameters_updated_after_loading_checkpoint = True

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,  # [..., T, C, H, W]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        """Patchify and optionally CP-split the input video tensor.

        The patchify pattern is
        ``... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)``.

        Returns:
            Patched tensor with shape ``[..., L, D]`` where
            ``L = T * H * W / (kt * kh * kw)``.
        """
        assert x.ndim >= 4, (
            f"x must have at least 4 trailing dims (T, C, H, W) "
            f"plus zero-or-more leading batch dims, but got shape {x.shape}"
        )

        x = rearrange(
            x,
            "... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
        )

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided "
                "and the number of dimensions must match the number of process groups"
            )
            for cp_dim, process_group in zip(cp_dims, process_groups):
                if process_group is not None:
                    assert cp_dim is not None, (
                        "Context parallel dimension must be provided if process group is provided"
                    )
                    x = split_inputs_cp(x, seq_dim=cp_dim, cp_group=process_group)
        return x

    def unpatchify_and_maybe_gather_cp(
        self,
        pH: int,
        pW: int,
        x: Tensor,  # [..., L, D]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        """Unpatchify and optionally CP-gather the tensor back to video shape.

        The unpatchify pattern is
        ``... (t h w) (c kt kh kw) -> ... (t kt) c (h kh) (w kw)``.

        Returns:
            Unpatched tensor with shape ``[..., T, C, H, W]``.
        """
        assert x.ndim >= 2, f"x must be a 2D or higher tensor, but got shape {x.shape}"

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided "
                "and the number of dimensions must match the number of process groups"
            )
            for cp_dim, process_group in zip(cp_dims, process_groups):
                if process_group is not None:
                    assert cp_dim is not None, (
                        "Context parallel dimension must be provided if process group is provided"
                    )
                    x = cat_outputs_cp(x, seq_dim=cp_dim, cp_group=process_group)

        x = rearrange(
            x,
            "... (t h w) (c kt kh kw) -> ... (t kt) c (h kh) (w kw)",
            h=pH,
            w=pW,
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
        )
        return x  # [..., T, C, H, W]

    def initialize_cache(
        self,
        # self attn
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross attn
        text_embeddings: Tensor,
    ) -> CosmosDiTNetworkCache:
        """Build a fresh autoregressive cache for the DiT given the chunk geometry."""
        if self.config.use_crossattn_projection:
            context = self.crossattn_proj(text_embeddings)
        else:
            context = text_embeddings

        block_caches: list[BlockCache] = []
        for block in self.blocks:
            assert isinstance(block, Block)
            block_caches.append(
                block.initialize_cache(chunk_size, window_size, sink_size, context)
            )
        return CosmosDiTNetworkCache(block_caches=block_caches)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: CosmosDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
    ) -> Tensor:
        """Run one denoising forward pass.

        Args:
            x: Patchified video tokens of shape ``[..., L, D_in]``;
                layout ``"... (t h w) (c kt kh kw)"``.
            timesteps: Scalar shape ``()`` or per-token shape ``[L]``.
            rope_freqs: RoPE cosine/sine embeddings of shape
                ``[L, 1, 1, head_dim // 2]``.
            cache: Per-block autoregressive cache produced by
                :meth:`initialize_cache`.
            current_chunk_idx: Current chunk index in autoregressive inference.
            eager_mode: ``True`` runs cache pre/post-update inside the forward;
                ``False`` expects the caller to drive ``before_update`` /
                ``after_update`` outside the (graph-captured) network.

        Returns:
            Network output, shape ``[..., L, prod(patch_size) * out_channels]``.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )

        timesteps = timesteps * self.config.timestep_scale
        batch_shape = x.shape[:-2]
        L = x.shape[-2]

        # Patch embedding
        x = self.x_embedder(x)

        # Time embedding
        if timesteps.ndim == 0:
            L_slot = 1
        else:
            assert timesteps.shape == batch_shape + (L,), (
                f"per-token timesteps shape {tuple(timesteps.shape)} must equal "
                f"batch_shape + (L,) = {tuple(batch_shape + (L,))}"
            )
            L_slot = L
        t_emb, adaln_lora = self.t_embedder(timesteps)
        t_emb = self.t_embedding_norm(t_emb)
        t_emb = torch.broadcast_to(t_emb, batch_shape + (L_slot, t_emb.shape[-1]))
        if adaln_lora is not None:
            adaln_lora = torch.broadcast_to(
                adaln_lora, batch_shape + (L_slot, adaln_lora.shape[-1])
            )

        # In non-eager mode the caller drives ``before_update``/``after_update``
        # outside the (graph-captured) network forward.
        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x = block(
                x=x,
                emb=t_emb,
                rope_freqs=rope_freqs,
                adaln_lora=adaln_lora,
                cache=cache[block_idx],
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final layer
        x = self.final_layer(x, t_emb, adaln_lora)
        return x


# uv run -m flashdreams.recipes.cosmos.transformer.impl.network
if __name__ == "__main__":
    import tyro

    config = tyro.cli(CosmosDiTNetworkConfig)
    network = config.setup()
    print("network parameters:", sum(p.numel() for p in network.parameters()))

    ckpt_path = "/root/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/15a82a2ec231bc318692aa0456a36537c806e7d4/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
    state_dict_raw = torch.load(ckpt_path)
    state_dict = state_dict_transform(state_dict_raw)
    network.load_state_dict(state_dict)
    print("network loaded")
