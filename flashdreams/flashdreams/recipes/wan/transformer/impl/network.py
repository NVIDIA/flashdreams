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

"""Wan 2.1 DiT network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)
from flashdreams.infra.config import InstantiateConfig
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    Head,
    MLPProj,
    sinusoidal_embedding_1d,
)


@dataclass
class WanDiTNetworkCache:
    """Cache container for all transformer blocks."""

    block_caches: list[BlockCache]
    """Per-transformer-block KV cache, indexed by block position."""

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
class WanDiTNetworkConfig(InstantiateConfig["WanDiTNetwork"]):
    """Configuration for the Wan DiT network."""

    _target: type["WanDiTNetwork"] = field(default_factory=lambda: WanDiTNetwork)

    patch_size: tuple[int, int, int] = (1, 2, 2)
    """Patch size for the input tensor."""
    text_len: int = 512
    """Maximum text token length."""
    in_dim: int = 16
    """Number of input latent channels before patch embedding."""
    dim: int = 1536
    """Transformer hidden size (width)."""
    ffn_dim: int = 8960
    """Feed-forward hidden dimension."""
    freq_dim: int = 256
    """Sinusoidal timestep embedding dimension."""
    text_dim: int = 4096
    """Text encoder output dimension."""
    out_dim: int = 16
    """Output latent channels after the head."""
    num_heads: int = 12
    """Number of attention heads."""
    num_layers: int = 30
    """Number of transformer blocks."""
    cross_attn_norm: bool = True
    """If True, apply ``LayerNorm`` before cross-attention."""
    cross_attn_enable_img: bool = False
    """If True, build image cross-attention and CLIP image projection (I2V)."""
    eps: float = 1e-6
    """Epsilon for normalization layers."""
    concat_padding_mask: bool = False
    """If True, concatenate one mask channel into the input channels."""
    patch_embedding_type: Literal["linear", "conv3d"] = "conv3d"
    """Type of patch embedding: ``"linear"`` (flattened patch MLP) or ``"conv3d"`` (strided conv)."""


@dataclass
class WanDiTNetwork1pt3BConfig(WanDiTNetworkConfig):
    """Configuration for the 1.3B Wan DiT network."""

    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30


@dataclass
class WanDiTNetwork14BConfig(WanDiTNetworkConfig):
    """Configuration for the 14B Wan DiT network."""

    dim: int = 5120
    ffn_dim: int = 13824
    num_heads: int = 40
    num_layers: int = 40


class WanDiTNetwork(nn.Module):
    """WAN diffusion backbone for text-to-video and image-to-video."""

    def __init__(self, config: WanDiTNetworkConfig) -> None:
        super().__init__()

        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.dim = config.dim
        self.ffn_dim = config.ffn_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.out_dim = config.out_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.cross_attn_norm = config.cross_attn_norm
        self.cross_attn_enable_img = config.cross_attn_enable_img
        self.eps = config.eps
        self.concat_padding_mask = config.concat_padding_mask
        self.patch_embedding_type = config.patch_embedding_type

        # Embedding layers
        in_dim = config.in_dim + 1 if self.concat_padding_mask else config.in_dim
        self.patch_embedding: nn.Linear | nn.Conv3d
        if config.patch_embedding_type == "linear":
            self.patch_embedding = nn.Linear(
                in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                self.dim,
            )
        elif config.patch_embedding_type == "conv3d":
            self.patch_embedding = nn.Conv3d(
                in_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        else:
            raise ValueError(
                f"Invalid patch embedding type: {config.patch_embedding_type}"
            )
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dim, self.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6)
        )
        if self.cross_attn_enable_img:
            self.img_emb = MLPProj(1280, self.dim)

        self.blocks = nn.ModuleList(
            [self._build_block(layer_idx) for layer_idx in range(self.num_layers)]
        )

        # Final projection head
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        self._is_shuffle_op_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def _build_block(self, layer_idx: int) -> Block:
        """Construct one transformer block."""
        return Block(
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps,
            i2v=self.cross_attn_enable_img,
        )

    def set_context_parallel_group(self, cp_group: ProcessGroup | None = None) -> None:
        """Set context-parallel process group for all blocks.

        This must be called before ``initialize_cache`` when CP is used.
        """
        for block in self.blocks:
            assert isinstance(block, Block)
            block.set_context_parallel_group(cp_group)

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
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
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
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
        )
        return x  # [..., T, C, H, W]

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        text_embeddings: Tensor,
        img_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
        """Initialize block caches from text/image context embeddings.

        Args:
            chunk_size: Number of tokens appended per self-attention update.
            window_size: Rolling-window size in tokens for self-attention cache.
            sink_size: Sink-token capacity preserved across updates.
            text_embeddings: Text embeddings. UMT5 has shape [..., 512, 4096].
            img_embeddings: Optional image embeddings for I2V. CLIP has shape [..., 256, 1280].

        Returns:
            ``WanDiTNetworkCache`` containing per-block caches.
        """
        assert text_embeddings.shape[-2] == self.text_len
        context_text = self.text_embedding(text_embeddings)
        if self.cross_attn_enable_img:
            assert img_embeddings is not None, (
                "img_embeddings is required when cross_attn_enable_img=True"
            )
            context_img = self.img_emb(img_embeddings)
        else:
            context_img = None

        block_caches: list[BlockCache] = []
        for block in self.blocks:
            assert isinstance(block, Block)
            block_caches.append(
                block.initialize_cache(
                    chunk_size, window_size, sink_size, context_text, context_img
                )
            )
        return WanDiTNetworkCache(block_caches=block_caches)

    def update_parameters_after_loading_checkpoint(self) -> None:
        """Fuse load-time-known ops into weights; call once after loading the checkpoint."""
        if self._parameters_updated_after_loading_checkpoint:
            return

        self._fuse_shuffle_op_into_last_layer()
        for block in self.blocks:
            assert isinstance(block, Block)
            block.update_parameters_after_loading_checkpoint()
        self.head.update_parameters_after_loading_checkpoint()

        self._parameters_updated_after_loading_checkpoint = True

    def _fuse_shuffle_op_into_last_layer(self) -> None:
        """Fuse the channel-shuffle that follows the last linear into its weights.

        In the WAN model the patchify pattern is
        ``b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)`` while the
        unpatchify pattern is
        ``b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)``. This mismatch
        (likely a bug in the official implementation) means the last dimension
        must be shuffled after the network. Folding that shuffle into
        ``head.head`` removes the explicit ``rearrange`` from the inference path.

        Calling this once is equivalent to running the following after the last
        layer::

            x = rearrange(
                x,
                "... (kt kh kw c) -> ... (c kt kh kw)",
                kt=self.patch_size[0],
                kh=self.patch_size[1],
                kw=self.patch_size[2],
                c=self.out_dim,
            )
        """
        if self._is_shuffle_op_fused:
            return

        self.head.head.weight.data = rearrange(
            self.head.head.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
            c=self.out_dim,
        ).contiguous()
        if self.head.head.bias is not None:
            self.head.head.bias.data = rearrange(
                self.head.head.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.patch_size[0],
                kh=self.patch_size[1],
                kw=self.patch_size[2],
                c=self.out_dim,
            ).contiguous()

        self._is_shuffle_op_fused = True

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        block_extra_kwargs: dict[str, Any] = {},
    ) -> Tensor:
        """Run one denoising forward pass.

        Args:
            x: Input tokens of shape [..., L, D_in] after patchify and CP.
                The layout is assumed to be "... (t h w) (c kt kh kw)".
            timesteps: Diffusion timesteps that is broadcast-able to shape [...].
            cache: Network KV caches.
            rope_freqs: RoPE frequencies of shape [L, 1, 1, head_dim // 2] after CP.
            current_chunk_idx: Current chunk index for streaming cache update.
            eager_mode: If True, run cache before/after update hooks.
            block_extra_kwargs: Extra kwargs to pass to the block.

        Returns:
            Tensor of shape [..., L, prod(patch_size) * out_dim].
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]

        # Patch embedding
        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)  # (..., L, D)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(
                self.dim, -1
            )  # [D, in_dim * kt * kh * kw]
            _bias = self.patch_embedding.bias  # [D] or None
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        # Timestep embedding and modulation projection
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )  # [..., D]
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))  # [..., 6, D]

        # Transformer blocks
        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x = block(
                x=x,
                e=torch.broadcast_to(e0, batch_shape + e0.shape[-2:]),
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                **block_extra_kwargs,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final head
        x = self.head(
            x, torch.broadcast_to(e, batch_shape + (1, e.shape[-1]))
        )  # (..., L, D)
        return x


# python -m flashdreams.recipes.wan.transformer.impl.network
if __name__ == "__main__":
    from flashdreams.core.checkpoint.load import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t2v_network_config = WanDiTNetwork1pt3BConfig()
    t2v_network = t2v_network_config.setup().to(device)
    t2v_state_dict = load_checkpoint(
        "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/diffusion_pytorch_model.safetensors"
    )
    t2v_network.load_state_dict(t2v_state_dict)
    print("Test T2V network loading done")

    i2v_network_config = WanDiTNetwork14BConfig(
        cross_attn_enable_img=True, in_dim=16 + 20
    )
    i2v_network = i2v_network_config.setup().to(device)
    i2v_state_dict = load_checkpoint(
        "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/blob/main/diffusion_pytorch_model.safetensors.index.json"
    )
    i2v_network.load_state_dict(i2v_state_dict)
    print("Test I2V network loading done")
