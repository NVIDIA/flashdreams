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

"""Cosmos DiT network: multi-view, HDMap-conditioned, AdaLN-LoRA."""

from dataclasses import dataclass, field

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

from .modules import (
    Block,
    BlockCache,
    FinalLayer,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
)


@dataclass
class CosmosDiTNetworkCache:
    block_caches: list[BlockCache]

    def __getitem__(self, index: int) -> BlockCache:
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


@dataclass
class CosmosDiTNetworkConfig(InstantiateConfig["CosmosDiTNetwork"]):
    _target: type["CosmosDiTNetwork"] = field(default_factory=lambda: CosmosDiTNetwork)

    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    concat_padding_mask: bool = True
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256
    use_crossattn_projection: bool = True
    crossattn_proj_in_channels: int = 100352
    crossattn_emb_channels: int = 1024
    timestep_scale: float = 0.001
    # hdmap conditioning
    additional_concat_ch: int = 0
    # multiview
    enable_cross_view_attn: bool = False
    view_condition_dim: int = 16
    n_cameras_emb: int = 7


class CosmosDiTNetwork(nn.Module):
    """
    DiT model for video generation with block-causal attention and KV-caching.

    Combines the Cosmos DiT architecture with causal attention masking for
    autoregressive video generation.
    """

    def __init__(self, config: CosmosDiTNetworkConfig):
        super().__init__()
        self.config = config

        # add 1 for the condition mask
        in_channels = config.in_channels + 1
        # optionally add 1 for the padding mask
        if self.config.concat_padding_mask:
            in_channels += 1

        # Build embeddings
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.config.patch_spatial,
            temporal_patch_size=self.config.patch_temporal,
            in_channels=in_channels,
            out_channels=self.config.model_channels,
        )
        # HDMap conditioning
        if self.config.additional_concat_ch > 0:
            self.additional_patch_embedding = PatchEmbed(
                spatial_patch_size=self.config.patch_spatial,
                temporal_patch_size=self.config.patch_temporal,
                in_channels=self.config.additional_concat_ch,
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
                    enable_cross_view_attn=self.config.enable_cross_view_attn,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=self.config.model_channels,
            spatial_patch_size=self.config.patch_spatial,
            temporal_patch_size=self.config.patch_temporal,
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

        if self.config.enable_cross_view_attn:
            self.adaln_view_embedder = nn.Embedding(
                self.config.n_cameras_emb, self.config.model_channels
            )
            self.adaln_view_proj = nn.Linear(
                self.config.model_channels, self.config.model_channels * 9
            )
        else:
            self.adaln_view_embedder = None
            self.adaln_view_proj = None

        self._is_shuffle_op_fused = False
        self._is_padding_mask_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def set_context_parallel_group(
        self,
        self_attn_group: ProcessGroup | None,
        cross_view_attn_group: ProcessGroup | None = None,
    ) -> None:
        for block in self.blocks:
            assert isinstance(block, Block)
            block.set_context_parallel_group(self_attn_group, cross_view_attn_group)

    def _fuse_shuffle_op_into_last_layer(self):
        """
        In the Cosmos model, the patchify operation is
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",

        while the unpatchify operation is
        "b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)"

        This is likely a bug in the Cosmos model where the last dimension is shuffled after the network.

        To fix this, we could fuse this shuffle op into the last linear layer,
        so that we do not have to do this shuffle op explicitly before returning the result.

        Calling this function to modify the last layer in place, is equivalent to the following code
        after the last layer:
        ```python
        x = rearrange(
            x,
            "... (kt kh kw c) -> ... (c kt kh kw)",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
            c=self.out_channels,
        )
        ```
        """
        if self._is_shuffle_op_fused:
            return

        self.final_layer.linear.weight.data = rearrange(
            self.final_layer.linear.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
            c=self.config.out_channels,
        ).contiguous()
        if self.final_layer.linear.bias is not None:
            self.final_layer.linear.bias.data = rearrange(
                self.final_layer.linear.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.config.patch_temporal,
                kh=self.config.patch_spatial,
                kw=self.config.patch_spatial,
                c=self.config.out_channels,
            ).contiguous()

        self._is_shuffle_op_fused = True
        return

    def _fuse_padding_mask_into_patch_embed(self) -> None:
        """
        Fuse the padding mask into the patch embedder in place.

        If `self.concat_padding_mask` is True, during training we are concatenating a
        padding_mask with shape [B, 1, T, H, W] to the input x_B_C_T_H_W on the C dimension,
        before passing it into the self.x_embedder. This is to work with data with different
        spatial resolutions during training, where `1` indicates padded regions. During
        inference, the padding_mask is always 0. So here we could simply remove the corresponding
        channels in the x_embedder in place

        Calling this function to modify the patch embedder in place, is equivalent to the following code
        before passing the input into the patch embedder:
        ```python
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)
        ```
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
        # This function should be called after loading the checkpoint, to fuse some operations in the model
        # weights to reduce computation during inference.
        if self._parameters_updated_after_loading_checkpoint:
            return

        self._fuse_padding_mask_into_patch_embed()
        self._fuse_shuffle_op_into_last_layer()
        self._parameters_updated_after_loading_checkpoint = True

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,  # [B, V, T, C, H, W]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        r"""
        Patchify the input tensor and maybe split it along cp_dim if a process group is provided.

        The patchify pattern is:
            "b v (t kt) c (h kh) (w kw) -> b v t (h w) (c kt kh kw)",

        Returns:
            Tensor: The patched tensor with shape [B, V, T, HW, D]
        """
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"

        x = rearrange(
            x,
            "... v (t kt) c (h kh) (w kw) -> ... v t (h w) (c kt kh kw)",
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
        )

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided"
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
        x: Tensor,  # [B, V, T, HW, D]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        r"""
        Unpatchify the input tensor and maybe gather it along cp_dim if a process group is provided.

        The unpatchify pattern is:
            "b v t (h w) (c kt kh kw) -> b v (t kt) c (h kh) (w kw)",

        Returns:
            Tensor: The unpatched tensor with shape [B, V, T, C, H, W]
        """
        assert x.ndim == 5, f"x must be a 5D tensor, but got shape {x.shape}"

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided"
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
            "b v t (h w) (c kt kh kw) -> b v (t kt) c (h kh) (w kw)",
            h=pH,
            w=pW,
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
        )
        return x  # [B, V, T, C, H, W]

    def initialize_cache(
        self,
        # self attn
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross attn
        context: Tensor,
    ) -> CosmosDiTNetworkCache:
        """
        Initialize the cache for the DiT.
        """
        if self.config.use_crossattn_projection:
            context = self.crossattn_proj(context)

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
        rope_freqs: Tensor,  # [L, 1, 1, D]
        cache: CosmosDiTNetworkCache,
        condition_video_input_mask: Tensor,
        current_chunk_idx: int = 0,
        hdmap_condition: Tensor | None = None,
        view_indices: Tensor | None = None,
        eager_mode: bool = True,
    ) -> Tensor:
        """
        Forward pass dispatching to training or inference mode.

        Args:
            x: Input video tensor [B, V, T, HW, D] after patchify
            timesteps: Timesteps [1] or [B]
            rope_freqs: RoPE cosine and sine embeddings [T*HW, 1, 1, D]
            cache: CosmosDiTCache
            condition_video_input_mask: Condition video input mask [B, V, T, HW, D] after patchify
            current_chunk_idx: Current chunk index in autoregressive inference
            hdmap_condition: HDMap tensor [B, V, T, HW, D]
            view_indices: View indices [B, V]
            eager_mode: Whether to run in eager mode (True) or not (False)
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )

        assert timesteps.ndim == 0, (
            f"timesteps must be a scalar tensor, got shape {tuple(timesteps.shape)}"
        )
        timesteps = timesteps * self.config.timestep_scale

        # Patch embedding
        x = torch.cat([x, condition_video_input_mask], dim=-1)
        x = self.x_embedder(x)

        if self.config.additional_concat_ch > 0:
            assert hdmap_condition is not None, (
                "hdmap is expected to be provided for additional concat channels"
            )
            additional_x = self.additional_patch_embedding(hdmap_condition)
            x = x + additional_x

        # Time embedding. ``timesteps`` is scalar; broadcast the resulting
        # embedding to the leading batch dim so downstream blocks/final
        # layer (which expect ``[B, D]`` and ``[B, 3D]``) work uniformly.
        t_emb, adaln_lora = self.t_embedder(timesteps)
        t_emb = self.t_embedding_norm(t_emb)
        B = x.shape[0]
        t_emb = t_emb.expand(B, -1)
        if adaln_lora is not None:
            adaln_lora = adaln_lora.expand(B, -1)

        # AdaLN view modulation if enabled
        if view_indices is not None:
            assert (
                self.adaln_view_embedder is not None
                and self.adaln_view_proj is not None
            ), (
                "adaln_view_embedder and adaln_view_proj must be provided if view_indices_B_V is provided"
            )
            view_emb = self.adaln_view_embedder(view_indices)  # [B, V, D]
            view_embedding_proj = self.adaln_view_proj(view_emb)  # [B, V, 9D]
        else:
            view_embedding_proj = None

        # Note: If not in eager mode, we should call `before_update` and `after_update` MANUALLY outside the network.
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
                view_embedding_proj=view_embedding_proj,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final layer
        x = self.final_layer(x, t_emb, adaln_lora)
        return x


# python -m flashdreams.recipes.alpadreams.transformer.impl.network --in-channels 3
if __name__ == "__main__":
    import tyro

    config = tyro.cli(CosmosDiTNetworkConfig)
    network = config.setup()
    print("network parameters:", sum(p.numel() for p in network.parameters()))
