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
"""Native LingBot-VA transformer with torch.compile support.

Wraps ``WanVADiTNetwork`` in the flashdreams ``Transformer`` interface
with ``predict_flow`` (video) and ``predict_action_flow`` (action).
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)
from lingbot_va.transformer.checkpoint import state_dict_transform
from lingbot_va.transformer.impl.network import (
    VideoKV,
    WanVADiTNetwork,
    WanVADiTNetworkCache,
    WanVADiTNetworkConfig,
    compute_rope_freqs_from_grid,
)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LingbotVATransformerCache(TransformerAutoregressiveCache):
    """Per-rollout AR cache."""

    network_cache: WanVADiTNetworkCache
    network_cache_uncond: WanVADiTNetworkCache | None = None
    video_kv_cond: VideoKV | None = None
    """Video KV produced by the conditional branch for the current chunk."""
    video_kv_uncond: VideoKV | None = None
    """Video KV produced by the unconditional branch for the current chunk."""
    autoregressive_index: int = -1

    def start(self, autoregressive_index: int) -> None:
        """Open cache window for the given AR step."""
        self.autoregressive_index = autoregressive_index
        for bc in self.network_cache.block_caches:
            bc.self_attn.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            for bc in self.network_cache_uncond.block_caches:
                bc.self_attn.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        """Close cache window and commit the AR step."""
        for bc in self.network_cache.block_caches:
            bc.self_attn.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            for bc in self.network_cache_uncond.block_caches:
                bc.self_attn.after_update(autoregressive_index)
        self.autoregressive_index = autoregressive_index


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LingbotVATransformerConfig(TransformerConfig):
    """Config for the native LingBot-VA transformer."""

    _target: type["LingbotVATransformer"] = field(
        default_factory=lambda: LingbotVATransformer
    )

    network: WanVADiTNetworkConfig = field(default_factory=WanVADiTNetworkConfig)
    checkpoint_root: str = ""
    dtype: torch.dtype = torch.bfloat16
    compile_network: bool = True
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0

    # Spatial layout
    latent_height: int = 0
    latent_width: int = 0
    frame_chunk_size: int = 4
    action_per_frame: int = 16
    attn_window: int = 72


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class LingbotVATransformer(Transformer[LingbotVATransformerCache]):
    """Native LingBot-VA transformer with torch.compile support."""

    _network: WanVADiTNetwork | None

    def __init__(self, config: LingbotVATransformerConfig) -> None:
        super().__init__(config)
        self.config: LingbotVATransformerConfig = config
        self._anchor = nn.Parameter(torch.empty(0))
        # Use object.__setattr__ to avoid nn.Module type checks on compiled modules
        object.__setattr__(self, "_network", None)

    def load_model(self, device: torch.device) -> None:
        """Build, load weights, and optionally compile the network."""
        cfg = self.config
        net = WanVADiTNetwork(cfg.network)
        net.eval()

        ckpt_path = os.path.join(cfg.checkpoint_root, "transformer")
        idx_path = os.path.join(
            ckpt_path, "diffusion_pytorch_model.safetensors.index.json"
        )
        if os.path.exists(idx_path):
            ckpt_path = idx_path
        state_dict = load_checkpoint(ckpt_path, map_location="cpu")
        state_dict = state_dict_transform(state_dict)
        net.load_state_dict(state_dict)
        del state_dict
        gc.collect()

        net.update_parameters_after_loading_checkpoint()
        net = net.to(dtype=cfg.dtype, device=device)

        if cfg.compile_network:
            object.__setattr__(
                net,
                "_forward_blocks_video",
                torch.compile(
                    net._forward_blocks_video,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=True,
                ),
            )
            object.__setattr__(
                net,
                "_forward_blocks_action",
                torch.compile(
                    net._forward_blocks_action,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=True,
                ),
            )

        object.__setattr__(self, "_network", net)

    @property
    def network(self) -> WanVADiTNetwork:
        assert self._network is not None, "Call load_model() first"
        return self._network

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        text_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
        batch_size: int = 1,
        **_unused: Any,
    ) -> LingbotVATransformerCache:
        cfg = self.config
        ps = cfg.network.patch_size
        video_chunk = (
            (cfg.frame_chunk_size // ps[0])
            * (cfg.latent_height // ps[1])
            * (cfg.latent_width // ps[2])
        )
        action_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        window_slots = cfg.attn_window // 2

        network_cache = self.network.initialize_cache(
            text_embeddings=text_embeddings,
            video_chunk=video_chunk,
            action_chunk=action_chunk,
            window_slots=window_slots,
            batch_size=batch_size,
        )

        network_cache_uncond: WanVADiTNetworkCache | None = None
        if cfg.guidance_scale > 1.0 or cfg.action_guidance_scale > 1.0:
            assert negative_text_embeddings is not None
            network_cache_uncond = self.network.initialize_cache(
                text_embeddings=negative_text_embeddings,
                video_chunk=video_chunk,
                action_chunk=action_chunk,
                window_slots=window_slots,
                batch_size=batch_size,
            )

        return LingbotVATransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
        )

    # ------------------------------------------------------------------
    # Flow prediction
    # ------------------------------------------------------------------

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: LingbotVATransformerCache,
        input: Any = None,
        persist: bool = False,
    ) -> Tensor:
        """Video-mode flow prediction with optional CFG."""
        grid_id = input["grid_id"]
        if grid_id.shape[0] == 4:
            grid_id = grid_id[:3]
        rope_freqs = compute_rope_freqs_from_grid(
            grid_id, self.config.network.dim // self.config.network.num_heads
        ).to(noisy_latent.device)

        flow_cond, video_kv_cond = self.network.forward_video(
            noisy_latent,
            timestep,
            cache.network_cache,
            rope_freqs,
            persist=persist,
        )
        if persist:
            assert video_kv_cond is not None
            cache.video_kv_cond = video_kv_cond

        if cache.network_cache_uncond is not None and (
            persist or self.config.guidance_scale > 1.0
        ):
            flow_uncond, video_kv_uncond = self.network.forward_video(
                noisy_latent,
                timestep,
                cache.network_cache_uncond,
                rope_freqs,
                persist=persist,
            )
            if persist:
                assert video_kv_uncond is not None
                cache.video_kv_uncond = video_kv_uncond
            if self.config.guidance_scale > 1.0:
                return flow_uncond + self.config.guidance_scale * (
                    flow_cond - flow_uncond
                )

        return flow_cond

    def predict_action_flow(
        self,
        noisy_action: Tensor,
        timestep: Tensor,
        cache: LingbotVATransformerCache,
        input: Any = None,
        persist: bool = False,
    ) -> Tensor:
        """Action-mode flow prediction with optional CFG."""
        grid_id = input["grid_id"]
        if grid_id.shape[0] == 4:
            grid_id = grid_id[:3]
        rope_freqs = compute_rope_freqs_from_grid(
            grid_id, self.config.network.dim // self.config.network.num_heads
        ).to(noisy_action.device)

        flow_cond = self.network.forward_action(
            noisy_action,
            timestep,
            cache.network_cache,
            rope_freqs,
            video_kv=cache.video_kv_cond,
            persist=persist,
        )
        if persist:
            cache.video_kv_cond = None

        if cache.network_cache_uncond is not None and (
            persist or self.config.action_guidance_scale > 1.0
        ):
            flow_uncond = self.network.forward_action(
                noisy_action,
                timestep,
                cache.network_cache_uncond,
                rope_freqs,
                video_kv=cache.video_kv_uncond,
                persist=persist,
            )
            if persist:
                cache.video_kv_uncond = None
            if self.config.action_guidance_scale > 1.0:
                return flow_uncond + self.config.action_guidance_scale * (
                    flow_cond - flow_uncond
                )

        return flow_cond

    # ------------------------------------------------------------------
    # Abstract method stubs
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return (0,)

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        return x

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        return x
