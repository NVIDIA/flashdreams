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
"""LingBot-VA inference pipeline with dual-denoise generate() override."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import torch
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from flashdreams.infra.pipeline import StreamInferencePipeline, StreamInferencePipelineConfig
from flashdreams.infra.pipeline.base import StreamInferencePipelineCache

from lingbot_va.scheduler import LingbotVAFlowMatchScheduler, LingbotVAFlowMatchSchedulerConfig
from lingbot_va.transformer import LingbotVATransformer, LingbotVATransformerCache
from lingbot_va.utils import get_mesh_id


class LingbotVAOutput(NamedTuple):
    """Output from one AR step of the LingBot-VA pipeline."""
    latent: Tensor
    action: Tensor


@dataclass(kw_only=True)
class LingbotVAInferencePipelineConfig(StreamInferencePipelineConfig):
    """Pipeline config for LingBot-VA Robotwin I2AV.

    Holds the video scheduler via ``diffusion_model.scheduler`` and the
    action scheduler separately.
    """

    _target: type["LingbotVAInferencePipeline"] = field(
        default_factory=lambda: LingbotVAInferencePipeline
    )

    checkpoint_root: str
    robotwin_height: int = 256
    robotwin_width: int = 320
    frame_chunk_size: int = 2
    action_dim: int = 30
    action_per_frame: int = 16
    attn_window: int = 64
    latent_height: int = 24
    latent_width: int = 20
    latent_channels: int = 48
    latent_token_per_chunk: int = 240
    action_token_per_chunk: int = 32

    action_scheduler: LingbotVAFlowMatchSchedulerConfig = field(
        default_factory=lambda: LingbotVAFlowMatchSchedulerConfig(
            num_inference_steps=10,
            shift=3.0,
        )
    )


class LingbotVAInferencePipeline(StreamInferencePipeline):
    """LingBot-VA pipeline with dual video+action denoising.

    Overrides ``generate()`` to run the video denoise loop followed by the
    action denoise loop, both writing to the shared KV cache. The
    ``DiffusionModel`` built by the parent provides the transformer and
    video scheduler; the action scheduler is held separately.
    """

    config: LingbotVAInferencePipelineConfig

    def __init__(self, config: LingbotVAInferencePipelineConfig) -> None:
        super().__init__(config)
        self.config = config
        self.action_scheduler = config.action_scheduler.setup()

    @property
    def transformer(self) -> LingbotVATransformer:
        return cast(LingbotVATransformer, self.diffusion_model.transformer)

    @property
    def video_scheduler(self) -> LingbotVAFlowMatchScheduler:
        return cast(LingbotVAFlowMatchScheduler, self.diffusion_model.scheduler)

    def initialize_cache(  # type: ignore[override]
        self,
        *,
        text_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
        batch_size: int = 1,
    ) -> StreamInferencePipelineCache:
        """Build the per-rollout cache."""
        transformer_cache = self.transformer.initialize_autoregressive_cache(
            text_embeddings=text_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            batch_size=batch_size,
        )
        return StreamInferencePipelineCache(
            transformer_cache=transformer_cache,
            encoder_cache=None,
            decoder_cache=None,
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache,
        input: dict[str, Any] | None = None,
    ) -> LingbotVAOutput:
        """Run video + action denoising for one AR chunk.

        Args:
            autoregressive_index: Chunk index (0-based).
            cache: Per-rollout cache from ``initialize_cache``.
            input: Dict with keys:
                - ``init_latent``: Encoded observation ``[B, C, 1, H, W]``.
                - ``action_mask``: Bool mask ``[action_dim]``.
                - ``device``: Target device.
                - ``dtype``: Target dtype.

        Returns:
            ``LingbotVAOutput(latent, action)`` for this chunk.
        """
        assert input is not None
        cfg = self.config
        transformer_cache = cast(
            LingbotVATransformerCache,
            cache.transformer_cache,
        )

        init_latent = input["init_latent"]
        action_mask = input["action_mask"]
        device = input["device"]
        dtype = input["dtype"]

        fcs = cfg.frame_chunk_size
        ps = self.transformer.config.network.patch_size
        lh = cfg.latent_height
        lw = cfg.latent_width
        frame_st_id = autoregressive_index * fcs

        # Initial noise
        latents = torch.randn(1, cfg.latent_channels, fcs, lh, lw, device=device, dtype=dtype)
        actions = torch.randn(1, cfg.action_dim, fcs, cfg.action_per_frame, 1, device=device, dtype=dtype)

        # Timesteps
        video_timesteps = self.video_scheduler.padded_timesteps
        action_timesteps = self.action_scheduler.padded_timesteps

        # RoPE grid IDs
        video_grid_id = get_mesh_id(
            fcs // ps[0], lh // ps[1], lw // ps[2], 0, 1, frame_st_id,
        ).to(device)
        action_grid_id = get_mesh_id(
            fcs, cfg.action_per_frame, 1, 1, 1, frame_st_id, action=True,
        ).to(device)

        # Open cache window
        transformer_cache.start(autoregressive_index)

        # --- Video denoise ---
        for i, t in enumerate(tqdm(video_timesteps, desc="video denoise")):
            last_step = i == len(video_timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(dtype) if frame_st_id == 0 else None

            noisy = latents.clone()
            if latent_cond is not None:
                noisy[:, :, 0:1] = latent_cond
            x = rearrange(
                noisy,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=ps[0], p2=ps[1], p3=ps[2],
            )

            t_val = float(t)
            n_tokens = x.shape[1]
            ts = torch.full((1, n_tokens), t_val, dtype=torch.float32, device=device)
            if latent_cond is not None and frame_st_id == 0:
                cond_tokens = (lh // ps[1]) * (lw // ps[2])
                ts[:, :cond_tokens] = 0.0

            pred = self.transformer.predict_flow(
                x, ts, transformer_cache, input={"grid_id": video_grid_id},
                persist=last_step,
            )

            if not last_step:
                pred = rearrange(
                    pred,
                    'b (f h w) (c kt kh kw) -> b c (f kt) (h kh) (w kw)',
                    f=fcs // ps[0], h=lh // ps[1], w=lw // ps[2],
                    kt=ps[0], kh=ps[1], kw=ps[2],
                )
                latents = self.video_scheduler.step(pred, t, latents)

            if latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

        # --- Action denoise ---
        for i, t in enumerate(tqdm(action_timesteps, desc="action denoise")):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros(1, cfg.action_dim, 1, cfg.action_per_frame, 1, device=device, dtype=dtype)
                if frame_st_id == 0 else None
            )

            noisy_a = actions.clone()
            if action_cond is not None:
                noisy_a[:, :, 0:1] = action_cond
            noisy_a[:, ~action_mask] *= 0
            x_a = rearrange(noisy_a, 'b c f h w -> b (f h w) c')

            t_val = float(t)
            n_tokens_a = x_a.shape[1]
            ts_a = torch.full((1, n_tokens_a), t_val, dtype=torch.float32, device=device)
            if action_cond is not None and frame_st_id == 0:
                ts_a[:, :cfg.action_per_frame] = 0.0

            pred = self.transformer.predict_action_flow(
                x_a, ts_a, transformer_cache, input={"grid_id": action_grid_id},
                persist=last_step,
            )

            if not last_step:
                pred = rearrange(pred, "b (f n) c -> b c f n 1", f=fcs)
                actions = self.action_scheduler.step(pred, t, actions)

            if action_cond is not None:
                actions[:, :, 0:1] = action_cond

        # Close cache window and commit
        transformer_cache.finalize(autoregressive_index)

        actions[:, ~action_mask] *= 0
        return LingbotVAOutput(latent=latents, action=actions)

    def finalize(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: StreamInferencePipelineCache,
    ) -> None:
        """No-op — cache is finalized within generate()."""
        pass
