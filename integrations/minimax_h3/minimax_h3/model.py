# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams diffusion model for MiniMax H3's paired latent streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from loguru import logger
from torch import Tensor, nn

from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig
from flashdreams.infra.diffusion.transformer import Transformer, TransformerConfig
from minimax_h3.scheduler import MiniMaxH3Scheduler, MiniMaxH3SchedulerConfig
from minimax_h3.transformer import (
    MiniMaxH3TransformerCache,
    MiniMaxH3TransformerConfig,
)


@dataclass(kw_only=True)
class MiniMaxH3DiffusionModelConfig(DiffusionModelConfig):
    """Native H3 transformer plus separate video and audio schedules."""

    _target: type[MiniMaxH3DiffusionModel] = field(
        default_factory=lambda: MiniMaxH3DiffusionModel
    )
    transformer: TransformerConfig = field(
        default_factory=lambda: MiniMaxH3TransformerConfig(
            device="cuda",
            execution_device="cuda",
            sequential_cpu_offload=False,
        )
    )
    scheduler: MiniMaxH3SchedulerConfig = field(
        default_factory=MiniMaxH3SchedulerConfig
    )
    audio_scheduler: MiniMaxH3SchedulerConfig = field(
        default_factory=lambda: MiniMaxH3SchedulerConfig(shift=3.0)
    )


@dataclass(kw_only=True)
class MiniMaxH3DenoiseState:
    """Packed conditioning, noise, and layout produced before denoising."""

    latents: Tensor
    audio_latents: Tensor
    prompt_embeds: Tensor
    position_ids: Tensor
    token_tags: Tensor
    video_indices: Tensor
    audio_indices: Tensor
    text_indices: Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_latent_frames: int
    latent_height: int
    latent_width: int


class MiniMaxH3DiffusionModel(DiffusionModel[MiniMaxH3TransformerCache]):
    """Run H3's joint forward under two FlashDreams-owned schedulers."""

    config: MiniMaxH3DiffusionModelConfig
    transformer: Transformer[MiniMaxH3TransformerCache]
    scheduler: MiniMaxH3Scheduler
    audio_scheduler: MiniMaxH3Scheduler

    def __init__(self, config: MiniMaxH3DiffusionModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.transformer = config.transformer.setup()
        self.scheduler = config.scheduler.setup()
        self.audio_scheduler = config.audio_scheduler.setup()

    @staticmethod
    def _row_timesteps(
        state: MiniMaxH3DenoiseState,
        video_timestep: Tensor,
        audio_timestep: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sequence_length = (
            state.video_indices.numel()
            + state.audio_indices.numel()
            + state.text_indices.numel()
        )
        row_timesteps = torch.full(
            (sequence_length,),
            float(video_timestep),
            dtype=torch.float32,
            device=state.video_indices.device,
        )
        video_condition = state.video_indices[: state.num_condition_video_rows]
        audio_condition = state.audio_indices[: state.num_condition_audio_rows]
        audio_target = state.audio_indices[state.num_condition_audio_rows :]
        row_timesteps[video_condition] = max(float(video_timestep), 0.999)
        row_timesteps[audio_target] = audio_timestep
        row_timesteps[audio_condition] = 1.0
        return torch.unique(row_timesteps, sorted=True, return_inverse=True)

    @torch.no_grad()
    def generate_joint(self, state: MiniMaxH3DenoiseState) -> Tensor:
        """Denoise both streams and return only unpacked video latents."""
        device = self.transformer.device
        video = state.latents.to(device)
        audio = state.audio_latents.to(device)
        state.prompt_embeds = state.prompt_embeds.to(device)
        state.position_ids = state.position_ids.to(device)
        state.token_tags = state.token_tags.to(device)
        state.video_indices = state.video_indices.to(device)
        state.audio_indices = state.audio_indices.to(device)
        state.text_indices = state.text_indices.to(device)

        video_sigmas, video_timesteps = self.scheduler.schedule(device)
        audio_sigmas, audio_timesteps = self.audio_scheduler.schedule(device)
        if len(video_timesteps) != len(audio_timesteps):
            raise RuntimeError("H3 video and audio schedules must have equal length")

        cache = MiniMaxH3TransformerCache(
            audio_hidden_states=audio[None],
            encoder_hidden_states=state.prompt_embeds,
            timestep=torch.empty(0, device=device),
            timestep_indices=torch.empty(0, dtype=torch.long, device=device),
            token_tags=state.token_tags,
            position_ids=state.position_ids,
            video_indices=state.video_indices,
            audio_indices=state.audio_indices,
            text_indices=state.text_indices,
        )
        for index, (video_timestep, audio_timestep) in enumerate(
            zip(video_timesteps, audio_timesteps, strict=True)
        ):
            logger.info(
                "MiniMax H3 denoise step {}/{}",
                index + 1,
                len(video_timesteps),
            )
            cache.timestep, cache.timestep_indices = self._row_timesteps(
                state, video_timestep, audio_timestep
            )
            cache.audio_hidden_states = audio[None]
            video_flow = self.transformer.predict_flow(
                video[None], video_timestep, cache
            )[0]
            if cache.last_audio_flow is None:
                raise RuntimeError("H3 transformer did not produce an audio flow")
            audio_flow = cache.last_audio_flow[0]

            video_start = state.num_condition_video_rows
            audio_start = state.num_condition_audio_rows
            video[video_start:] = self.scheduler.step(
                video[video_start:],
                video_flow[video_start:].float(),
                video_timestep,
                video_sigmas[index],
                video_sigmas[index + 1],
            )
            audio[audio_start:] = self.audio_scheduler.step(
                audio[audio_start:],
                audio_flow[audio_start:].float(),
                audio_timestep,
                audio_sigmas[index],
                audio_sigmas[index + 1],
            )

        rows = video[state.num_condition_video_rows :]
        transformer_config = cast(Any, self.config.transformer)
        patch_t, patch_h, patch_w = transformer_config.patch_size
        channels = transformer_config.in_channels
        rows = rows.reshape(
            -1,
            state.num_latent_frames // patch_t,
            state.latent_height // patch_h,
            state.latent_width // patch_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return (
            rows.reshape(
                -1,
                channels,
                state.num_latent_frames,
                state.latent_height,
                state.latent_width,
            )
            .contiguous()
            .cpu()
        )


__all__ = [
    "MiniMaxH3DenoiseState",
    "MiniMaxH3DiffusionModel",
    "MiniMaxH3DiffusionModelConfig",
]
