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

"""FlashDreams diffusion model for MiniMax H3's paired latent streams."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
    num_audio_latents: int
    audio_channels: int


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3JointLatents:
    """Unpacked generated latents for H3's synchronized media streams."""

    video: Tensor
    """Video latents shaped ``[1, channels, time, height, width]``."""

    audio: Tensor
    """Stereo audio latents shaped ``[2, channels, time]``."""

    def __post_init__(self) -> None:
        expected = (
            ("video", self.video, 5, 1),
            ("audio", self.audio, 3, 2),
        )
        for name, value, rank, batch in expected:
            if value.ndim != rank or value.shape[0] != batch:
                raise ValueError(
                    f"{name} latents must have rank {rank} and leading dimension "
                    f"{batch}."
                )
            if any(size <= 0 for size in value.shape):
                raise ValueError(f"{name} latents must have non-empty dimensions.")
            if not value.is_floating_point():
                raise ValueError(f"{name} latents must use a floating-point dtype.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} latents must contain only finite values.")


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3DenoiseProgress:
    """One resumable, synchronized point in H3's paired schedules."""

    video: Tensor
    """Packed video conditioning and targets at ``next_step``."""

    audio: Tensor
    """Packed audio conditioning and targets at ``next_step``."""

    next_step: int
    """Zero-based schedule index to execute next."""

    def __post_init__(self) -> None:
        if type(self.next_step) is not int or self.next_step < 0:
            raise ValueError("next_step must be a non-negative integer.")
        for name, value in (("video", self.video), ("audio", self.audio)):
            if value.ndim != 2 or any(size <= 0 for size in value.shape):
                raise ValueError(f"Packed {name} progress must be a non-empty matrix.")
            if not value.is_floating_point():
                raise ValueError(f"Packed {name} progress must be floating point.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Packed {name} progress must contain finite values.")


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
    def _validate_state(state: MiniMaxH3DenoiseState, transformer_config: Any) -> None:
        """Reject malformed packed streams before allocating execution memory."""
        counts = {
            "num_condition_video_rows": state.num_condition_video_rows,
            "num_condition_audio_rows": state.num_condition_audio_rows,
        }
        dimensions = {
            "num_latent_frames": state.num_latent_frames,
            "latent_height": state.latent_height,
            "latent_width": state.latent_width,
            "num_audio_latents": state.num_audio_latents,
        }
        for name, value in counts.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(state.audio_channels) is not int or state.audio_channels != 2:
            raise ValueError("MiniMax H3 generation requires exactly 2 audio channels.")

        patch_size = tuple(transformer_config.patch_size)
        if len(patch_size) != 3 or any(
            type(size) is not int or size <= 0 for size in patch_size
        ):
            raise ValueError("transformer patch_size must contain 3 positive integers.")
        patch_t, patch_h, patch_w = patch_size
        if (
            state.num_latent_frames % patch_t
            or state.latent_height % patch_h
            or state.latent_width % patch_w
        ):
            raise ValueError(
                "Generated video latent dimensions must align to patch_size."
            )

        expected_video_targets = (
            state.num_latent_frames
            // patch_t
            * (state.latent_height // patch_h)
            * (state.latent_width // patch_w)
        )
        expected_video_rows = state.num_condition_video_rows + expected_video_targets
        expected_video_width = transformer_config.in_channels * (
            patch_t * patch_h * patch_w
        )
        if tuple(state.latents.shape) != (
            expected_video_rows,
            expected_video_width,
        ):
            raise ValueError(
                "Packed video latents do not match their conditioning and geometry."
            )

        expected_audio_targets = state.audio_channels * state.num_audio_latents
        expected_audio_rows = state.num_condition_audio_rows + expected_audio_targets
        if tuple(state.audio_latents.shape) != (
            expected_audio_rows,
            transformer_config.audio_in_channels,
        ):
            raise ValueError(
                "Packed audio latents do not match their conditioning and geometry."
            )
        if state.num_condition_audio_rows % state.audio_channels:
            raise ValueError(
                "Condition audio rows must contain complete stereo samples."
            )

        floating_inputs = {
            "latents": state.latents,
            "audio_latents": state.audio_latents,
            "prompt_embeds": state.prompt_embeds,
        }
        for name, value in floating_inputs.items():
            if not value.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values.")

        indices = {
            "video_indices": state.video_indices,
            "audio_indices": state.audio_indices,
            "text_indices": state.text_indices,
        }
        for name, value in indices.items():
            if value.ndim != 1 or value.dtype != torch.long:
                raise ValueError(f"{name} must be a one-dimensional torch.long tensor.")
        if state.video_indices.numel() != expected_video_rows:
            raise ValueError("video_indices must identify every packed video row.")
        if state.audio_indices.numel() != expected_audio_rows:
            raise ValueError("audio_indices must identify every packed audio row.")

        text_rows = state.text_indices.numel()
        if state.prompt_embeds.ndim != 3 or state.prompt_embeds.shape[:2] != (
            1,
            text_rows,
        ):
            raise ValueError("prompt_embeds must have shape [1, text_rows, hidden].")
        if state.prompt_embeds.shape[2] != transformer_config.text_dim:
            raise ValueError(
                "prompt_embeds hidden width must match transformer text_dim."
            )

        sequence_length = expected_video_rows + expected_audio_rows + text_rows
        if tuple(state.position_ids.shape) != (sequence_length, 3):
            raise ValueError("position_ids must have shape [sequence_length, 3].")
        if tuple(state.token_tags.shape) != (sequence_length,):
            raise ValueError("token_tags must have shape [sequence_length].")
        if state.token_tags.dtype != torch.long:
            raise ValueError("token_tags must use torch.long dtype.")

        packed_indices = torch.cat(
            tuple(value.detach().cpu() for value in indices.values())
        )
        expected_indices = torch.arange(sequence_length, dtype=torch.long)
        if not torch.equal(packed_indices.sort().values, expected_indices):
            raise ValueError(
                "Modality indices must partition the packed sequence exactly once."
            )

    @staticmethod
    def _validate_resume(
        state: MiniMaxH3DenoiseState,
        resume: MiniMaxH3DenoiseProgress,
    ) -> None:
        """Require resumable targets to retain this request's condition rows."""
        expected = {
            "video": state.latents,
            "audio": state.audio_latents,
        }
        actual = {
            "video": resume.video,
            "audio": resume.audio,
        }
        condition_rows = {
            "video": state.num_condition_video_rows,
            "audio": state.num_condition_audio_rows,
        }
        for name in expected:
            source = expected[name]
            restored = actual[name]
            if restored.shape != source.shape or restored.dtype != source.dtype:
                raise ValueError(
                    f"Resumed {name} latents do not match the packed request shape "
                    "and dtype."
                )
            rows = condition_rows[name]
            if not torch.equal(
                restored[:rows].detach().cpu(), source[:rows].detach().cpu()
            ):
                raise ValueError(
                    f"Resumed {name} conditioning does not match this request."
                )

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
        video_timestep = video_timestep.to(
            device=state.video_indices.device, dtype=torch.float32
        )
        audio_timestep = audio_timestep.to(
            device=state.video_indices.device, dtype=torch.float32
        )
        row_timesteps = video_timestep.expand(sequence_length).clone()
        video_condition = state.video_indices[: state.num_condition_video_rows]
        audio_condition = state.audio_indices[: state.num_condition_audio_rows]
        audio_target = state.audio_indices[state.num_condition_audio_rows :]
        row_timesteps[video_condition] = video_timestep.clamp_min(0.999)
        row_timesteps[audio_target] = audio_timestep
        row_timesteps[audio_condition] = 1.0
        return torch.unique(row_timesteps, sorted=True, return_inverse=True)

    @torch.no_grad()
    def generate_joint(
        self,
        state: MiniMaxH3DenoiseState,
        *,
        resume: MiniMaxH3DenoiseProgress | None = None,
        checkpoint: Callable[[MiniMaxH3DenoiseProgress], None] | None = None,
    ) -> MiniMaxH3JointLatents:
        """Denoise, optionally checkpoint, and unpack both synchronized streams."""
        transformer_config = cast(Any, self.config.transformer)
        self._validate_state(state, transformer_config)
        if resume is not None:
            self._validate_resume(state, resume)
        device = self.transformer.device
        next_step = 0 if resume is None else resume.next_step
        video_sigmas, video_timesteps = self.scheduler.schedule(device)
        audio_sigmas, audio_timesteps = self.audio_scheduler.schedule(device)
        if len(video_timesteps) != len(audio_timesteps):
            raise RuntimeError("H3 video and audio schedules must have equal length")
        if next_step > len(video_timesteps):
            raise ValueError(
                "Resumed next_step exceeds the configured denoising schedule."
            )

        video_source = state.latents if resume is None else resume.video
        audio_source = state.audio_latents if resume is None else resume.audio
        video = video_source.to(device).clone()
        audio = audio_source.to(device).clone()
        execution_state = replace(
            state,
            prompt_embeds=state.prompt_embeds.to(device),
            position_ids=state.position_ids.to(device),
            token_tags=state.token_tags.to(device),
            video_indices=state.video_indices.to(device),
            audio_indices=state.audio_indices.to(device),
            text_indices=state.text_indices.to(device),
        )

        cache = MiniMaxH3TransformerCache(
            audio_hidden_states=audio[None],
            encoder_hidden_states=execution_state.prompt_embeds,
            timestep=torch.empty(0, device=device),
            timestep_indices=torch.empty(0, dtype=torch.long, device=device),
            token_tags=execution_state.token_tags,
            position_ids=execution_state.position_ids,
            video_indices=execution_state.video_indices,
            audio_indices=execution_state.audio_indices,
            text_indices=execution_state.text_indices,
        )
        for index in range(next_step, len(video_timesteps)):
            video_timestep = video_timesteps[index]
            audio_timestep = audio_timesteps[index]
            logger.info(
                "MiniMax H3 denoise step {}/{}",
                index + 1,
                len(video_timesteps),
            )
            cache.timestep, cache.timestep_indices = self._row_timesteps(
                execution_state, video_timestep, audio_timestep
            )
            cache.audio_hidden_states = audio[None]
            cache.last_audio_flow = None
            video_flow = self.transformer.predict_flow(
                video[None], video_timestep, cache
            )[0]
            if tuple(video_flow.shape) != tuple(video.shape):
                raise RuntimeError(
                    "H3 transformer returned an invalid video flow shape"
                )
            if cache.last_audio_flow is None:
                raise RuntimeError("H3 transformer did not produce an audio flow")
            audio_flow = cache.last_audio_flow[0]
            if tuple(audio_flow.shape) != tuple(audio.shape):
                raise RuntimeError(
                    "H3 transformer returned an invalid audio flow shape"
                )

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
            if checkpoint is not None:
                checkpoint(
                    MiniMaxH3DenoiseProgress(
                        video=video.detach().clone(),
                        audio=audio.detach().clone(),
                        next_step=index + 1,
                    )
                )

        rows = video[state.num_condition_video_rows :]
        patch_t, patch_h, patch_w = transformer_config.patch_size
        channels = transformer_config.in_channels
        rows = rows.reshape(
            1,
            state.num_latent_frames // patch_t,
            state.latent_height // patch_h,
            state.latent_width // patch_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        video_latents = (
            rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(
                1,
                channels,
                state.num_latent_frames,
                state.latent_height,
                state.latent_width,
            )
            .contiguous()
        )
        audio_rows = audio[state.num_condition_audio_rows :]
        audio_latents = (
            audio_rows.reshape(
                state.audio_channels,
                state.num_audio_latents,
                transformer_config.audio_in_channels,
            )
            .permute(0, 2, 1)
            .contiguous()
        )
        return MiniMaxH3JointLatents(video=video_latents, audio=audio_latents)


__all__ = [
    "MiniMaxH3DenoiseProgress",
    "MiniMaxH3DenoiseState",
    "MiniMaxH3DiffusionModel",
    "MiniMaxH3DiffusionModelConfig",
    "MiniMaxH3JointLatents",
]
