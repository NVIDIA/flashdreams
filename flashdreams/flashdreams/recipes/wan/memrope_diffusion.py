# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MemRoPE-specific diffusion rollout alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Generic

import torch
from torch import Tensor

from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchScheduler
from flashdreams.infra.diffusion.transformer import TransformerCacheT


@dataclass(kw_only=True)
class MemRoPEDiffusionModelConfig(DiffusionModelConfig):
    """Diffusion config matching the official MemRoPE rollout layout."""

    _target: type["MemRoPEDiffusionModel"] = field(
        default_factory=lambda: MemRoPEDiffusionModel
    )


class MemRoPEDiffusionModel(
    DiffusionModel[TransformerCacheT],
    Generic[TransformerCacheT],
):
    """Diffusion model with official MemRoPE noise ordering.

    The reference MemRoPE code samples the whole rollout noise on a CPU
    generator in BTCHW latent layout, then re-noises every intermediate x0
    estimate in that same unpatchified layout before the next DiT forward.
    FlashDreams' generic diffusion model samples directly in patchified token
    layout on the model device. Both are distributionally equivalent, but the
    exact RNG stream and first-frame realization differ. This subclass keeps
    that behavior contained to MemRoPE configs.
    """

    config: MemRoPEDiffusionModelConfig

    def __init__(self, config: MemRoPEDiffusionModelConfig) -> None:
        super().__init__(config)
        self._cpu_rng: torch.Generator | None = None

    @property
    def cpu_rng(self) -> torch.Generator | None:
        if self.config.seed is None:
            return None
        if self._cpu_rng is None:
            self._cpu_rng = torch.Generator(device="cpu").manual_seed(
                self.config.seed
            )
        return self._cpu_rng

    def _unpatchified_chunk_shape(self) -> tuple[int, ...]:
        cfg = self.transformer.config
        patch_size = cfg.network.patch_size
        channels = self.latent_shape[-1] // math.prod(patch_size)
        assert channels * math.prod(patch_size) == self.latent_shape[-1], (
            "latent feature dimension must be divisible by patch volume"
        )
        return (*self.latent_shape[:-2], cfg.len_t, channels, cfg.height, cfg.width)

    def _draw_unpatchified_chunk_noise(self) -> Tensor:
        noise = torch.randn(
            self._unpatchified_chunk_shape(),
            generator=self.cpu_rng,
            device="cpu",
            dtype=self.dtype,
        )
        return noise.to(self.device)

    def _draw_unpatchified_renoise(self, clean_unpatchified: Tensor) -> Tensor:
        batch_shape = clean_unpatchified.shape[:-4]
        len_t, channels, height, width = clean_unpatchified.shape[-4:]
        flat_shape = (math.prod(batch_shape) * len_t, channels, height, width)
        noise = torch.randn(
            flat_shape,
            generator=self.cpu_rng,
            device="cpu",
            dtype=clean_unpatchified.dtype,
        ).to(clean_unpatchified.device)
        return noise.reshape(*batch_shape, len_t, channels, height, width)

    def _official_timestep_tensor(self, timestep: Tensor, noisy_latent: Tensor) -> Tensor:
        len_t = self._unpatchified_chunk_shape()[-4]
        return torch.ones(
            (*noisy_latent.shape[:-2], len_t),
            device=noisy_latent.device,
            dtype=timestep.dtype,
        ) * timestep.to(device=noisy_latent.device)

    def _official_add_noise(
        self,
        clean_unpatchified: Tensor,
        noise_unpatchified: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        assert isinstance(self.scheduler, FlowMatchScheduler)
        flat_clean = clean_unpatchified.flatten(0, 1)
        flat_noise = noise_unpatchified.flatten(0, 1)
        flat_timestep = timestep.to(device=flat_noise.device) * torch.ones(
            [flat_noise.shape[0]], device=flat_noise.device, dtype=torch.long
        )
        full_timesteps = self.scheduler._full_timesteps.to(flat_noise.device)
        full_sigmas = self.scheduler._full_sigmas.to(flat_noise.device)
        timestep_id = torch.argmin(
            (full_timesteps.unsqueeze(0) - flat_timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma = full_sigmas[timestep_id].reshape(-1, 1, 1, 1)
        noisy = (1 - sigma) * flat_clean + sigma * flat_noise
        return noisy.type_as(flat_noise).unflatten(0, clean_unpatchified.shape[:2])

    def _sample_official_layout(
        self,
        *,
        initial_noise: Tensor,
        predict_flow: Any,
    ) -> Tensor:
        assert isinstance(self.scheduler, FlowMatchScheduler), (
            "MemRoPE official-layout sampling currently expects FlowMatchScheduler"
        )
        input_dtype = initial_noise.dtype
        sigmas = self.scheduler.denoising_sigmas
        timesteps = self.scheduler.denoising_step_list

        noisy = initial_noise
        noisy_unpatchified = self.transformer.unpatchify_and_maybe_gather_cp(noisy)
        clean: Tensor | None = None
        clean_unpatchified: Tensor | None = None
        for i in range(timesteps.shape[0]):
            sigma = sigmas[i]
            timestep = self._official_timestep_tensor(timesteps[i], noisy)
            if i > 0:
                assert clean_unpatchified is not None
                noise_unpatchified = self._draw_unpatchified_renoise(
                    clean_unpatchified
                )
                noisy_unpatchified = self._official_add_noise(
                    clean_unpatchified,
                    noise_unpatchified,
                    timesteps[i],
                )
                noisy = self.transformer.patchify_and_maybe_split_cp(
                    noisy_unpatchified
                )
            flow = predict_flow(noisy, timestep)
            flow_unpatchified = self.transformer.unpatchify_and_maybe_gather_cp(flow)
            clean_unpatchified = (
                noisy_unpatchified.double() - sigma.double() * flow_unpatchified.double()
            ).to(input_dtype)
            clean = self.transformer.patchify_and_maybe_split_cp(clean_unpatchified)
        assert clean is not None, "denoising_step_list is empty"
        return clean.to(input_dtype)

    def generate(
        self,
        autoregressive_index: int,
        cache: TransformerCacheT,
        input: Any = None,
    ) -> tuple[Tensor, "DiffusionModel.FinalState[TransformerCacheT]"]:
        if input is not None:
            input = self.transformer.patchify_and_maybe_split_cp(input)
        cache.start(autoregressive_index)

        initial_noise = self.transformer.patchify_and_maybe_split_cp(
            self._draw_unpatchified_chunk_noise()
        )

        def predict_flow(noisy_latent: Tensor, timestep: Tensor) -> Tensor:
            return self.transformer.predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
            )

        clean_latent = self._sample_official_layout(
            initial_noise=initial_noise,
            predict_flow=predict_flow,
        )
        clean_latent = self.transformer.postprocess_clean_latent(
            clean_latent=clean_latent,
            cache=cache,
            input=input,
        )

        final_state = DiffusionModel.FinalState(
            clean_latent=clean_latent,
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )

        clean_latent = self.transformer.unpatchify_and_maybe_gather_cp(clean_latent)
        return clean_latent, final_state
