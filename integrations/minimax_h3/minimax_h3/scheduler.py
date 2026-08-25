# SPDX-FileCopyrightText: Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
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

"""FlashDreams scheduler for MiniMax H3's data-ward velocity."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowPredictor,
    Scheduler,
    SchedulerConfig,
)


@dataclass(kw_only=True)
class MiniMaxH3SchedulerConfig(SchedulerConfig):
    """Configuration for one of H3's modality-specific schedules."""

    _target: type[MiniMaxH3Scheduler] = field(
        default_factory=lambda: MiniMaxH3Scheduler
    )
    num_inference_steps: int = 30
    shift: float = 12.0


class MiniMaxH3Scheduler(Scheduler):
    """Rectified-flow Euler schedule used by the released H3 checkpoint."""

    config: MiniMaxH3SchedulerConfig

    def __init__(self, config: MiniMaxH3SchedulerConfig) -> None:
        super().__init__(config)
        if config.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")
        if config.shift <= 0:
            raise ValueError("shift must be positive")

    def schedule(self, device: torch.device | str) -> tuple[Tensor, Tensor]:
        """Return the shifted sigma grid and its H3 timesteps."""
        base = torch.linspace(
            1.0, 0.0, self.config.num_inference_steps, dtype=torch.float32
        )
        shift = self.config.shift
        sigmas = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
        sigmas = sigmas.to(device)
        return sigmas, 1.0 - sigmas[:-1]

    @staticmethod
    def step(
        sample: Tensor,
        flow: Tensor,
        timestep: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
    ) -> Tensor:
        """Take one deterministic data-ward Euler step."""
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        denoised = sample + sigma_from_timestep * flow
        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )
        ratio = sigma_next.to(sample.device, compute_dtype) / sigma.to(
            sample.device, compute_dtype
        )
        previous = ratio * sample.to(compute_dtype) + (1 - ratio) * denoised.to(
            compute_dtype
        )
        return previous.to(sample.dtype)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Denoise one stream with the H3 schedule."""
        del rng
        sigmas, timesteps = self.schedule(initial_noise.device)
        sample = initial_noise
        for index, timestep in enumerate(timesteps):
            flow = predict_flow(sample, timestep)
            sample = self.step(sample, flow, timestep, sigmas[index], sigmas[index + 1])
        return sample

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Mix clean input with Gaussian noise under H3's time convention."""
        noise = torch.randn(
            clean_input.shape,
            dtype=clean_input.dtype,
            device=clean_input.device,
            generator=rng,
        )
        time = timestep.to(clean_input.device, clean_input.dtype)
        while time.ndim < clean_input.ndim:
            time = time.unsqueeze(-1)
        return time * clean_input + (1 - time) * noise


__all__ = ["MiniMaxH3Scheduler", "MiniMaxH3SchedulerConfig"]
