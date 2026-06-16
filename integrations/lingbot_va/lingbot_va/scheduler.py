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
"""LingBot-VA upstream-compatible flow-match scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import FlowPredictor, Scheduler, SchedulerConfig


def warp_sigmas(sigmas: Tensor, shift: float) -> Tensor:
    """Apply upstream LingBot-VA sigma shift ``shift * s / (1 + (shift - 1) * s)``."""
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


@dataclass(kw_only=True)
class LingbotVAFlowMatchSchedulerConfig(SchedulerConfig):
    """Config matching ``raw/lingbot-va/wan_va/utils/scheduler.py`` inference behavior."""

    _target: type["LingbotVAFlowMatchScheduler"] = field(
        default_factory=lambda: LingbotVAFlowMatchScheduler
    )

    num_inference_steps: int = 25
    num_train_timesteps: int = 1000
    shift: float = 5.0
    sigma_max: float = 1.0
    sigma_min: float = 0.0
    inverse_timesteps: bool = False
    extra_one_step: bool = True
    reverse_sigmas: bool = False


class LingbotVAFlowMatchScheduler(Scheduler):
    """Explicit Euler flow-match scheduler used by LingBot-VA video/action loops."""

    timesteps: Tensor
    sigmas: Tensor

    def __init__(self, config: LingbotVAFlowMatchSchedulerConfig) -> None:
        super().__init__(config)
        self.config: LingbotVAFlowMatchSchedulerConfig = config
        sigmas, timesteps = self.build_schedule(config)
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.register_buffer("timesteps", timesteps, persistent=False)

    @staticmethod
    def build_schedule(config: LingbotVAFlowMatchSchedulerConfig) -> tuple[Tensor, Tensor]:
        sigma_start = config.sigma_min + (config.sigma_max - config.sigma_min)
        if config.extra_one_step:
            sigmas = torch.linspace(
                sigma_start,
                config.sigma_min,
                config.num_inference_steps + 1,
                dtype=torch.float32,
            )[:-1]
        else:
            sigmas = torch.linspace(
                sigma_start,
                config.sigma_min,
                config.num_inference_steps,
                dtype=torch.float32,
            )
        if config.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        sigmas = warp_sigmas(sigmas, config.shift)
        if config.reverse_sigmas:
            sigmas = 1.0 - sigmas
        timesteps = sigmas * config.num_train_timesteps
        return sigmas, timesteps

    @property
    def padded_timesteps(self) -> Tensor:
        """Timesteps with the trailing zero that upstream appends before the final KV update."""
        return torch.nn.functional.pad(self.timesteps, (0, 1), mode="constant", value=0)

    def step(self, model_output: Tensor, timestep: Tensor, sample: Tensor) -> Tensor:
        """Apply one upstream Euler step in sigma space."""
        timestep_cpu = timestep.detach().to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        timestep_id = torch.argmin((self.timesteps - timestep_cpu).abs())
        sigma = self.sigmas[timestep_id].to(device=sample.device, dtype=sample.dtype)
        if int(timestep_id.item()) + 1 >= self.sigmas.shape[0]:
            sigma_next = torch.zeros((), device=sample.device, dtype=sample.dtype)
        else:
            sigma_next = self.sigmas[timestep_id + 1].to(device=sample.device, dtype=sample.dtype)
        return sample + model_output * (sigma_next - sigma)

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Run Euler denoising over ``timesteps``. ``rng`` is accepted for interface parity."""
        del rng
        sample = initial_noise
        for timestep in self.timesteps:
            flow = predict_flow(sample, timestep.to(device=sample.device, dtype=sample.dtype))
            sample = self.step(flow, timestep, sample)
        return sample

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Apply upstream forward corruption at the nearest scheduler timestep."""
        noise = torch.empty_like(clean_input).normal_(generator=rng)
        timestep_cpu = timestep.detach().to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        timestep_id = torch.argmin((self.timesteps - timestep_cpu).abs())
        sigma = self.sigmas[timestep_id].to(device=clean_input.device, dtype=clean_input.dtype)
        return (1.0 - sigma) * clean_input + sigma * noise
