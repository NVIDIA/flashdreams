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

"""SANA-WM LTX-style Euler scheduler boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowPredictor,
    Scheduler,
    SchedulerConfig,
)


@dataclass(kw_only=True)
class SanaWMLTXEulerSchedulerConfig(SchedulerConfig):
    """Config for SANA-WM's LTX-style flow-matching Euler scheduler."""

    _target: type["SanaWMLTXEulerScheduler"] = field(
        default_factory=lambda: SanaWMLTXEulerScheduler
    )

    num_inference_steps: int = 60
    """Default number of Euler steps when using the generic scheduler API."""

    shift: float = 9.8
    """Default flow-match schedule shift when using the generic scheduler API."""

    num_train_timesteps: int = 1000
    """Training timestep scale used by the public SANA-WM release."""

    denoising_step_list: tuple[int, ...] | None = None
    """Optional explicit training-scale timestep schedule.

    Streaming SANA-WM uses a distilled student with a fixed schedule that
    already includes the intended flow shift. When this is set, the scheduler
    uses these values verbatim and requires the final entry to be ``0``.
    """


class SanaWMLTXEulerScheduler(Scheduler):
    """Euler scheduler with SANA-WM per-token timestep support."""

    config: SanaWMLTXEulerSchedulerConfig

    def __init__(self, config: SanaWMLTXEulerSchedulerConfig) -> None:
        super().__init__(config)
        self.config = config

    def timesteps(
        self,
        *,
        num_inference_steps: int,
        shift: float,
        device: torch.device | str,
    ) -> Tensor:
        """Return diffusers-compatible FlowMatch Euler timesteps."""
        if self.config.denoising_step_list is not None:
            del num_inference_steps, shift
            return _explicit_timesteps(
                self.config.denoising_step_list,
                device=device,
                num_train_timesteps=self.config.num_train_timesteps,
            )

        steps = int(num_inference_steps)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {steps}.")
        num_train_timesteps = int(self.config.num_train_timesteps)
        init_timesteps = np.linspace(
            1,
            num_train_timesteps,
            num_train_timesteps,
            dtype=np.float32,
        )[::-1].copy()
        init_sigmas = init_timesteps / num_train_timesteps
        init_sigmas = shift * init_sigmas / (1.0 + (shift - 1.0) * init_sigmas)
        sigma_min = float(init_sigmas[-1])
        sigma_max = float(init_sigmas[0])
        timesteps = np.linspace(
            sigma_max * num_train_timesteps,
            sigma_min * num_train_timesteps,
            steps,
        )
        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        return torch.from_numpy(sigmas * num_train_timesteps).to(device=device)

    def step_ltx(
        self,
        *,
        model_output: Tensor,
        timestep: Tensor,
        next_timestep: Tensor,
        sample: Tensor,
        per_token_timesteps: Tensor | None = None,
        schedule_timesteps: Tensor | None = None,
    ) -> Tensor:
        """Apply one SANA-WM LTX Euler step in token layout.

        Args:
            model_output: Flow tensor in ``[B, N, C]`` token layout.
            timestep: Current scalar training-scale timestep.
            next_timestep: Next scalar training-scale timestep.
            sample: Current latent in ``[B, N, C]`` token layout.
            per_token_timesteps: Optional ``[B, N]`` timestep table. Tokens
                with timestep ``0`` stay at sigma zero, which matches the
                first-frame-pinning branch used by SANA-WM.
            schedule_timesteps: Optional full scheduler timestep table used
                to find the next lower sigma for each per-token timestep.

        Returns:
            Updated sample in ``[B, N, C]`` layout.
        """
        num_train_timesteps = float(self.config.num_train_timesteps)
        sample = sample.to(torch.float32)
        if per_token_timesteps is None:
            sigma = timestep.to(device=sample.device, dtype=sample.dtype)
            sigma_next = next_timestep.to(device=sample.device, dtype=sample.dtype)
            dt = (sigma_next - sigma) / num_train_timesteps
            return (sample + dt * model_output).to(model_output.dtype)

        per_token_sigmas = (
            per_token_timesteps.to(device=sample.device, dtype=sample.dtype)
            / num_train_timesteps
        )
        if schedule_timesteps is not None:
            sigmas = (
                schedule_timesteps.to(device=sample.device, dtype=sample.dtype)
                / num_train_timesteps
            )
            sigmas = sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = (lower_mask * sigmas).max(dim=0).values
        else:
            next_sigma_scalar = (
                next_timestep.to(device=sample.device, dtype=sample.dtype)
                / num_train_timesteps
            )
            lower_sigmas = torch.where(
                per_token_sigmas > next_sigma_scalar + 1e-6,
                next_sigma_scalar.expand_as(per_token_sigmas),
                torch.zeros_like(per_token_sigmas),
            )
        dt = per_token_sigmas - lower_sigmas
        return sample + dt.unsqueeze(-1) * model_output

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Run the generic scalar-timestep Euler loop."""
        del rng
        timesteps = self.timesteps(
            num_inference_steps=self.config.num_inference_steps,
            shift=self.config.shift,
            device=initial_noise.device,
        )
        noisy = initial_noise
        for index, timestep in enumerate(timesteps[:-1]):
            flow = predict_flow(noisy, timestep.to(dtype=initial_noise.dtype))
            noisy = self.step_ltx(
                model_output=flow,
                timestep=timestep,
                next_timestep=timesteps[index + 1],
                sample=noisy,
            )
        return noisy.to(initial_noise.dtype)

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Apply forward flow-match corruption at ``timestep``."""
        sigma = timestep.to(device=clean_input.device, dtype=clean_input.dtype)
        sigma = sigma / float(self.config.num_train_timesteps)
        noise = torch.randn(
            clean_input.shape,
            generator=rng,
            device=clean_input.device,
            dtype=clean_input.dtype,
        )
        return ((1.0 - sigma) * clean_input + sigma * noise).to(clean_input.dtype)


__all__ = [
    "SanaWMLTXEulerScheduler",
    "SanaWMLTXEulerSchedulerConfig",
]


def _explicit_timesteps(
    values: tuple[int, ...],
    *,
    device: torch.device | str,
    num_train_timesteps: int,
) -> Tensor:
    """Validate and materialize a fixed training-scale timestep schedule."""
    if len(values) < 2:
        raise ValueError("denoising_step_list must contain at least two timesteps.")
    if int(values[-1]) != 0:
        raise ValueError("denoising_step_list must end with 0.")
    if any(int(value) < 0 for value in values):
        raise ValueError("denoising_step_list cannot contain negative timesteps.")
    if any(int(a) < int(b) for a, b in zip(values, values[1:])):
        raise ValueError("denoising_step_list must be monotonically non-increasing.")
    upper = int(num_train_timesteps)
    if any(int(value) > upper for value in values):
        raise ValueError(
            "denoising_step_list cannot exceed num_train_timesteps="
            f"{num_train_timesteps}."
        )
    return torch.tensor(
        tuple(float(value) for value in values),
        dtype=torch.float32,
        device=device,
    )
