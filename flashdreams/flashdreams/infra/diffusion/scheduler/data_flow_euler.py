# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-ward rectified-flow Euler sampling on a shifted endpoint-inclusive grid."""

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler.base import (
    FlowPredictor,
    Scheduler,
    SchedulerConfig,
)
from flashdreams.infra.diffusion.scheduler.synchronized import sample_synchronized


@dataclass(kw_only=True)
class DataFlowEulerSchedulerConfig(SchedulerConfig):
    """Shifted sigma grid with a velocity pointing toward clean data."""

    _target: type["DataFlowEulerScheduler"] = field(
        default_factory=lambda: DataFlowEulerScheduler
    )
    num_inference_steps: int = 30
    """Number of grid points; there are one fewer model evaluations."""
    shift: float = 12.0
    """Positive rational warp of the endpoint-inclusive sigma grid."""


class DataFlowEulerScheduler(Scheduler):
    """Blend the current sample and its data prediction in sigma space."""

    def __init__(self, config: DataFlowEulerSchedulerConfig) -> None:
        super().__init__(config)
        if config.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2 grid points")
        if not math.isfinite(config.shift) or config.shift <= 0:
            raise ValueError("shift must be finite and positive")
        base = torch.linspace(
            1.0, 0.0, config.num_inference_steps, dtype=torch.float32, device="cpu"
        )
        sigmas = torch.unique_consecutive(
            config.shift * base / (1 + (config.shift - 1) * base)
        )
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.register_buffer("timesteps", 1.0 - sigmas[:-1], persistent=False)

    def _apply(self, fn, recurse=True):
        grids = {name: getattr(self, name) for name in ("sigmas", "timesteps")}
        super()._apply(fn, recurse=recurse)
        for name, original in grids.items():
            setattr(self, name, original.to(device=getattr(self, name).device))
        return self

    def step(self, sample: Tensor, flow: Tensor, index: int) -> Tensor:
        """Advance one step, preserving data-ward blend rounding."""
        time = self.timesteps[index].to(sample.device, sample.dtype)
        denoised = sample + (1 - time) * flow
        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )
        ratio = self.sigmas[index + 1].to(sample.device, compute_dtype) / self.sigmas[
            index
        ].to(sample.device, compute_dtype)
        return (
            ratio * sample.to(compute_dtype) + (1 - ratio) * denoised.to(compute_dtype)
        ).to(sample.dtype)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Sample one stream through the shared synchronized loop."""
        del rng
        return sample_synchronized(
            (initial_noise,),
            (self,),
            lambda samples, times: (predict_flow(samples[0], times[0]),),
        )[0]

    def add_noise(
        self, clean_input: Tensor, timestep: Tensor, rng: torch.Generator | None = None
    ) -> Tensor:
        """Mix clean data and Gaussian noise under the data-ward time convention."""
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
