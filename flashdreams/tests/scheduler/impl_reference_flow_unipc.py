"""Frozen reference for :mod:`flashdreams.infra.diffusion.scheduler.fm_unipc`.

Verbatim snapshot of the production scheduler at the moment the cleanup
PR started. Used by :mod:`test_scheduler_equivalence` to guard the
rewrite. The 640-line inner solver lives in :mod:`_flow_unipc_inner`
(also a verbatim snapshot of ``flow_unipc/impl.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from _flow_unipc_inner import (  # type: ignore[import-not-found]
    FlowUniPCMultistepScheduler as _FlowUniPCMultistepScheduler,
)
from torch import Tensor


@dataclass(kw_only=True)
class FlowUniPCReferenceConfig:
    """Mirror of ``FlowMatchUniPCSchedulerConfig`` (no ``InstantiateConfig`` glue)."""

    num_inference_steps: int = 50
    shift: float = 5.0
    num_train_timesteps: int = 1000
    solver_order: int = 2


class FlowUniPCSchedulerReference(nn.Module):
    """Reference (legacy) implementation, frozen at start of cleanup PR."""

    def __init__(self, config: FlowUniPCReferenceConfig) -> None:
        super().__init__()
        self.config = config

        self._fm = _FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            solver_order=config.solver_order,
            shift=1.0,
        )
        self._cached_device: torch.device | None = None

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        device = initial_noise.device
        rebuild_schedule = device != self._cached_device
        self._fm.set_timesteps(
            num_inference_steps=self.config.num_inference_steps,
            device=device if rebuild_schedule else self._cached_device,
            shift=self.config.shift,
        )
        if rebuild_schedule:
            self._cached_device = device

        x = initial_noise
        for timestep in self._fm.timesteps:
            flow = predict_flow(x, timestep)
            x = self._fm.step(
                model_output=flow,
                timestep=timestep,
                sample=x,
            ).prev_sample  # ty:ignore[unresolved-attribute]
        return x

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        noise = torch.randn_like(clean_input, generator=rng)
        return self._fm.add_noise(
            original_samples=clean_input,
            noise=noise,
            timesteps=timestep.reshape(1).to(
                device=clean_input.device, dtype=torch.int64
            ),  # ty:ignore[invalid-argument-type]
        )
