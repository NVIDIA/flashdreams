"""Frozen reference for :mod:`flashdreams.infra.diffusion.scheduler.fm`.

This is a verbatim snapshot of the production scheduler at the moment
the cleanup PR started, used by :mod:`test_scheduler_equivalence` to
guard the rewrite. Do not edit; if the upstream behavior intentionally
changes, regenerate by re-copying.

Contents:

- :class:`_FlowMatchSchedulerInner` - DiffSynth-style sigma/timestep
  helper (was ``flow_match/impl.py``).
- :class:`FlowMatchSchedulerReference` - the public wrapper (was
  ``flow_match/__init__.py``: ``FlowMatchScheduler``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Inner solver (verbatim from flow_match/impl.py).
# ---------------------------------------------------------------------------


class _FlowMatchSchedulerInner:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self, num_inference_steps=100, denoising_strength=1.0, training=False
    ):
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        batch_shape = timestep.shape
        batch_size = math.prod(batch_shape)
        timestep = timestep.reshape(batch_size)
        sigmas = self.sigmas.to(timestep.device)
        timesteps = self.timesteps.to(timestep.device)
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id]
        sigma = sigma.to(timestep.dtype)
        return sigma.reshape(batch_shape)


# ---------------------------------------------------------------------------
# Wrapper (verbatim from flow_match/__init__.py at the start of cleanup).
# ---------------------------------------------------------------------------


def _add_noise(
    clean_input: Tensor, sigma: Tensor, rng: torch.Generator | None = None
) -> Tensor:
    noise = torch.randn_like(clean_input, generator=rng)
    return (1.0 - sigma) * clean_input + sigma * noise


def _denoise(noisy_input: Tensor, sigma: Tensor, predicted_flow: Tensor) -> Tensor:
    assert noisy_input.shape == predicted_flow.shape
    return noisy_input - sigma * predicted_flow


@dataclass(kw_only=True)
class FlowMatchReferenceConfig:
    """Mirror of ``FlowMatchSchedulerConfig`` (no ``InstantiateConfig`` glue)."""

    num_inference_steps: int = 4
    shift: float = 8.0
    denoising_timesteps: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True
    num_train_timesteps: int = 1000
    sigma_min: float = 0.0
    extra_one_step: bool = True


class FlowMatchSchedulerReference(nn.Module):
    """Reference (legacy) implementation, frozen at start of cleanup PR."""

    def __init__(self, config: FlowMatchReferenceConfig) -> None:
        super().__init__()
        self.config = config

        assert config.num_inference_steps == len(config.denoising_timesteps)

        self._fm = _FlowMatchSchedulerInner(
            shift=config.shift,
            sigma_min=config.sigma_min,
            extra_one_step=config.extra_one_step,
        )
        self._fm.set_timesteps(config.num_train_timesteps, training=True)

        if config.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self._fm.timesteps.cpu(),
                    torch.tensor([0.0], dtype=torch.float32),
                )
            )
            denoising_step_list = timesteps[
                config.num_train_timesteps
                - torch.tensor(config.denoising_timesteps, dtype=torch.long)
            ]
        else:
            denoising_step_list = torch.tensor(
                config.denoising_timesteps, dtype=torch.float32
            )
        self.register_buffer(
            "denoising_step_list", denoising_step_list, persistent=False
        )

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        noisy = initial_noise
        clean: Tensor | None = None
        for i, t_cpu in enumerate(self.denoising_step_list):  # ty:ignore[invalid-argument-type]
            timestep = t_cpu.to(device=initial_noise.device, dtype=initial_noise.dtype)  # ty:ignore[unresolved-attribute]
            sigma = self._timestep_to_sigma(timestep, like=noisy)
            if i > 0:
                assert clean is not None
                noisy = _add_noise(clean, sigma, rng=rng)
            flow = predict_flow(noisy, timestep)
            clean = _denoise(noisy, sigma, flow)
        assert clean is not None
        return clean

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        sigma = self._timestep_to_sigma(timestep, like=clean_input)
        return _add_noise(clean_input, sigma, rng=rng)

    def _timestep_to_sigma(self, timestep: Tensor, like: Tensor) -> Tensor:
        assert timestep.shape == ()
        sigma = self._fm.timestep_to_sigma(timestep.reshape(1)).reshape(())
        return sigma.to(device=like.device, dtype=like.dtype)
