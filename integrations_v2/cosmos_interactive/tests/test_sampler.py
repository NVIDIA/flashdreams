# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-step sampler math (matches Cosmos3 ``FixedStepSampler``)."""

from __future__ import annotations

import pytest
import torch
from cosmos_interactive._sampler import fixed_step_denoise

pytestmark = pytest.mark.ci_cpu


def test_ode_euler_step_matches_formula() -> None:
    # Constant flow -> deterministic Euler integration.
    v = torch.tensor([2.0, -1.0])
    x0 = torch.zeros(2)
    t_list = [1.0, 0.5]  # -> [1.0, 0.5, 0.0]

    def vf(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return v

    out = fixed_step_denoise(vf, x0, t_list=t_list, sample_type="ode")
    # step0: 1.0->0.5: latent = 0 + (0.5-1.0)*v = -0.5v
    # step1: 0.5->0.0: latent = x0_pred = latent - 0.5*v = -0.5v - 0.5v = -1.0v
    assert torch.allclose(out, -1.0 * v)


def test_final_step_returns_x0() -> None:
    # Single-step schedule [1.0, 0.0]: latent_next = x0_pred = latent - 1.0*v.
    v = torch.tensor([3.0])
    noise = torch.tensor([1.0])

    def vf(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return v

    out = fixed_step_denoise(vf, noise, t_list=[1.0], sample_type="ode")
    assert torch.allclose(out, noise - 1.0 * v)


def test_timestep_passed_is_sigma_times_train_steps() -> None:
    seen: list[float] = []

    def vf(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        seen.append(float(timestep))
        return torch.zeros_like(latent)

    fixed_step_denoise(
        vf, torch.zeros(1), t_list=[1.0, 0.5], num_train_timesteps=1000.0
    )
    assert seen == [1000.0, 500.0]  # sigma_cur * num_train_timesteps


def test_sde_is_deterministic_with_generator() -> None:
    def vf(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return 0.1 * latent

    noise = torch.randn(4)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = fixed_step_denoise(
        vf, noise, t_list=[1.0, 0.5], sample_type="sde", generator=g1
    )
    b = fixed_step_denoise(
        vf, noise, t_list=[1.0, 0.5], sample_type="sde", generator=g2
    )
    assert torch.allclose(a, b)


def test_condition_mask_pins_reference() -> None:
    v = torch.zeros(3)

    def vf(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return v

    noise = torch.tensor([5.0, 5.0, 5.0])
    ref = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([1.0, 0.0, 1.0])  # pin positions 0 and 2
    out = fixed_step_denoise(
        vf,
        noise,
        t_list=[1.0],
        sample_type="ode",
        condition_reference=ref,
        condition_mask=mask,
    )
    # v=0 -> x0_pred = noise; masked positions replaced by ref.
    assert out[0] == 1.0 and out[2] == 3.0
    assert out[1] == noise[1]
