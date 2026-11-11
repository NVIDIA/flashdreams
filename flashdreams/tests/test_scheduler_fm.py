# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for self-forcing flow-match scheduling."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowMatchSchedulerConfig,
    FlowMatchUniPCSchedulerConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_flow_match_uses_the_supplied_generator_for_transition_noise() -> None:
    """Advance the supplied generator for self-forcing transition noise."""
    scheduler = FlowMatchSchedulerConfig(
        num_inference_steps=2,
        shift=5.0,
        denoising_timesteps=[1000, 500],
    ).setup()
    initial = torch.zeros(2, 3)

    def zero_flow(sample: Tensor, timestep: Tensor) -> Tensor:
        del timestep
        return torch.zeros_like(sample)

    actual = scheduler.sample(
        initial_noise=initial,
        predict_flow=zero_flow,
        rng=torch.Generator().manual_seed(123),
    )
    noise = torch.empty_like(initial).normal_(
        generator=torch.Generator().manual_seed(123)
    )
    sigma = scheduler.denoising_sigmas[1]
    torch.testing.assert_close(actual, (1.0 - sigma) * initial + sigma * noise)


def test_unipc_single_step_returns_the_x0_estimate() -> None:
    """A one-step UniPC schedule lands on sigma 0 in its only predictor step."""
    scheduler = FlowMatchUniPCSchedulerConfig(num_inference_steps=1).setup()
    initial = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))
    flow = torch.randn(2, 3, generator=torch.Generator().manual_seed(1))

    actual = scheduler.sample(
        initial_noise=initial,
        predict_flow=lambda sample, timestep: flow,
    )

    torch.testing.assert_close(actual, initial - scheduler.sigmas[0] * flow)
