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

"""Tests for self-forcing flow-match scheduling."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig

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


def test_flow_match_uses_the_sample_dtype_for_scheduler_math() -> None:
    """Promote a BF16 network result before the distilled FP32 update."""
    scheduler = FlowMatchSchedulerConfig(
        num_inference_steps=2,
        shift=5.0,
        denoising_timesteps=[1000, 500],
    ).setup()
    initial = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    flow = torch.tensor([0.123, -0.456, 0.789], dtype=torch.bfloat16)

    generator = torch.Generator().manual_seed(123)
    actual = scheduler.sample(
        initial_noise=initial,
        predict_flow=lambda _sample, _timestep: flow,
        rng=generator,
    )

    noise = torch.empty_like(initial).normal_(
        generator=torch.Generator().manual_seed(123)
    )
    first = initial - scheduler.denoising_sigmas[0] * flow.float()
    noisy = (1.0 - scheduler.denoising_sigmas[1]) * first
    noisy += scheduler.denoising_sigmas[1] * noise
    expected = noisy - scheduler.denoising_sigmas[1] * flow.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
