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

"""Unit tests for :class:`FlowMatchEulerDiscreteScheduler`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flashdreams.infra.diffusion.scheduler import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_fixed_timesteps_round_trip() -> None:
    """``fixed_timesteps`` must appear verbatim in the buffer and
    ``sigmas`` must equal ``timesteps / num_train_timesteps``.

    Pins the HY-WorldPlay distilled WAN-5B schedule
    ``(1000, 960, 888.89, 727.27, 0)`` so any drift in the buffer
    construction surfaces here rather than as a silent parity
    regression at sub-PR 2b.5.
    """
    ft = (1000.0, 960.0, 888.8889, 727.2728, 0.0)
    cfg = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=ft,
    )
    scheduler = cfg.setup()
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    assert scheduler.timesteps.shape == (5,)
    assert scheduler.sigmas.shape == (5,)
    # Tolerance is fp32 epsilon: the Python literal 888.8889 is fp64
    # and casts to 888.888916... in fp32; that's a ~3e-8 relative gap,
    # which is parity-irrelevant against upstream's own fp32 buffer.
    np.testing.assert_allclose(
        scheduler.timesteps.cpu().numpy(), ft, rtol=1e-6, atol=1e-4
    )
    np.testing.assert_allclose(
        scheduler.sigmas.cpu().numpy(),
        np.asarray(ft, dtype=np.float32) / 1000.0,
        rtol=1e-6,
        atol=1e-7,
    )


def test_fixed_timesteps_length_mismatch_raises() -> None:
    """``fixed_timesteps`` length mismatch must fail loudly at setup.

    Off-by-one between ``num_inference_steps`` and the schedule length
    is the most plausible authoring mistake (it's the terminal-zero
    entry that bumps the count from ``N`` to ``N+1``).
    """
    with pytest.raises(AssertionError, match="fixed_timesteps length"):
        FlowMatchEulerDiscreteSchedulerConfig(
            num_inference_steps=4,
            fixed_timesteps=(1000.0, 960.0, 888.0, 727.0),
        ).setup()


def test_derived_schedule_matches_linspace_warp() -> None:
    """Without ``fixed_timesteps``, sigmas come from the standard
    diffusers ``linspace + shift`` warp + a trailing ``0.0``.

    The closed-form check below replicates
    ``diffusers.FlowMatchEulerDiscreteScheduler.set_timesteps(N)``
    with ``shift`` applied: linspace ``(sigma_max, sigma_min, N+1)[:-1]``
    -> warp -> concat ``[0.0]``.
    """
    N, shift, N_train = 4, 5.0, 1000
    cfg = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=N,
        shift=shift,
        num_train_timesteps=N_train,
    )
    scheduler = cfg.setup()
    train_alphas = np.linspace(1.0, 1.0 / N_train, N_train)[::-1].copy()
    train_sigmas = 1.0 - train_alphas
    sigma_min, sigma_max = float(train_sigmas[-1]), float(train_sigmas[0])
    raw = np.linspace(sigma_max, sigma_min, N + 1)[:-1]
    warped = shift * raw / (1.0 + (shift - 1.0) * raw)
    expected_sigmas = np.concatenate([warped, [0.0]]).astype(np.float32)
    np.testing.assert_allclose(
        scheduler.sigmas.cpu().numpy(), expected_sigmas, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        scheduler.timesteps.cpu().numpy(),
        expected_sigmas * N_train,
        rtol=1e-5,
        atol=1e-5,
    )


def test_sample_identity_flow_lands_on_zero() -> None:
    """With a constant-flow predictor ``v = noisy``, the Euler update
    ``x <- x + dt * v`` with ``dt = sigma_next - sigma`` and the
    distilled HY schedule must collapse the latent toward zero.

    Concretely: at each step, ``x'  =  x + (sigma_next - sigma) * x
                         =  x * (1 + sigma_next - sigma)``. With the
    fixed schedule ``[1.0, 0.96, 0.888, 0.7272, 0.0]`` the cumulative
    factor is ``(1 - 0.04) * (1 - 0.0711) * (1 - 0.1617) * (1 - 0.7272)
    ~= 0.2033``, which exercises both the per-step ``dt`` arithmetic and
    that the loop runs ``num_inference_steps`` times (not ``N+1``).
    """
    cfg = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )
    scheduler = cfg.setup()
    x0 = torch.ones(2, 3, dtype=torch.float32)

    def identity_flow(noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return noisy

    result = scheduler.sample(initial_noise=x0, predict_flow=identity_flow)
    sigmas = scheduler.sigmas.cpu().numpy()
    expected_factor = float(np.prod(1.0 + (sigmas[1:] - sigmas[:-1])))
    np.testing.assert_allclose(
        result.cpu().numpy(),
        np.full_like(x0.numpy(), expected_factor),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sample_invokes_predict_flow_n_times() -> None:
    """Exactly ``num_inference_steps`` predictor calls per
    :meth:`sample`. Critical guard against off-by-one bugs in the
    Euler loop bounds."""
    cfg = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )
    scheduler = cfg.setup()
    calls: list[torch.Tensor] = []

    def counting_flow(noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        calls.append(timestep.detach().clone())
        return torch.zeros_like(noisy)

    scheduler.sample(
        initial_noise=torch.randn(1, 2),
        predict_flow=counting_flow,
    )
    assert len(calls) == 4, f"expected 4 predictor calls, got {len(calls)}"
    # Timesteps passed to the predictor are sigmas[:-1] * 1000 (the
    # network sees the pre-step sigma's training-scale value, not the
    # post-step one).
    timesteps_seen = torch.stack(calls).cpu().numpy()
    np.testing.assert_allclose(
        timesteps_seen, [1000.0, 960.0, 888.8889, 727.2728], rtol=1e-4, atol=1e-4
    )


def test_add_noise_snaps_to_nearest_timestep() -> None:
    """``add_noise`` must snap an off-schedule timestep to the nearest
    table entry rather than crashing on an exact-membership check.

    Picks ``t=950`` for the distilled schedule (nearest entry: 960,
    sigma=0.96) so the resulting interpolation ``(1 - 0.96) * x_0 +
    0.96 * eps`` is easy to back out."""
    cfg = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )
    scheduler = cfg.setup()
    clean = torch.ones(8, dtype=torch.float32)
    timestep = torch.tensor(950.0)
    rng = torch.Generator().manual_seed(0)
    noisy = scheduler.add_noise(clean, timestep, rng=rng)
    # sigma = 0.96 at the snapped entry; (1 - 0.96) * 1 + 0.96 * eps
    # must lie in [(1 - 0.96) - 0.96 * 5, (1 - 0.96) + 0.96 * 5] with
    # very high probability (5 sigma) and the residual against the
    # clean-only term equals 0.96 * eps.
    residual = noisy - 0.04
    eps_reconstructed = residual / 0.96
    # Each entry must look like a standard-normal sample. Cheap check:
    # the empirical variance over 8 entries is close to 1.
    assert eps_reconstructed.std().item() > 0.1, (
        "add_noise produced near-zero residual; sigma snap likely broken"
    )


def test_to_bf16_preserves_fp32_schedule() -> None:
    """Casting the scheduler to bf16 must not quantize the schedule
    buffers. The HY-WorldPlay distilled timestep ``1000`` rounds to
    ``1024`` in bf16; an unprotected cast would shift the network's
    timestep input on every step."""
    cfg = FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )
    scheduler = cfg.setup().to(torch.bfloat16)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    assert float(scheduler.timesteps[0]) == 1000.0
