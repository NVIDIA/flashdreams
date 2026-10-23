# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-step sampler for a DMD2-distilled student.

Per-frame and per-step seed strides make noise reproducible across rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]


def fixed_step_denoise(
    velocity_fn: VelocityFn,
    noise: Tensor,
    *,
    t_list: list[float],
    sample_type: str = "sde",
    num_train_timesteps: float = 1000.0,
    generator: torch.Generator | None = None,
    sde_noise_fn: Callable[[int], Tensor] | None = None,
    condition_reference: Tensor | None = None,
    condition_mask: Tensor | None = None,
    fp32_math: bool = False,
) -> Tensor:
    """Run the fixed-step denoise loop; returns the final clean latent.

    Args:
        velocity_fn: ``(latent, timestep) -> flow``; ``timestep`` is
            ``sigma_cur * num_train_timesteps`` (raw diffusion time).
        noise: initial noise latent (any shape).
        t_list: sigma schedule (0.0 auto-appended if absent).
        sample_type: ``"sde"`` (this checkpoint) or ``"ode"``.
        num_train_timesteps: scales sigma → raw timestep for ``velocity_fn``.
        generator: RNG for SDE re-noising (used only when ``sde_noise_fn`` is None).
        sde_noise_fn: ``step_idx -> eps`` (same shape as the latent). When given,
            it supplies the SDE re-noise per step — the rollout uses this to match
            the model's exact per-step seeding (``seed + frame_idx*1_000_003 +
            step_idx*9_176``); ``generator`` is ignored.
        condition_reference, condition_mask: optional clean values to pin
            (``mask==1`` keeps reference); both set or both None.
        fp32_math: accumulate the sampler arithmetic in float32 even when
            ``velocity_fn`` returns bf16. Only a few steps run, so the cost is
            negligible next to the network, and it keeps the distilled schedule
            from losing precision in the re-noise term.
    """
    assert (condition_reference is None) == (condition_mask is None), (
        "condition_reference and condition_mask must be both set or both None"
    )
    schedule = list(t_list)
    if schedule[-1] != 0.0:
        schedule.append(0.0)
    assert len(schedule) >= 2

    latent = noise.float() if fp32_math else noise
    for step_idx, (sigma_cur, sigma_next) in enumerate(pairwise(schedule)):
        timestep = torch.tensor(
            sigma_cur * num_train_timesteps,
            device=latent.device,
            dtype=torch.float32 if fp32_math else None,
        )
        v_pred = velocity_fn(latent, timestep)
        if fp32_math:
            # A Python float times a float32 tensor stays float32, so promoting
            # the operands here keeps the whole step in float32.
            latent = latent.float()
            v_pred = v_pred.float()
        x0_pred = latent - sigma_cur * v_pred

        if sigma_next > 0:
            if sample_type == "ode":
                latent = latent + (sigma_next - sigma_cur) * v_pred
            else:
                if sde_noise_fn is not None:
                    eps = sde_noise_fn(step_idx).to(
                        device=x0_pred.device, dtype=x0_pred.dtype
                    )
                else:
                    eps = torch.randn(
                        x0_pred.shape,
                        device=x0_pred.device,
                        dtype=x0_pred.dtype,
                        generator=generator,
                    )
                latent = (1.0 - sigma_next) * x0_pred + sigma_next * eps
        else:
            latent = x0_pred

        if condition_reference is not None:
            assert condition_mask is not None
            ref = condition_reference.to(dtype=latent.dtype, device=latent.device)
            m = condition_mask.to(dtype=latent.dtype, device=latent.device)
            latent = m * ref + (1.0 - m) * latent

    return latent
