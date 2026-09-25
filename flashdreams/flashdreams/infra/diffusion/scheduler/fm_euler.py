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

"""Flow-matching Euler-discrete scheduler.

A first-order explicit Euler solver for flow-matching ODEs in
sigma-space. Mirrors ``diffusers.FlowMatchEulerDiscreteScheduler``'s
default behaviour while exposing an optional ``fixed_timesteps`` knob
so distilled few-step checkpoints (HY-WorldPlay's distilled WAN-5B,
upstream's ``few_step=True`` branch in
``wan/inference/pipeline_wan_w_mem_relative_rope.py``) can pin the
exact 5-entry timestep schedule
``(1000, 960, 888.89, 727.27, 0)`` instead of round-tripping through
``set_timesteps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from flashdreams.infra.diffusion.scheduler import (
    FlowPredictor,
    Scheduler,
    SchedulerConfig,
)


@dataclass(kw_only=True)
class FlowMatchEulerDiscreteSchedulerConfig(SchedulerConfig):
    """Config for the flow-matching Euler-discrete scheduler.

    Defaults match diffusers' :class:`FlowMatchEulerDiscreteScheduler`
    with ``num_train_timesteps=1000`` and the standard
    ``linspace + shift`` warp. Set :attr:`fixed_timesteps` to a length
    ``num_inference_steps + 1`` tuple (terminal value typically ``0.0``)
    to pin an externally-derived distilled schedule.
    """

    _target: type["FlowMatchEulerDiscreteScheduler"] = field(
        default_factory=lambda: FlowMatchEulerDiscreteScheduler
    )

    num_inference_steps: int = 4
    """Number of Euler steps. Matches the distilled WAN-5B 4-step path
    out of the box; bump to 30-50 for non-distilled checkpoints."""

    shift: float = 5.0
    """Schedule warp factor applied to the linspaced sigma grid. Ignored
    when :attr:`fixed_timesteps` is set."""

    num_train_timesteps: int = 1000
    """Length of the training sigma table. Used to convert between
    integer timesteps and ``[0, 1]`` sigmas (``sigma = timestep / N``)."""

    fixed_timesteps: tuple[float, ...] | None = None
    """Override the derived schedule with a precomputed list of
    timesteps. When set, must have length ``num_inference_steps + 1``;
    the trailing entry is typically ``0.0`` so the last Euler step
    lands on the clean latent. ``None`` (default) derives the schedule
    from :attr:`num_inference_steps` + :attr:`shift` via the standard
    flow-matching linspace + warp.

    The HY-WorldPlay distilled WAN-5B path pins this to
    ``(1000.0, 960.0, 888.8889, 727.2728, 0.0)``."""

    enable_tqdm: bool = False
    """Whether to enable the tqdm progress bar inside :meth:`sample`."""


class FlowMatchEulerDiscreteScheduler(Scheduler):
    """First-order explicit Euler solver for flow-matching ODEs.

    Each iteration runs::

        sigma_i, sigma_next = sigmas[i], sigmas[i + 1]
        flow = predict_flow(noisy, timesteps[i])
        noisy = noisy + (sigma_next - sigma_i) * flow

    After ``num_inference_steps`` iterations, ``sigma`` reaches ``0``
    and ``noisy`` holds the clean latent (the formula above collapses
    to ``x = x - sigma_i * flow`` at the terminal step). Bit-matches
    upstream HY-WorldPlay's
    ``wan/inference/pipeline_wan_w_mem_relative_rope.py`` few-step path
    when configured with the distilled 4-step :attr:`fixed_timesteps`.

    Examples:

        scheduler = FlowMatchEulerDiscreteSchedulerConfig(
            num_inference_steps=4,
            fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
        ).setup().to("cuda")
        clean = scheduler.sample(initial_noise=noise, predict_flow=fn)

    Schedule buffers (``timesteps`` + ``sigmas``) stay fp32 regardless
    of ``module.to(bf16)`` -- a stray cast would otherwise quantize the
    sigma grid and shift the inference timesteps by one LSB.
    """

    timesteps: Tensor
    sigmas: Tensor

    def __init__(self, config: FlowMatchEulerDiscreteSchedulerConfig) -> None:
        super().__init__(config)
        self.config: FlowMatchEulerDiscreteSchedulerConfig = config

        N = config.num_inference_steps
        N_train = config.num_train_timesteps
        assert N > 0, f"num_inference_steps must be > 0 (got {N})"

        if config.fixed_timesteps is not None:
            # Distilled / hand-tuned schedule path. Caller supplies an
            # N+1-entry timestep list; sigmas follow trivially as
            # timesteps/N_train. The trailing entry is typically 0.0 so
            # the last Euler step reaches the clean latent.
            ft = config.fixed_timesteps
            assert len(ft) == N + 1, (
                f"fixed_timesteps length {len(ft)} must equal "
                f"num_inference_steps + 1 = {N + 1}"
            )
            timesteps_np = np.asarray(ft, dtype=np.float32)
            sigmas_np = (timesteps_np / N_train).astype(np.float32)
        else:
            # Standard diffusers FlowMatchEulerDiscrete schedule: train
            # sigmas come from alphas = linspace(1, 1/N_train, N_train)[::-1]
            # / sigmas = 1 - alphas (no warp); inference sigmas are then
            # linspace(sigma_max, sigma_min, N+1)[:-1] warped by ``shift``
            # and capped with a terminal 0.0.
            train_alphas = np.linspace(1.0, 1.0 / N_train, N_train)[::-1].copy()
            train_sigmas = 1.0 - train_alphas
            sigma_min, sigma_max = float(train_sigmas[-1]), float(train_sigmas[0])
            inf_sigmas = np.linspace(sigma_max, sigma_min, N + 1)[:-1]
            inf_sigmas = (
                config.shift * inf_sigmas / (1.0 + (config.shift - 1.0) * inf_sigmas)
            )
            sigmas_np = np.concatenate([inf_sigmas, [0.0]]).astype(np.float32)
            timesteps_np = (sigmas_np * N_train).astype(np.float32)

        # Buffers move with .to(device) but stay fp32 -- a stray bf16
        # cast would otherwise round 1000 -> 1024 and shift every step.
        self.register_buffer(
            "timesteps",
            torch.from_numpy(timesteps_np),
            persistent=False,
        )
        self.register_buffer(
            "sigmas",
            torch.from_numpy(sigmas_np),
            persistent=False,
        )

    _FP32_BUFFERS = ("timesteps", "sigmas")

    def _apply(self, fn, recurse=True):
        """Move buffers with the parent ``.to(...)`` but keep them fp32.

        ``fn`` may be a lossy bf16 cast; snapshot the fp32 originals before
        ``super()._apply`` (which would overwrite them) and restore them
        with a pure device move afterwards.
        """
        saved = {name: getattr(self, name) for name in self._FP32_BUFFERS}
        super()._apply(fn, recurse=recurse)
        for name, original in saved.items():
            target_device = getattr(self, name).device
            setattr(self, name, original.to(device=target_device))
        return self

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Run the explicit Euler denoising loop.

        Each iteration takes one step ``noisy <- noisy + dt * flow`` in
        sigma-space, where ``dt = sigma_next - sigma`` is negative.
        Internal arithmetic stays in ``initial_noise.dtype`` (matches
        upstream HY-WorldPlay's behaviour, which does *not* promote to
        fp32 around the Euler update -- distilled checkpoints are
        trained to be robust at the network's native dtype).

        ``rng`` is unused (deterministic ODE) but accepted for interface
        conformance.
        """
        input_dtype = initial_noise.dtype
        N = self.config.num_inference_steps

        noisy = initial_noise
        for i in tqdm(
            range(N),
            disable=not self.config.enable_tqdm,
            desc="FlowMatchEulerDiscreteScheduler",
        ):
            # Schedule buffers are pinned to fp32 (preserves integer
            # timestep values under a stray ``module.to(bf16)``); cast
            # both the timestep handed to the network and the per-step
            # ``dt`` to the input dtype so downstream modulation /
            # Linear layers stay consistent.
            timestep = self.timesteps[i].to(dtype=input_dtype)
            dt = (self.sigmas[i + 1] - self.sigmas[i]).to(dtype=input_dtype)

            flow = predict_flow(noisy, timestep)
            noisy = noisy + dt * flow

        return noisy.to(input_dtype)

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Apply the forward corruption at an arbitrary timestep.

        Snaps ``timestep`` to the nearest entry of the inference
        schedule and applies ``x_t = (1 - sigma) * x_0 + sigma * eps``.
        """
        assert timestep.shape == (), f"expected scalar timestep, got {timestep.shape}"
        ts = self.timesteps
        idx = torch.argmin((ts - timestep.to(ts.dtype)).abs()).reshape(1)
        sigma = self.sigmas.index_select(0, idx).reshape(())
        noise = torch.empty_like(clean_input).normal_(generator=rng)
        return ((1.0 - sigma) * clean_input + sigma * noise).to(clean_input.dtype)
