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

"""Flow-matching scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowPredictor,
    Scheduler,
    SchedulerConfig,
)


def _warp(sigmas: Tensor, shift: float) -> Tensor:
    """``shift * s / (1 + (shift - 1) * s)`` -- DiffSynth schedule warp."""
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


@dataclass(kw_only=True)
class FlowMatchSchedulerConfig(SchedulerConfig):
    """Config for the flow-matching scheduler."""

    _target: type["FlowMatchScheduler"] = field(
        default_factory=lambda: FlowMatchScheduler
    )

    num_inference_steps: int = 4
    """Must equal ``len(denoising_timesteps)``."""

    shift: float = 8.0
    """Schedule warp factor."""

    denoising_timesteps: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    """Per-step diffusion timesteps in ``[0, num_train_timesteps]``."""

    warp_denoising_step: bool = True
    """Map ``denoising_timesteps`` through the warped sigma schedule."""

    num_train_timesteps: int = 1000
    """Length of the training sigma table."""

    sigma_min: float = 0.0
    """Floor of the warped sigma schedule. Only ``0.0`` is supported here;
    asserted at construction."""

    extra_one_step: bool = True
    """Append the ``sigma=0`` step to the schedule. Only ``True`` is
    supported here; asserted at construction."""


class FlowMatchScheduler(Scheduler):
    """Flow-matching scheduler with self-forcing renoise (DiffSynth-style).

    Each iteration converts the predicted flow to an ``x0`` estimate, then
    re-noises at the same sigma to feed the next iteration. The final
    ``x0`` is returned::

        x_t = initial_noise
        for t in denoising_step_list:
            v = predict_flow(x_t, t)
            x0 = x_t - sigma(t) * v
            x_t = (1 - sigma(t)) * x0 + sigma(t) * eps
        return x0

    Examples:

        scheduler = FlowMatchSchedulerConfig(
            num_inference_steps=4,
            shift=8.0,
            denoising_timesteps=[1000, 750, 500, 250],
        ).setup().to("cuda")
        clean = scheduler.sample(initial_noise=noise, predict_flow=fn)

    Schedule buffers are pinned to fp32 even after ``module.to(bf16)``;
    integer timesteps like 1000 would otherwise round to 1024.
    """

    denoising_step_list: Tensor
    denoising_sigmas: Tensor
    _full_sigmas: Tensor
    _full_timesteps: Tensor

    def __init__(self, config: FlowMatchSchedulerConfig) -> None:
        super().__init__(config)
        self.config: FlowMatchSchedulerConfig = config

        N = config.num_train_timesteps
        assert config.num_inference_steps == len(config.denoising_timesteps), (
            f"num_inference_steps ({config.num_inference_steps}) must equal "
            f"len(denoising_timesteps) ({len(config.denoising_timesteps)})"
        )
        assert config.extra_one_step, "extra_one_step=False not exercised; not ported"
        assert config.sigma_min == 0.0, "sigma_min != 0 not exercised; not ported"

        # Full warped schedule: identical to DiffSynth's
        #   sigmas = linspace(1, 0, N + 1)[:-1]
        #   sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        # (matches reference exactly in fp32).
        full_sigmas = _warp(
            torch.linspace(1.0, 0.0, N + 1, dtype=torch.float32)[:-1],
            config.shift,
        )
        full_timesteps = full_sigmas * N

        # Pre-resolve per-step (sigma, timestep) so sample() does no
        # per-step argmin. Replicates the legacy resolution exactly:
        #   - warp_denoising_step=True: denoising_step_list[i] is read
        #     from a (full_timesteps ++ [0.0]) buffer at index
        #     N - denoising_timesteps[i]. The legacy then argmin's that
        #     value against the 1000-entry full_timesteps to get sigma.
        #     For idx in [0, N), the answer is full_sigmas[idx]; the
        #     idx==N corner (only hit when denoising_timesteps[i]==0)
        #     argmin's 0.0 to the smallest entry, full_sigmas[N-1].
        #   - warp_denoising_step=False: denoising_step_list[i] is the
        #     raw int; legacy argmin's it against full_timesteps and
        #     returns the snapped sigma.
        idxs = [N - t for t in config.denoising_timesteps]
        if config.warp_denoising_step:
            step_list = [full_timesteps[idx].item() if idx < N else 0.0 for idx in idxs]
            sigma_list = [full_sigmas[idx if idx < N else N - 1].item() for idx in idxs]
        else:
            step_list = [float(t) for t in config.denoising_timesteps]
            snapped_idx = [
                int(torch.argmin((full_timesteps - t).abs()).item()) for t in step_list
            ]
            sigma_list = [full_sigmas[i].item() for i in snapped_idx]

        # Buffers move with .to(device) but are pinned to fp32 by the
        # ``_apply`` override below -- a stray ``model.to(bf16)`` would
        # otherwise round integer timesteps (1000 -> 1024) and quantize
        # the sigma table.
        self.register_buffer(
            "denoising_step_list",
            torch.tensor(step_list, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "denoising_sigmas",
            torch.tensor(sigma_list, dtype=torch.float32),
            persistent=False,
        )
        # Full table only used by add_noise (rare path, called from
        # finalize when context_noise > 0).
        self.register_buffer("_full_sigmas", full_sigmas, persistent=False)
        self.register_buffer("_full_timesteps", full_timesteps, persistent=False)

    # Pinned to fp32 by ``_apply`` regardless of the parent module's dtype.
    _FP32_BUFFERS = (
        "denoising_step_list",
        "denoising_sigmas",
        "_full_sigmas",
        "_full_timesteps",
    )

    def _apply(self, fn, recurse=True):
        """Move buffers with the parent ``.to(...)`` but keep them fp32.

        ``fn`` may be a lossy bf16 cast; snapshot the fp32 originals before
        ``super()._apply`` (which would overwrite them) and restore them
        with a pure device move afterward.
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
        """Run the self-forcing flow-match denoising loop.

        Iteration 0 trusts ``initial_noise`` as the ``sigma=1`` sample;
        later iterations re-noise the previous ``x0`` estimate to the new
        sigma before the network forward. Schedule arithmetic auto-promotes
        to fp32; the result is cast back to ``initial_noise.dtype``.
        """
        input_dtype = initial_noise.dtype
        sigmas = self.denoising_sigmas
        timesteps = self.denoising_step_list

        noisy = initial_noise
        clean: Tensor | None = None
        for i in range(timesteps.shape[0]):
            sigma = sigmas[i]
            # Schedule buffers are pinned to fp32 (to preserve integer
            # timestep values under a stray `module.to(bf16)`), but the
            # network expects timesteps in the input dtype so that
            # downstream modulation / Linear layers stay consistent.
            timestep = timesteps[i].to(dtype=input_dtype)
            if i > 0:
                assert clean is not None
                noise = torch.empty_like(noisy).normal_(generator=rng)
                noisy = ((1.0 - sigma) * clean + sigma * noise).to(input_dtype)
            flow = predict_flow(noisy, timestep)
            clean = noisy - sigma * flow
        assert clean is not None, "denoising_step_list is empty"
        return clean.to(input_dtype)

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Apply the forward corruption at an arbitrary timestep.

        Snaps ``timestep`` to the nearest entry of the warped training table
        and uses it as sigma in the standard lerp.
        """
        assert timestep.shape == (), f"expected scalar timestep, got {timestep.shape}"
        full_t = self._full_timesteps
        idx = torch.argmin((full_t - timestep.to(full_t.dtype)).abs()).reshape(1)
        sigma = self._full_sigmas.index_select(0, idx).reshape(())
        noise = torch.empty_like(clean_input).normal_(generator=rng)
        return ((1.0 - sigma) * clean_input + sigma * noise).to(clean_input.dtype)
