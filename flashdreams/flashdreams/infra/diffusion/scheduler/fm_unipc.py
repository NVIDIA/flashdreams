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

"""Flow-matching UniPC multistep scheduler."""

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


def _build_per_step_coefs(sigmas: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-bake every per-step UniPC scalar from the sigma schedule.

    Args:
        sigmas: ``[N+1]`` fp32 inference sigma schedule with the
            mandatory trailing ``0.0`` (``final_sigmas_type="zero"``).

    Returns:
        Dict of ``[N]`` fp32 arrays, with the order pattern
        ``[1, 2, ..., 2, 1]`` already folded in:

        - ``a_pred`` -- predictor coef on ``x_i``.
        - ``b_pred_m0`` -- predictor coef on ``m_curr`` (current
          step's ``x0`` estimate).
        - ``b_pred_dprev`` -- predictor coef on ``m_prev - m_curr``
          (zero on the order-1 warmup ``i=0`` and order-1 final
          ``i=N-1``).
        - ``a_corr`` -- corrector coef on ``last_sample`` (zero at
          ``i=0`` since no corrector runs on the first step).
        - ``b_corr_m0`` -- corrector coef on ``m_prev``.
        - ``b_corr_dprev`` -- corrector coef on
          ``m_prev_prev - m_prev`` (zero at ``i in {0, 1}``).
        - ``b_corr_dt`` -- corrector coef on ``m_curr - m_prev``.
    """
    N = len(sigmas) - 1
    s = sigmas.astype(np.float64)  # do scalar math in fp64; downcast at end

    # Predictor at step i: x_{i+1} = a * x_i - alpha_t * h_phi_1 * (m0 + 0.5/rks_p * (m_prev - m0))
    # where:
    #   sigma_s0, sigma_t = s[i], s[i+1]
    #   alpha_t = 1 - sigma_t
    #   h_p     = log(1 - sigma_t) - log(sigma_t) - log(1 - sigma_s0) + log(sigma_s0)
    #   h_phi_1 = expm1(-h_p)
    # The expm1(-h) form gracefully handles the i = N-1 step where
    # sigma_t = 0 (h -> +inf, h_phi_1 -> -1, alpha_t = 1, final
    # x_t = m0).
    sigma_s0_p = s[:-1]
    sigma_t_p = s[1:]
    alpha_t_p = 1.0 - sigma_t_p
    a_pred = sigma_t_p / sigma_s0_p

    # Avoid log(0) at the last step (sigma_t_p[-1] == 0). The order-1
    # tail ignores h_phi_1 anyway via b_pred_dprev[-1] = 0, and the
    # m0 coefficient is -alpha_t * h_phi_1 = -1 * (-1) = 1 there.
    with np.errstate(divide="ignore"):
        log_alpha_t_p = np.log(np.maximum(alpha_t_p, 0.0))
        log_sigma_t_p = np.log(sigma_t_p)
    log_alpha_s0_p = np.log(1.0 - sigma_s0_p)
    log_sigma_s0_p = np.log(sigma_s0_p)
    h_p = (log_alpha_t_p - log_sigma_t_p) - (log_alpha_s0_p - log_sigma_s0_p)
    h_phi_1_p = np.expm1(-h_p)
    # Replace the last entry (sigma_t = 0 -> h = +inf -> -alpha_t*h_phi_1 = 1).
    b_pred_m0 = -alpha_t_p * h_phi_1_p
    b_pred_m0[-1] = 1.0  # closed-form limit at sigma_t = 0

    # rks_p[i] = (lambda(sigmas[i-1]) - lambda(sigmas[i])) / h_p[i],
    # only defined for i in [1, N-1); only used when the predictor is
    # order-2. (Step 0 is order-1 warmup, step N-1 is order-1 due to
    # lower_order_final; both leave b_pred_dprev = 0.)
    b_pred_dprev = np.zeros_like(b_pred_m0)
    if N >= 3:
        sigma_prev_p = s[:-3]  # i runs 1..N-2, prev is s[i-1] = s[0..N-3]
        log_alpha_prev_p = np.log(1.0 - sigma_prev_p)
        log_sigma_prev_p = np.log(sigma_prev_p)
        lambda_prev = log_alpha_prev_p - log_sigma_prev_p
        lambda_s0 = (log_alpha_s0_p - log_sigma_s0_p)[1:-1]  # i = 1..N-2
        rks_p = (lambda_prev - lambda_s0) / h_p[1:-1]
        b_pred_dprev[1:-1] = b_pred_m0[1:-1] * 0.5 / rks_p

    # Corrector at step i (i >= 1): operates on sigmas[i-1] -> sigmas[i].
    # For i = 0 we leave the corrector slot zeroed and skip the call
    # in sample().
    a_corr = np.zeros_like(a_pred)
    b_corr_m0 = np.zeros_like(a_pred)
    b_corr_dprev = np.zeros_like(a_pred)
    b_corr_dt = np.zeros_like(a_pred)
    if N >= 1:
        sigma_s0_c = s[:-2]  # corrector at step i uses s[i-1]; i runs 1..N-1
        sigma_t_c = s[1:-1]  # corrector at step i uses s[i]
        alpha_t_c = 1.0 - sigma_t_c
        a_corr[1:] = sigma_t_c / sigma_s0_c
        log_alpha_t_c = np.log(alpha_t_c)
        log_sigma_t_c = np.log(sigma_t_c)
        log_alpha_s0_c = np.log(1.0 - sigma_s0_c)
        log_sigma_s0_c = np.log(sigma_s0_c)
        h_c = (log_alpha_t_c - log_sigma_t_c) - (log_alpha_s0_c - log_sigma_s0_c)
        h_phi_1_c = np.expm1(-h_c)
        b_corr_m0[1:] = -alpha_t_c * h_phi_1_c

        # Order-1 corrector: only used at step 1 (the very first
        # corrector call, since step 0's predictor was order 1).
        # rhos_c = [0.5] hardcoded; b_corr_dprev[1] stays 0,
        # b_corr_dt[1] = -alpha_t * h_phi_1 * 0.5.
        b_corr_dt[1] = b_corr_m0[1] * 0.5

        # Order-2 corrector: steps 2..N-1. rhos_c is the solution of
        #   [[1, 1], [rks_c, 1]] @ rhos_c = b
        # with b derived from the same recurrence as upstream:
        #   B_h     = expm1(-h_c) = h_phi_1_c
        #   hh      = -h_c
        #   k0      = h_phi_1_c / hh - 1                    -> b[0] = k0      / B_h
        #   k1      = k0 / hh - 1/2                         -> b[1] = 2 * k1  / B_h
        # Closed-form solve of the 2x2 system:
        #   det = 1 - rks_c
        #   r0  = (b0 - b1) / det
        #   r1  = (b1 - rks_c * b0) / det
        if N >= 3:
            sigma_prev_c = s[:-3]  # rks uses s[i-2]; i runs 2..N-1
            log_alpha_prev_c = np.log(1.0 - sigma_prev_c)
            log_sigma_prev_c = np.log(sigma_prev_c)
            lambda_prev_c = log_alpha_prev_c - log_sigma_prev_c
            lambda_s0_c = (log_alpha_s0_c - log_sigma_s0_c)[1:]
            rks_c = (lambda_prev_c - lambda_s0_c) / h_c[1:]
            hh = -h_c[1:]
            B_h = h_phi_1_c[1:]
            k0 = h_phi_1_c[1:] / hh - 1.0
            k1 = k0 / hh - 0.5
            b0 = k0 / B_h
            b1 = 2.0 * k1 / B_h
            det = 1.0 - rks_c
            r0 = (b0 - b1) / det
            r1 = (b1 - rks_c * b0) / det
            # corr coefficients fold the leading -alpha_t * B_h.
            # Reference forms ``D1s = (m_prev_prev - m_prev) / rks_c``
            # and then ``corr_res = r0 * D1s[0]``, so the final m_pp-m_p
            # coefficient is ``-alpha_t * B_h * r0 / rks_c``.
            scale = -alpha_t_c[1:] * B_h
            b_corr_dprev[2:] = scale * r0 / rks_c
            b_corr_dt[2:] = scale * r1

    return {
        "a_pred": a_pred.astype(np.float32),
        "b_pred_m0": b_pred_m0.astype(np.float32),
        "b_pred_dprev": b_pred_dprev.astype(np.float32),
        "a_corr": a_corr.astype(np.float32),
        "b_corr_m0": b_corr_m0.astype(np.float32),
        "b_corr_dprev": b_corr_dprev.astype(np.float32),
        "b_corr_dt": b_corr_dt.astype(np.float32),
    }


@dataclass(kw_only=True)
class FlowMatchUniPCSchedulerConfig(SchedulerConfig):
    """Config for the flow-matching UniPC scheduler.

    Defaults match the official Wan 2.1 inference recipe (UniPC, BH2,
    order 2, shift 5.0). Override ``shift`` per checkpoint as recommended
    upstream (e.g. 3.0 for Wan 2.1 14B I2V 480P).
    """

    _target: type["FlowMatchUniPCScheduler"] = field(
        default_factory=lambda: FlowMatchUniPCScheduler
    )

    num_inference_steps: int = 50
    """Number of UniPC denoising steps."""

    shift: float = 5.0
    """Schedule warp factor."""

    num_train_timesteps: int = 1000
    """Length of the training sigma table."""

    solver_order: int = 2
    """UniPC solver order; only 2 is supported."""

    use_kerras_sigma: bool = False
    """Whether to use the exact sigma used in edm sampler."""

    enable_tqdm: bool = False
    """Whether to enable tqdm progress bar."""


class FlowMatchUniPCScheduler(Scheduler):
    """Order-2 UniPC predictor-corrector for flow-matching.

    Specialized + pre-baked variant of the upstream Wan 2.1 UniPC solver.
    Schedule buffers (sigmas + per-step coefficients) stay fp32 regardless
    of ``module.to(dtype)``.

    Examples:

        scheduler = FlowMatchUniPCSchedulerConfig(
            num_inference_steps=50, shift=5.0,
        ).setup().to("cuda")
        clean = scheduler.sample(initial_noise=noise, predict_flow=fn)
    """

    timesteps: Tensor
    sigmas: Tensor
    _sigmas_full: Tensor
    a_pred: Tensor
    b_pred_m0: Tensor
    b_pred_dprev: Tensor
    a_corr: Tensor
    b_corr_m0: Tensor
    b_corr_dprev: Tensor
    b_corr_dt: Tensor

    def __init__(self, config: FlowMatchUniPCSchedulerConfig) -> None:
        super().__init__(config)
        self.config: FlowMatchUniPCSchedulerConfig = config
        assert config.solver_order == 2, (
            f"Only solver_order=2 is supported (got {config.solver_order}); "
            "the slim UniPC bakes coefficients for the order-1/order-2 "
            "warmup+steady-state pattern. Use the legacy scheduler if you "
            "need other orders."
        )
        N = config.num_inference_steps
        N_train = config.num_train_timesteps

        if self.config.use_kerras_sigma:
            # force to use the exact sigma used in edm sampler
            sigma_max = 200
            sigma_min = 0.01
            rho = 7
            sigmas = np.arange(N + 1) / N
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + sigmas * (min_inv_rho - max_inv_rho)) ** rho
            sigmas_inf = sigmas / (1 + sigmas)
        else:
            # Schedule build: identical to upstream
            # ``set_timesteps(num_inference_steps, shift=config.shift)``:
            #
            #   sigmas = linspace(sigma_max, sigma_min, N + 1)[:-1]
            #   sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            #   sigmas = concat(sigmas, [0.0])
            #
            # where (sigma_min, sigma_max) come from the
            # ``alphas = linspace(1, 1/num_train_timesteps, num_train_timesteps)[::-1]``
            # / ``sigmas = 1 - alphas`` table built at construction.
            # Training-time sigma table: ``alphas = linspace(1, 1/N, N)[::-1]``
            # then ``sigmas = 1 - alphas``. Reference builds this with
            # ``shift=1.0`` (no warp); only the inference schedule below
            # gets warped with ``config.shift``.
            train_alphas = np.linspace(1.0, 1.0 / N_train, N_train)[::-1].copy()
            train_sigmas = 1.0 - train_alphas
            sigma_min, sigma_max = float(train_sigmas[-1]), float(train_sigmas[0])
            sigmas_inf = np.linspace(sigma_max, sigma_min, N + 1)[:-1]
            sigmas_inf = (
                config.shift * sigmas_inf / (1.0 + (config.shift - 1.0) * sigmas_inf)
            )

        # Match reference: timesteps come from the fp64 ``sigmas * N``
        # cast directly to int64 (truncation), not via fp32. The fp32
        # round-trip can shift entries by 1 LSB and break the
        # exact-match index lookup in ``add_noise``.
        timesteps_fp64 = sigmas_inf * N_train
        sigmas_full = np.concatenate([sigmas_inf, [0.0]]).astype(np.float32)

        coefs = _build_per_step_coefs(sigmas_full)

        # Buffers move with .to(device) but the float ones are pinned
        # to fp32 by the ``_apply`` override below -- a stray
        # ``model.to(bf16)`` would otherwise quantize the snap table
        # and the per-step coefficients. ``timesteps`` stays int64
        # (PyTorch already skips int casts in ``module.to(dtype)``).
        self.register_buffer(
            "timesteps",
            torch.from_numpy(timesteps_fp64).to(torch.int64),
            persistent=False,
        )
        # ``sigmas[i]`` is the noise level at step i (used for convert
        # = sample - sigma * model_output).
        self.register_buffer(
            "sigmas",
            torch.from_numpy(sigmas_full[:-1].copy()),
            persistent=False,
        )
        self.register_buffer(
            "_sigmas_full",
            torch.from_numpy(sigmas_full),
            persistent=False,
        )
        for k, v in coefs.items():
            self.register_buffer(k, torch.from_numpy(v), persistent=False)
        self._FP32_BUFFERS = ("sigmas", "_sigmas_full", *coefs.keys())

    def _apply(self, fn, recurse=True):
        """Move buffers with the parent ``.to(...)`` but keep them fp32.

        ``fn`` may be a lossy bf16 cast; snapshot the fp32 originals before
        ``super()._apply`` and restore them with a pure device move after.
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
        """Run the order-2 UniPC predictor-corrector denoising loop.

        Each iteration: network → flow → ``x0`` → corrector (skipped at
        step 0) → predictor. All per-step coefficients are pre-baked at
        construction; the loop is pure tensor ops. Internal arithmetic is
        fp32; the result is cast back to ``initial_noise.dtype``. ``rng``
        is unused (deterministic ODE) but accepted for interface conformance.
        """
        input_dtype = initial_noise.dtype
        N = self.timesteps.shape[0]

        sample = initial_noise
        m_prev: Tensor | None = None
        m_prev_prev: Tensor | None = None
        last_sample: Tensor | None = None

        for i in tqdm(
            range(N),
            disable=not self.config.enable_tqdm,
            desc="FlowMatchUniPCScheduler",
        ):
            # Schedule buffers are pinned to fp32 (to preserve integer
            # timestep values under a stray `module.to(bf16)`), but the
            # network expects timesteps in the input dtype so that
            # downstream modulation / Linear layers stay consistent.
            timestep = self.timesteps[i].to(dtype=input_dtype)

            # Network forward (heavy compute -- everything else here is
            # ~free relative to this).
            flow = predict_flow(sample, timestep)

            # Convert to x0 estimate (predict_x0 + flow_prediction):
            #   m_curr = sample - sigma_i * flow
            # Promote to fp32 to match upstream's
            # ``model_output = model_output.to(dtype=torch.float32)``
            # before convert_model_output.
            m_curr = sample.to(torch.float32) - self.sigmas[i] * flow.to(torch.float32)

            # Corrector (skip on first step).
            #
            # last_sample is the sample saved before the previous
            # predictor (matches upstream ``self.last_sample``).
            # m_prev is model_outputs[-1] (previous step's m_curr),
            # m_prev_prev is model_outputs[-2] (2 steps back).
            # b_corr_dprev[1] = 0 (order-1 corrector) so the
            # m_prev_prev term vanishes at i=1; we alias it to m_prev
            # there to skip the zero-tensor allocation.
            if i >= 1:
                assert last_sample is not None and m_prev is not None
                m_pp = m_prev_prev if m_prev_prev is not None else m_prev
                corrected = (
                    self.a_corr[i] * last_sample.to(torch.float32)
                    + self.b_corr_m0[i] * m_prev
                    + self.b_corr_dprev[i] * (m_pp - m_prev)
                    + self.b_corr_dt[i] * (m_curr - m_prev)
                )
                sample = corrected.to(input_dtype)

            # Save sample BEFORE the predictor so the next iteration's
            # corrector can use it as "last_sample".
            last_sample = sample

            # Predictor advances sample to next sigma:
            #   x_{i+1} = a_pred[i] * x_i + b_pred_m0[i] * m0
            #            + b_pred_dprev[i] * (m_prev - m0)
            # b_pred_dprev[0] = 0 (order-1 warmup) and
            # b_pred_dprev[-1] = 0 (order-1 final) so the same line
            # serves both order branches; alias m_prev to m_curr at
            # the warmup step to skip a zero-tensor allocation.
            m_p = m_prev if m_prev is not None else m_curr
            predicted = (
                self.a_pred[i] * sample.to(torch.float32)
                + self.b_pred_m0[i] * m_curr
                + self.b_pred_dprev[i] * (m_p - m_curr)
            )
            sample = predicted.to(input_dtype)

            # Roll the model-output history.
            m_prev_prev = m_prev
            m_prev = m_curr

        return sample

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Apply the forward corruption at an arbitrary timestep.

        Snaps ``timestep`` to the nearest entry of the inference schedule
        on-device (no Python sync) and uses the matching sigma in the lerp.
        """
        assert timestep.shape == (), f"expected scalar timestep, got {timestep.shape}"
        ts = self.timesteps
        idx = torch.argmin((ts - timestep.to(ts.dtype)).abs()).reshape(1)
        sigma = self._sigmas_full.index_select(0, idx).reshape(())
        noise = torch.empty_like(clean_input).normal_(generator=rng)
        return ((1.0 - sigma) * clean_input + sigma * noise).to(clean_input.dtype)
