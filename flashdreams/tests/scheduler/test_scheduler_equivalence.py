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

"""Numerical equivalence between reference and slim diffusion schedulers.

Compares the frozen reference wrappers in :mod:`impl_reference_flow_match`
and :mod:`impl_reference_flow_unipc` against the rewrites in
:mod:`flashdreams.infra.diffusion.scheduler` on representative
``sample()`` and ``add_noise()`` calls.

Runs on CPU and (when available) GPU. The stub flow predictor is a
deterministic affine map of the input, so any divergence isolates a
solver bug rather than predictor noise.
"""

from __future__ import annotations

from collections.abc import Callable

# Sibling modules; ``conftest.py`` adds this directory to ``sys.path``.
import impl_reference_flow_match as _ref_fm  # noqa: E402
import impl_reference_flow_unipc as _ref_unipc  # noqa: E402
import pytest
import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowMatchScheduler,
    FlowMatchSchedulerConfig,
    FlowMatchUniPCScheduler,
    FlowMatchUniPCSchedulerConfig,
)

# ---------------------------------------------------------------------------
# Stub flow predictors. Pure tensor math, no module: keeps the parity
# test focused on the solver. ``timestep`` is folded into the result so
# that any silently-wrong dtype/scalar passed by a scheduler shows up.
# ---------------------------------------------------------------------------


def _stub_predict_flow_factory(
    scale: float, bias: float
) -> Callable[[Tensor, Tensor], Tensor]:
    def _predict_flow(noisy_latent: Tensor, timestep: Tensor) -> Tensor:
        t = timestep.to(noisy_latent.dtype) / 1000.0
        return scale * noisy_latent + bias + t

    return _predict_flow


def _devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


# ---------------------------------------------------------------------------
# FlowMatch
# ---------------------------------------------------------------------------


_FM_DENOISING = [1000, 750, 500, 250]
_FM_SHIFT = 8.0


def _build_fm_pair() -> tuple[_ref_fm.FlowMatchSchedulerReference, FlowMatchScheduler]:
    ref_cfg = _ref_fm.FlowMatchReferenceConfig(
        num_inference_steps=len(_FM_DENOISING),
        shift=_FM_SHIFT,
        denoising_timesteps=list(_FM_DENOISING),
    )
    new_cfg = FlowMatchSchedulerConfig(
        num_inference_steps=len(_FM_DENOISING),
        shift=_FM_SHIFT,
        denoising_timesteps=list(_FM_DENOISING),
    )
    return _ref_fm.FlowMatchSchedulerReference(ref_cfg), new_cfg.setup()  # ty:ignore[invalid-return-type]


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-6, 1e-6),
        # bf16 tolerance is loose because legacy runs argmin against a
        # bf16-cast timestep (e.g. 888.89 -> 888.0) and can pick a
        # neighboring schedule index, while the rewrite snaps in fp32
        # at construction. The legacy drift is bf16 quantization noise
        # of the schedule index; both produce sigmas within bf16 epsilon
        # of the warped formula, so the loop output differs by O(bf16).
        (torch.bfloat16, 2e-2, 1e-2),
    ],
    ids=["fp32", "bf16"],
)
@torch.no_grad()
def test_flow_match_sample_parity(
    device: torch.device, dtype: torch.dtype, atol: float, rtol: float
) -> None:
    if dtype == torch.bfloat16 and device.type == "cpu":
        pytest.skip("bf16 add_noise on CPU is unstable for this test")
    ref, new = _build_fm_pair()
    ref.to(device=device)
    new.to(device=device)

    torch.manual_seed(0)
    noise = torch.empty(2, 4, 8, 16, 16, dtype=dtype, device=device).uniform_(-1, 1)
    predict_flow = _stub_predict_flow_factory(0.7, 0.1)

    rng_ref = torch.Generator(device=device).manual_seed(123)
    rng_new = torch.Generator(device=device).manual_seed(123)
    out_ref = ref.sample(noise, predict_flow, rng=rng_ref)
    out_new = new.sample(noise, predict_flow, rng=rng_new)  # ty:ignore[invalid-argument-type]
    torch.testing.assert_close(out_new, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-6, 1e-6),
        # bf16: legacy mixes a bf16 sigma into the lerp; the new code
        # keeps sigma in fp32 and casts the result, which is more
        # accurate but differs by ~1 bf16 ulp on the lerp output.
        (torch.bfloat16, 1e-2, 1e-2),
    ],
    ids=["fp32", "bf16"],
)
@torch.no_grad()
def test_flow_match_add_noise_parity(
    device: torch.device, dtype: torch.dtype, atol: float, rtol: float
) -> None:
    if dtype == torch.bfloat16 and device.type == "cpu":
        pytest.skip("bf16 add_noise on CPU is unstable for this test")
    ref, new = _build_fm_pair()
    ref.to(device=device)
    new.to(device=device)

    torch.manual_seed(1)
    clean = torch.empty(2, 4, 8, 16, 16, dtype=dtype, device=device).uniform_(-1, 1)
    for t_value in (1000.0, 750.0, 500.0, 250.0, 128.0):
        timestep = torch.tensor(t_value, dtype=dtype, device=device)
        rng_ref = torch.Generator(device=device).manual_seed(42)
        rng_new = torch.Generator(device=device).manual_seed(42)
        ref_out = ref.add_noise(clean, timestep, rng=rng_ref)
        new_out = new.add_noise(clean, timestep, rng=rng_new)
        torch.testing.assert_close(
            new_out,
            ref_out,
            atol=atol,
            rtol=rtol,
            msg=lambda m, t=t_value: f"add_noise(t={t}): {m}",
        )


# ---------------------------------------------------------------------------
# FlowUniPC
# ---------------------------------------------------------------------------


_UNIPC_STEPS = 50
_UNIPC_SHIFT = 5.0
_UNIPC_ORDER = 2


def _build_unipc_pair() -> tuple[
    _ref_unipc.FlowUniPCSchedulerReference, FlowMatchUniPCScheduler
]:
    ref_cfg = _ref_unipc.FlowUniPCReferenceConfig(
        num_inference_steps=_UNIPC_STEPS,
        shift=_UNIPC_SHIFT,
        solver_order=_UNIPC_ORDER,
    )
    new_cfg = FlowMatchUniPCSchedulerConfig(
        num_inference_steps=_UNIPC_STEPS,
        shift=_UNIPC_SHIFT,
        solver_order=_UNIPC_ORDER,
    )
    return _ref_unipc.FlowUniPCSchedulerReference(ref_cfg), new_cfg.setup()  # ty:ignore[invalid-return-type]


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        # bf16 atol is loose (~bf16 epsilon * sqrt(num_steps)) because
        # the rewrite folds a couple of legacy multiply-adds per step
        # into a single fp32 sum and casts to bf16 once, which shifts
        # the rounding pattern slightly. Both paths exit with values in
        # the same neighborhood; fp32 is bit-exact (above) so the math
        # is verified.
        (torch.bfloat16, 5e-2, 5e-2),
    ],
    ids=["fp32", "bf16"],
)
@torch.no_grad()
def test_flow_unipc_sample_parity(
    device: torch.device, dtype: torch.dtype, atol: float, rtol: float
) -> None:
    ref, new = _build_unipc_pair()
    ref.to(device=device)
    new.to(device=device)

    torch.manual_seed(0)
    noise = torch.empty(2, 4, 8, 16, 16, dtype=dtype, device=device).uniform_(-1, 1)
    predict_flow = _stub_predict_flow_factory(0.7, 0.1)

    out_ref = ref.sample(noise, predict_flow)
    out_new = new.sample(noise, predict_flow)  # ty:ignore[invalid-argument-type]
    torch.testing.assert_close(out_new, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 5e-3, 5e-3),
    ],
    ids=["fp32", "bf16"],
)
@torch.no_grad()
def test_flow_unipc_add_noise_parity(
    device: torch.device, dtype: torch.dtype, atol: float, rtol: float
) -> None:
    if dtype == torch.bfloat16:
        # Schedule timesteps (e.g. 999) round-trip through bf16 to
        # 1000.0; the new scheduler snaps that back to 999 (nearest)
        # but the upstream reference uses an exact-match ``nonzero``
        # lookup and IndexError's. Skip parity here -- the new
        # behaviour is strictly better.
        pytest.skip("reference exact-match crashes on bf16-rounded timesteps")
    ref, new = _build_unipc_pair()
    ref.to(device=device)
    new.to(device=device)
    # add_noise indexes a schedule built lazily on the first sample()
    # call; trigger the build with a no-op sample so reference and
    # rewrite compare on the same schedule shape.
    dummy = torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device)
    ref.sample(dummy, _stub_predict_flow_factory(0.0, 0.0))
    new.sample(dummy, _stub_predict_flow_factory(0.0, 0.0))  # ty:ignore[invalid-argument-type]

    torch.manual_seed(1)
    clean = torch.empty(2, 4, 8, 16, 16, dtype=dtype, device=device).uniform_(-1, 1)
    # On-schedule timesteps -- nearest-match and reference exact-match
    # agree exactly here.
    schedule_ts = ref._fm.timesteps.tolist()
    for t_int in (schedule_ts[0], schedule_ts[len(schedule_ts) // 2], schedule_ts[-1]):
        timestep = torch.tensor(t_int, dtype=dtype, device=device)
        rng_ref = torch.Generator(device=device).manual_seed(42)
        rng_new = torch.Generator(device=device).manual_seed(42)
        ref_out = ref.add_noise(clean, timestep, rng=rng_ref)
        new_out = new.add_noise(clean, timestep, rng=rng_new)
        torch.testing.assert_close(
            new_out,
            ref_out,
            atol=atol,
            rtol=rtol,
            msg=lambda m, t=t_int: f"add_noise(t={t}): {m}",
        )
