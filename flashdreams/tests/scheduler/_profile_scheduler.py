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

"""Compare per-step latency: reference vs slim diffusion schedulers.

Run on a GPU (typically inside an interactive srun)::

    PYTHONPATH=./flashdreams python flashdreams/tests/scheduler/_profile_scheduler.py
    PYTHONPATH=./flashdreams python flashdreams/tests/scheduler/_profile_scheduler.py --schedulers flow_match
    PYTHONPATH=./flashdreams python flashdreams/tests/scheduler/_profile_scheduler.py --n-repeat 50

The schedulers contain no learned weights -- their per-step cost is
host-side Python plus a handful of pointwise CUDA ops on the latent.
The stub flow predictor is a single pointwise op so what we measure
is dominated by the solver itself, which is the thing the cleanup is
trying to make smaller.

We report total ``sample()`` ms and per-step ms (``total / num_steps``)
for both schedulers in both fp32 and bf16.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowMatchScheduler,
    FlowMatchSchedulerConfig,
    FlowMatchUniPCScheduler,
    FlowMatchUniPCSchedulerConfig,
)

# Sibling modules: when run as a script (not via pytest), `conftest.py`
# is not loaded, so add this directory to `sys.path` ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from impl_reference_flow_match import (  # noqa: E402
    FlowMatchReferenceConfig,
    FlowMatchSchedulerReference,
)
from impl_reference_flow_unipc import (  # noqa: E402
    FlowUniPCReferenceConfig,
    FlowUniPCSchedulerReference,
)

_FM_DENOISING = [1000, 750, 500, 250]
_FM_SHIFT = 8.0

_UNIPC_STEPS = 50
_UNIPC_SHIFT = 5.0
_UNIPC_ORDER = 2


def _log(msg: str, t0: float | None = None) -> float:
    t = time.perf_counter()
    prefix = f"[{t - t0:6.2f}s] " if t0 is not None else ""
    print(f"  {prefix}{msg}", flush=True)
    return t


def _build_fm_pair() -> tuple[FlowMatchSchedulerReference, FlowMatchScheduler]:
    ref = FlowMatchSchedulerReference(
        FlowMatchReferenceConfig(
            num_inference_steps=len(_FM_DENOISING),
            shift=_FM_SHIFT,
            denoising_timesteps=list(_FM_DENOISING),
        )
    )
    new = FlowMatchSchedulerConfig(
        num_inference_steps=len(_FM_DENOISING),
        shift=_FM_SHIFT,
        denoising_timesteps=list(_FM_DENOISING),
    ).setup()
    return ref, new  # ty:ignore[invalid-return-type]


def _build_unipc_pair() -> tuple[FlowUniPCSchedulerReference, FlowMatchUniPCScheduler]:
    ref = FlowUniPCSchedulerReference(
        FlowUniPCReferenceConfig(
            num_inference_steps=_UNIPC_STEPS,
            shift=_UNIPC_SHIFT,
            solver_order=_UNIPC_ORDER,
        )
    )
    new = FlowMatchUniPCSchedulerConfig(
        num_inference_steps=_UNIPC_STEPS,
        shift=_UNIPC_SHIFT,
        solver_order=_UNIPC_ORDER,
    ).setup()
    return ref, new  # ty:ignore[invalid-return-type]


def _make_stub_predict_flow():
    """Stub predictor: single pointwise op so the solver dominates."""

    def _predict_flow(noisy: Tensor, timestep: Tensor) -> Tensor:
        return noisy * 0.7

    return _predict_flow


@torch.no_grad()
def _time_sample(
    scheduler: torch.nn.Module,
    *,
    noise: Tensor,
    n_repeat: int,
    n_warmup: int,
) -> float:
    predict_flow = _make_stub_predict_flow()

    for _ in range(n_warmup):
        scheduler.sample(noise, predict_flow)  # ty:ignore[call-non-callable]
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_repeat):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        scheduler.sample(noise, predict_flow)  # ty:ignore[call-non-callable]
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    return sum(times_ms[:-1]) / max(1, len(times_ms) - 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--schedulers",
        nargs="+",
        choices=["flow_match", "flow_unipc"],
        default=["flow_match", "flow_unipc"],
    )
    p.add_argument(
        "--dtypes",
        nargs="+",
        choices=["fp32", "bf16"],
        default=["bf16"],
        help="Latent dtype(s). Per-step solver cost barely changes with dtype.",
    )
    p.add_argument("--n-repeat", type=int, default=20)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument(
        "--shape",
        nargs=5,
        type=int,
        default=[1, 16, 21, 90, 160],
        metavar=("B", "C", "T", "H", "W"),
        help="Latent tensor shape (defaults match Wan 2.1 720p, 21 latent frames).",
    )
    return p.parse_args()


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda")

    print("=" * 78)
    print(
        f"Diffusion scheduler latency: legacy vs slim "
        f"(device={device}, shape={tuple(args.shape)})"
    )
    print(f"  schedulers = {args.schedulers}")
    print(f"  dtypes     = {args.dtypes}")
    print(f"  n_repeat   = {args.n_repeat} (warmup {args.n_warmup})")
    print("=" * 78, flush=True)

    rows: list[tuple[str, str, int, float, float]] = []

    for sched_key in args.schedulers:
        if sched_key == "flow_match":
            ref, new = _build_fm_pair()
            num_steps = len(_FM_DENOISING)
        else:
            ref, new = _build_unipc_pair()
            num_steps = _UNIPC_STEPS
        ref.to(device=device)
        new.to(device=device)

        for dtype_key in args.dtypes:
            dtype = _DTYPES[dtype_key]
            torch.manual_seed(0)
            noise = torch.empty(*args.shape, dtype=dtype, device=device).uniform_(-1, 1)

            t0 = time.perf_counter()
            print(f"\n[{sched_key} / {dtype_key}]", flush=True)
            t_leg = _time_sample(
                ref, noise=noise, n_repeat=args.n_repeat, n_warmup=args.n_warmup
            )
            _log(f"legacy total: {t_leg:.3f} ms ({t_leg / num_steps:.3f} ms/step)", t0)

            t = time.perf_counter()
            t_new = _time_sample(
                new, noise=noise, n_repeat=args.n_repeat, n_warmup=args.n_warmup
            )
            _log(f"new total:    {t_new:.3f} ms ({t_new / num_steps:.3f} ms/step)", t)
            rows.append((sched_key, dtype_key, num_steps, t_leg, t_new))

        del ref, new

    print()
    hdr = (
        f"{'scheduler':<12} {'dtype':<6} {'#steps':>6} "
        f"{'legacy ms':>11} {'new ms':>10} {'speedup':>9} "
        f"{'legacy us/step':>15} {'new us/step':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for sched_key, dtype_key, n_steps, leg, new_t in rows:
        speedup = leg / new_t if new_t > 0 else float("inf")
        print(
            f"{sched_key:<12} {dtype_key:<6} {n_steps:>6} "
            f"{leg:>11.3f} {new_t:>10.3f} {speedup:>8.2f}x "
            f"{leg * 1000 / n_steps:>15.1f} {new_t * 1000 / n_steps:>13.1f}"
        )


if __name__ == "__main__":
    main()
