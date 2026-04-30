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

"""Check that scheduler ``sample()`` / ``add_noise()`` don't sync.

We use ``torch.cuda.set_sync_debug_mode("error")`` so any sync-causing
op raises ``RuntimeError`` instead of silently stalling. Run on a GPU::

    PYTHONPATH=./flashdreams python flashdreams/tests/scheduler/_check_no_sync.py
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashdreams.infra.diffusion.scheduler import (
    FlowMatchSchedulerConfig,
    FlowMatchUniPCSchedulerConfig,
)


def _stub(noisy: Tensor, timestep: Tensor) -> Tensor:
    return noisy * 0.7


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    fm = (
        FlowMatchSchedulerConfig(
            num_inference_steps=4,
            shift=8.0,
            denoising_timesteps=[1000, 750, 500, 250],
        )
        .setup()
        .to(device)
    )
    unipc = (
        FlowMatchUniPCSchedulerConfig(num_inference_steps=50, shift=5.0, solver_order=2)
        .setup()
        .to(device)
    )

    noise = torch.randn(1, 16, 21, 90, 160, dtype=dtype, device=device)
    clean = torch.randn(1, 16, 21, 90, 160, dtype=dtype, device=device)
    # Build timesteps on the host then async-copy so the assertion
    # below only catches sync ops *inside* the schedulers (not the
    # one-shot scalar construction in the test driver).
    fm_t = torch.empty((), dtype=torch.int64, device=device)
    fm_t.fill_(128)
    unipc_t = unipc.timesteps[0]  # ty:ignore[not-subscriptable]

    # warm-up (allocators, autograd state, etc.) without sync mode on
    fm.sample(noise, _stub)  # ty:ignore[invalid-argument-type]
    unipc.sample(noise, _stub)  # ty:ignore[invalid-argument-type]
    fm.add_noise(clean, fm_t)
    unipc.add_noise(clean, unipc_t)
    torch.cuda.synchronize()

    # Now turn sync detection on. ``"error"`` raises on any sync op.
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = fm.sample(noise, _stub)  # ty:ignore[invalid-argument-type]
        print(f"  FlowMatch.sample      -> ok ({tuple(out.shape)} {out.dtype})")
        out = unipc.sample(noise, _stub)  # ty:ignore[invalid-argument-type]
        print(f"  FlowUniPC.sample      -> ok ({tuple(out.shape)} {out.dtype})")
        out = fm.add_noise(clean, fm_t)
        print(f"  FlowMatch.add_noise   -> ok ({tuple(out.shape)} {out.dtype})")
        out = unipc.add_noise(clean, unipc_t)
        print(f"  FlowUniPC.add_noise   -> ok ({tuple(out.shape)} {out.dtype})")
    finally:
        torch.cuda.set_sync_debug_mode("default")

    print("\nNo CPU<->GPU sync detected in any scheduler call.")


if __name__ == "__main__":
    main()
