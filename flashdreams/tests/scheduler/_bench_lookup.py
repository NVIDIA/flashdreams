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

"""Bench: argmin/argmax vs searchsorted for a single scalar lookup.

Mirrors what `fm.add_noise` (1000-entry, nearest) and
`fm_unipc.add_noise` (50-entry, exact) actually do -- a single scalar
query, not a batch. Compares against both:

- the diffusers-style ``[(t == s).nonzero() for s in ts]`` baseline,
- the cleaner ``searchsorted`` form the user proposed.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


def timeit(repeats: int, f: Callable, *args, **kwargs) -> tuple[float, Any]:
    for _ in range(10):  # warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        results = f(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1e6 / repeats, results  # us / call


# ---------------------------------------------------------------------------
# Scenario A: fm_unipc.add_noise -- exact-match, 50-entry descending int64
# ---------------------------------------------------------------------------
def unipc_argmax_eq(timesteps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Old fm_unipc impl: equality scan + argmax (exact-match only)."""
    return (timesteps == t).to(torch.int8).argmax().reshape(1)


def unipc_argmin_abs(timesteps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """New fm_unipc impl: argmin of abs-diff (nearest-match)."""
    return torch.argmin((timesteps - t).abs()).reshape(1)


def unipc_searchsorted(asc_timesteps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Searchsorted on flipped (ascending) array (exact-match only)."""
    return torch.searchsorted(asc_timesteps, t.reshape(1))


def unipc_nonzero_loop(timesteps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Diffusers-style: per-element nonzero in a Python loop."""
    return torch.cat([(timesteps == ti).nonzero() for ti in t.reshape(1)])


# ---------------------------------------------------------------------------
# Scenario B: fm.add_noise -- nearest-neighbor, 1000-entry descending fp32
# ---------------------------------------------------------------------------
def fm_argmin_abs(full_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Current fm impl: argmin of abs-diff."""
    return torch.argmin((full_t - t).abs()).reshape(1)


def fm_searchsorted_nearest(asc_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Searchsorted + pick-nearest neighbor."""
    N = asc_t.shape[0]
    i = torch.searchsorted(asc_t, t.reshape(1)).clamp(1, N - 1)
    lo = asc_t.index_select(0, i - 1)
    hi = asc_t.index_select(0, i)
    return torch.where((t - lo).abs() < (hi - t).abs(), i - 1, i)


# ---------------------------------------------------------------------------
def main() -> None:
    device = "cuda"
    REPS = 2000

    print("=" * 72)
    print("Scenario A: fm_unipc.add_noise (50 entries, exact-match)")
    print("=" * 72)

    # 50-entry descending int64 schedule (mimics our self.timesteps)
    ts_desc = torch.linspace(999.0, 20.0, 50, device=device).round().to(torch.int64)
    ts_asc = ts_desc.flip(0)
    # pick a real schedule entry
    query = ts_desc[17].clone()

    t_argmax, _ = timeit(REPS, unipc_argmax_eq, ts_desc, query)
    t_argmin, _ = timeit(REPS, unipc_argmin_abs, ts_desc, query)
    t_search, _ = timeit(REPS, unipc_searchsorted, ts_asc, query)
    t_nonzero, _ = timeit(REPS, unipc_nonzero_loop, ts_desc, query)
    print(f"  argmax(==)   [old]     : {t_argmax:7.2f} us / call (exact)")
    print(f"  argmin(abs)  [new]     : {t_argmin:7.2f} us / call (nearest)")
    print(f"  searchsorted           : {t_search:7.2f} us / call (exact)")
    print(f"  nonzero in py loop     : {t_nonzero:7.2f} us / call  <- diffusers")

    # sanity check
    a = unipc_argmax_eq(ts_desc, query).item()
    b = unipc_searchsorted(ts_asc, query).item()
    # convert searchsorted-on-asc index back to the desc index
    b_desc = ts_desc.shape[0] - 1 - b
    assert a == b_desc, f"mismatch: argmax={a} searchsorted-translated={b_desc}"
    print(f"  (sanity: idx_desc={a}, search_asc={b} -> desc={b_desc}) OK")

    print()
    print("=" * 72)
    print("Scenario B: fm.add_noise (1000 entries, nearest-neighbor, fp32)")
    print("=" * 72)

    # 1000-entry descending fp32 (mimics our _full_timesteps after warp)
    sigmas = torch.linspace(1.0, 0.0, 1001, device=device, dtype=torch.float32)[:-1]
    shift = 8.0
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    full_t_desc = sigmas * 1000.0
    full_t_asc = full_t_desc.flip(0)

    # Off-schedule query (matches the real `context_noise` use case)
    query_f = torch.tensor(347.5, device=device, dtype=torch.float32)

    t_argmin, _ = timeit(REPS, fm_argmin_abs, full_t_desc, query_f)
    t_search, _ = timeit(REPS, fm_searchsorted_nearest, full_t_asc, query_f)
    print(f"  argmin(abs)            : {t_argmin:7.2f} us / call")
    print(f"  searchsorted+nearest   : {t_search:7.2f} us / call")

    # sanity
    a = fm_argmin_abs(full_t_desc, query_f).item()
    b = fm_searchsorted_nearest(full_t_asc, query_f).item()
    b_desc = full_t_desc.shape[0] - 1 - b
    print(
        f"  (sanity: argmin={a}, search_asc={b} -> desc={b_desc}; "
        f"vals={full_t_desc[a].item():.2f} vs {full_t_desc[b_desc].item():.2f})"  # ty:ignore[invalid-argument-type]
    )


if __name__ == "__main__":
    main()
