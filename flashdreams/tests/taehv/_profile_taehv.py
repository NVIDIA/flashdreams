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

"""Compare 2nd-rollout latency: reference vs slim TAEHV decode.

Run on a GPU (typically inside an interactive srun)::

    PYTHONPATH=./flashdreams python flashdreams/tests/taehv/_profile_taehv.py
    PYTHONPATH=./flashdreams python flashdreams/tests/taehv/_profile_taehv.py --quick

The first chunk of every decode populates the streaming cache and
allocates workspaces; we measure the *second* chunk (and only the
second) to get a stable, cache-warm number that mirrors steady-state
rollout.

Notes on first-run cost: with ``use_compile=True`` (the default),
Inductor autotunes ``max-autotune-no-cudagraphs`` for each unique input
shape. TAEHV is much smaller than Wan VAE so first run is typically
~1-3 min. Subsequent runs hit the on-disk Inductor + triton caches.
Pass ``--quick`` to skip compilation entirely (CUDA-graph capture only).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Park Inductor + triton caches on local disk before torch is imported, so
# repeat runs of this script reuse autotune decisions.
_CACHE_ROOT = f"/tmp/{os.environ.get('USER', 'flashdreams')}/taehv_profile"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{_CACHE_ROOT}/inductor")
os.environ.setdefault("TRITON_CACHE_DIR", f"{_CACHE_ROOT}/triton")

import torch  # noqa: E402

from flashdreams.core.checkpoint.load import load_checkpoint  # noqa: E402
from flashdreams.recipes.taehv import (  # noqa: E402
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
)
from flashdreams.recipes.taehv.impl import TAEHV as TAEHVNew  # noqa: E402

# Sibling module: this script is meant to be run as `python file.py` (its
# own process), so a simple sys.path hack is sufficient. The package-form
# relative import used by `test_taehv_equivalence.py` doesn't apply here
# because `__name__ == "__main__"` and there is no parent package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import impl_reference as _impl_reference  # noqa: E402  # ty:ignore[unresolved-import]

TAEHVLegacy = _impl_reference.TAEHV


def _build_pair(
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_compile: bool,
) -> tuple[TAEHVLegacy, TAEHVNew]:
    import copy

    weights = load_checkpoint(checkpoint_path)
    weights = {k: v.to(dtype) for k, v in weights.items()}  # ty:ignore[call-non-callable]

    # Return a fresh shallow copy each call: the legacy ``patch_tgrow_layers``
    # mutates the dict in place, which would otherwise feed the new impl
    # already-truncated TGrow weights when the legacy ctor ran first.
    def _cached(_path):
        return copy.copy(weights)

    with (
        patch.object(_impl_reference, "load_checkpoint", _cached),
        patch("flashdreams.recipes.taehv.impl.load_checkpoint", _cached),
    ):
        legacy = TAEHVLegacy(checkpoint_path=checkpoint_path).to(
            device=device, dtype=dtype
        )
        new = TAEHVNew(checkpoint_path=checkpoint_path, use_compile=use_compile).to(
            device=device, dtype=dtype
        )
    return legacy, new


def _log(msg: str, t0: float | None = None) -> float:
    t = time.perf_counter()
    prefix = f"[{t - t0:6.1f}s] " if t0 is not None else ""
    print(f"  {prefix}{msg}", flush=True)
    return t


@torch.no_grad()
def _time_second_decode_legacy(
    model: TAEHVLegacy, latents: torch.Tensor, *, chunk_t: int, n_repeat: int
) -> float:
    """Warm the cache, then time ``n_repeat`` chunk-B decodes (legacy)."""
    z_a = latents[:, :chunk_t]
    z_b = latents[:, chunk_t : 2 * chunk_t]

    cache = model.prepare_cache()
    model.decode_video(z_a, parallel=True, cache=cache)
    # Legacy has no CUDA graph; one warm chunk is enough.
    model.decode_video(z_b, parallel=True, cache=cache)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_repeat):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        model.decode_video(z_b, parallel=True, cache=cache)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    return sum(times_ms[:-1]) / (len(times_ms) - 1)


@torch.no_grad()
def _time_second_decode_new(
    model: TAEHVNew, latents: torch.Tensor, *, chunk_t: int, n_repeat: int
) -> float:
    """Warm the cache + capture the CUDA graph, then time ``n_repeat`` decodes."""
    z_a = latents[:, :chunk_t]
    z_b = latents[:, chunk_t : 2 * chunk_t]

    cache = model.prepare_cache()
    model.decode(z_a, cache=cache)
    # 2 wrapper warmups + 1 capture pass = wrapped graph ready before timing.
    for _ in range(3):
        model.decode(z_b, cache=cache)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_repeat):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        model.decode(z_b, cache=cache)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    return sum(times_ms[:-1]) / (len(times_ms) - 1)


def _peak_mem_mib(fn, *args, **kwargs) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    return out, peak


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpts",
        nargs="+",
        choices=sorted(AVAILABLE_TAEHV_CHECKPOINT_PATHS),
        default=["lighttae"],
    )
    p.add_argument("--quick", action="store_true", help="Disable torch.compile.")
    p.add_argument("--n-repeat", type=int, default=10)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument(
        "--chunk-t", type=int, default=2, help="Latent timesteps per AR chunk."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = torch.bfloat16
    device = torch.device("cuda")
    h, w = args.height, args.width

    use_compile = not args.quick
    print("=" * 78)
    print(
        f"TAEHV 2nd-rollout decode latency: legacy vs slim "
        f"(bf16, {device}, H={h}, W={w}, use_compile={use_compile})"
    )
    print(f"  decode chunk A=B={args.chunk_t} latents (warm + timed x{args.n_repeat})")
    print(f"  ckpts={args.ckpts}")
    print(f"  inductor cache: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    print(f"  triton cache:   {os.environ['TRITON_CACHE_DIR']}")
    print("=" * 78, "\n", flush=True)

    rows: list[tuple[str, float, float, float, float]] = []
    for ckpt_key in args.ckpts:
        ckpt_path = AVAILABLE_TAEHV_CHECKPOINT_PATHS[ckpt_key]
        t_ckpt = time.perf_counter()
        print(f"[{ckpt_key}] building models...", flush=True)
        legacy, new = _build_pair(ckpt_path, dtype, device, use_compile=use_compile)
        _log("models built", t_ckpt)

        torch.manual_seed(0)
        latents = torch.empty(
            1, 2 * args.chunk_t, 16, h // 8, w // 8, dtype=dtype, device=device
        ).uniform_(-1, 1)

        t = time.perf_counter()
        leg_dec, leg_dec_mem = _peak_mem_mib(
            _time_second_decode_legacy,
            legacy,
            latents,
            chunk_t=args.chunk_t,
            n_repeat=args.n_repeat,
        )
        _log(f"legacy decode timed: {leg_dec:.3f} ms", t)

        t = time.perf_counter()
        new_dec, new_dec_mem = _peak_mem_mib(
            _time_second_decode_new,
            new,
            latents,
            chunk_t=args.chunk_t,
            n_repeat=args.n_repeat,
        )
        _log(f"new decode timed:    {new_dec:.3f} ms", t)

        _log(f"[{ckpt_key}] total elapsed", t_ckpt)
        print()

        rows.append((ckpt_key, leg_dec, new_dec, leg_dec_mem, new_dec_mem))

        del legacy, new, latents
        torch.cuda.empty_cache()

    hdr = (
        f"{'ckpt':<12} {'legacy ms':>10} {'new ms':>10} "
        f"{'speedup':>9} {'legacy MiB':>11} {'new MiB':>10} {'mem dlt':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ckpt, lt, nt, lm, nm in rows:
        speedup = lt / nt if nt > 0 else float("inf")
        mem_dlt = nm - lm
        print(
            f"{ckpt:<12} {lt:>10.3f} {nt:>10.3f} "
            f"{speedup:>8.2f}x {lm:>11.1f} {nm:>10.1f} {mem_dlt:>+10.1f}"
        )


if __name__ == "__main__":
    main()
