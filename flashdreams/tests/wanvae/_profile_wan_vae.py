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

"""Compare 2nd-rollout latency: reference vs slim Wan VAE (encode + decode).

Run on a GPU (typically inside an interactive srun)::

    PYTHONPATH=./flashdreams python flashdreams/tests/wanvae/_profile_wan_vae.py
    PYTHONPATH=./flashdreams python flashdreams/tests/wanvae/_profile_wan_vae.py --all-ckpts
    PYTHONPATH=./flashdreams python flashdreams/tests/wanvae/_profile_wan_vae.py --quick

The first chunk of every encode/decode populates the streaming cache and
allocates workspaces; we measure the *second* chunk (and only the second)
to get a stable, cache-warm number that mirrors steady-state rollout.

Notes on first-run cost: with ``use_compile=True`` (the default), Inductor
autotunes ``max-autotune-no-cudagraphs`` for each unique input shape.
Wan VAE has many conv3d layers, so first run can take 3-8 min per ckpt.
Subsequent runs hit the on-disk Inductor + triton caches and are ~5x
faster. Pass ``--quick`` to skip compilation entirely for fast iteration
(CUDA-graph capture only) -- useful when you only care about the
shape/wrapper/cache mechanics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Park Inductor + triton caches on local disk before torch is imported, so
# repeat runs of this script reuse autotune decisions instead of redoing
# them through Lustre. Honour the user's explicit override if present.
_CACHE_ROOT = f"/tmp/{os.environ.get('USER', 'flashdreams')}/wanvae_profile"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{_CACHE_ROOT}/inductor")
os.environ.setdefault("TRITON_CACHE_DIR", f"{_CACHE_ROOT}/triton")

import torch  # noqa: E402

from flashdreams.core.checkpoint.load import load_checkpoint  # noqa: E402
from flashdreams.recipes.wan.autoencoder.vae import (  # noqa: E402
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAE as WanVAENew,
)

# Sibling module: this script is meant to be run as `python file.py` (its
# own process), so a simple sys.path hack is sufficient. The package-form
# relative import used by `test_wan_vae_equivalence.py` doesn't apply
# here because `__name__ == "__main__"` and there is no parent package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import impl_reference as _impl_reference  # noqa: E402  # ty:ignore[unresolved-import]

WanVAELegacy = _impl_reference.WanVAE


def _build_pair(
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_compile: bool,
) -> tuple[WanVAELegacy, WanVAENew]:
    use_lightvae = "lightvae" in checkpoint_path
    weights = load_checkpoint(checkpoint_path)
    weights = {k: v.to(dtype) for k, v in weights.items()}  # ty:ignore[call-non-callable]

    def _cached(_path):
        return weights

    with (
        patch.object(_impl_reference, "load_checkpoint", _cached),
        patch("flashdreams.recipes.wan.autoencoder.vae.load_checkpoint", _cached),
    ):
        legacy = WanVAELegacy(
            vae_path=checkpoint_path,
            use_lightvae=use_lightvae,
            dtype=dtype,
            device=device,
        )
        new = WanVAENew(
            vae_path=checkpoint_path,
            use_lightvae=use_lightvae,
            use_compile=use_compile,
        ).to(device=device, dtype=dtype)
    return legacy, new


def _log(msg: str, t0: float | None = None) -> float:
    """Print a progress line; return current wall time so callers can chain."""
    t = time.perf_counter()
    prefix = f"[{t - t0:6.1f}s] " if t0 is not None else ""
    print(f"  {prefix}{msg}", flush=True)
    return t


@torch.no_grad()
def _time_second_encode(
    model, video, *, chunk_a_t: int, chunk_b_t: int, n_repeat: int
) -> float:
    """Warm the cache + capture any CUDA graph, then time ``n_repeat``
    chunk-B encodes and return mean(ms) excluding the slowest sample."""
    enc_a = video[:, :, :chunk_a_t]
    enc_b = video[:, :, chunk_a_t : chunk_a_t + chunk_b_t]

    cache = model.prepare_cache()
    model.encode(enc_a, cache=cache)
    # 2 wrapper warmups + 1 capture pass = wrapped graph ready before timing.
    for _ in range(3):
        model.encode(enc_b, cache=cache)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_repeat):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        model.encode(enc_b, cache=cache)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    # Drop the slowest one as a poor-man's outlier filter.
    return sum(times_ms[:-1]) / (len(times_ms) - 1)


@torch.no_grad()
def _time_second_decode(
    model, latents, *, chunk_a_t: int, chunk_b_t: int, n_repeat: int
) -> float:
    """Same pattern as :func:`_time_second_encode` for decode."""
    z_a = latents[:, :, :chunk_a_t]
    z_b = latents[:, :, chunk_a_t : chunk_a_t + chunk_b_t]

    cache = model.prepare_cache()
    model.decode(z_a, cache=cache)
    for _ in range(3):
        model.decode(z_b, cache=cache)
    torch.cuda.synchronize()

    times_ms = []
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
    """Return (result, peak_alloc_MiB) for ``fn(*args, **kwargs)`` on CUDA."""
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
        choices=sorted(AVAILABLE_WAN_VAE_CHECKPOINT_PATHS),
        default=["vae"],
        help=(
            "Which checkpoints to profile. Default is just 'vae'; "
            "lightvae roughly doubles wall time."
        ),
    )
    p.add_argument(
        "--all-ckpts",
        action="store_true",
        help="Shortcut for --ckpts lightvae vae.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Disable torch.compile on the slim VAE (CUDA-graph capture only). "
            "Saves ~3-8 min per ckpt of first-run compile time at the cost "
            "of ~10-30%% steady-state speed."
        ),
    )
    p.add_argument(
        "--n-repeat",
        type=int,
        default=10,
        help="Timed iterations per cell (CUDA graphs are very stable; 10 is plenty).",
    )
    p.add_argument(
        "--height", type=int, default=720, help="Frame height (defaults to 720)."
    )
    p.add_argument(
        "--width", type=int, default=1280, help="Frame width (defaults to 1280)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.all_ckpts:
        args.ckpts = ["lightvae", "vae"]

    dtype = torch.bfloat16
    device = torch.device("cuda")

    # AR step 0: 5 video frames (1 seed + 1 body chunk of 4) -> 2 latents.
    # AR step 1+: 8 video frames (2 body chunks of 4) -> 2 latents.
    enc_chunk_a, enc_chunk_b = 5, 8
    dec_chunk_a, dec_chunk_b = 2, 2
    h, w = args.height, args.width

    use_compile = not args.quick
    print(f"{'=' * 78}")
    print(
        f"Wan VAE 2nd-rollout latency: legacy vs slim "
        f"(bf16, {device}, H={h}, W={w}, use_compile={use_compile})"
    )
    print(
        f"  encode chunk A={enc_chunk_a} frames (warm), "
        f"chunk B={enc_chunk_b} frames (timed x{args.n_repeat})"
    )
    print(
        f"  decode chunk A={dec_chunk_a} latents (warm), "
        f"chunk B={dec_chunk_b} latents (timed x{args.n_repeat})"
    )
    print(f"  ckpts={args.ckpts}")
    print(f"  inductor cache: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    print(f"  triton cache:   {os.environ['TRITON_CACHE_DIR']}")
    print(f"{'=' * 78}\n", flush=True)

    rows: list[tuple[str, str, float, float, float, float]] = []
    for ckpt_key in args.ckpts:
        ckpt_path = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS[ckpt_key]
        t_ckpt = time.perf_counter()
        print(f"[{ckpt_key}] building models...", flush=True)
        legacy, new = _build_pair(ckpt_path, dtype, device, use_compile=use_compile)
        _log("models built", t_ckpt)

        torch.manual_seed(0)
        video = torch.empty(
            1, 3, enc_chunk_a + enc_chunk_b, h, w, dtype=dtype, device=device
        ).uniform_(-1, 1)
        latents = torch.empty(
            1, 16, dec_chunk_a + dec_chunk_b, h // 8, w // 8, dtype=dtype, device=device
        ).uniform_(-1, 1)

        t = time.perf_counter()
        leg_enc, leg_enc_mem = _peak_mem_mib(
            _time_second_encode,
            legacy,
            video,
            chunk_a_t=enc_chunk_a,
            chunk_b_t=enc_chunk_b,
            n_repeat=args.n_repeat,
        )
        _log(f"legacy encode timed: {leg_enc:.3f} ms", t)

        t = time.perf_counter()
        new_enc, new_enc_mem = _peak_mem_mib(
            _time_second_encode,
            new,
            video,
            chunk_a_t=enc_chunk_a,
            chunk_b_t=enc_chunk_b,
            n_repeat=args.n_repeat,
        )
        _log(f"new encode timed:    {new_enc:.3f} ms", t)

        t = time.perf_counter()
        leg_dec, leg_dec_mem = _peak_mem_mib(
            _time_second_decode,
            legacy,
            latents,
            chunk_a_t=dec_chunk_a,
            chunk_b_t=dec_chunk_b,
            n_repeat=args.n_repeat,
        )
        _log(f"legacy decode timed: {leg_dec:.3f} ms", t)

        t = time.perf_counter()
        new_dec, new_dec_mem = _peak_mem_mib(
            _time_second_decode,
            new,
            latents,
            chunk_a_t=dec_chunk_a,
            chunk_b_t=dec_chunk_b,
            n_repeat=args.n_repeat,
        )
        _log(f"new decode timed:    {new_dec:.3f} ms", t)

        _log(f"[{ckpt_key}] total elapsed", t_ckpt)
        print()

        rows.append((ckpt_key, "encode", leg_enc, new_enc, leg_enc_mem, new_enc_mem))
        rows.append((ckpt_key, "decode", leg_dec, new_dec, leg_dec_mem, new_dec_mem))

        del legacy, new, video, latents
        torch.cuda.empty_cache()

    hdr = (
        f"{'ckpt':<10} {'phase':<7} {'legacy ms':>10} {'new ms':>10} "
        f"{'speedup':>9} {'legacy MiB':>11} {'new MiB':>10} {'mem dlt':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ckpt, phase, lt, nt, lm, nm in rows:
        speedup = lt / nt if nt > 0 else float("inf")
        mem_dlt = nm - lm
        print(
            f"{ckpt:<10} {phase:<7} {lt:>10.3f} {nt:>10.3f} "
            f"{speedup:>8.2f}x {lm:>11.1f} {nm:>10.1f} {mem_dlt:>+10.1f}"
        )


if __name__ == "__main__":
    main()
