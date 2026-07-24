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

"""Faithful step-0 gate: alpha* on real drifted-vs-clean loop pairs (Omnidreams).

Measures the corrector's premise on ``build_pairs.py`` output: at a late
chunk of a tiled-HDMap loop rollout, the KV-window history (the host's
entire history mechanism) is swapped for its lap-1 clean counterparts —
same HDMap conditioning, same RoPE positions, same context-noise eps,
clean content — and the *velocity* prediction gap is decomposed over noise
seeds into systematic bias vs variance. Probes span rollout depths up to
~21.5 s because this host fights drift with Self-Forcing distillation and
its residual accumulation shows late; the report resolves ``rel`` by depth.

Kill bars (pre-stated): mean rel gap < 0.01 -> nothing to correct, STOP;
alpha* below the reference bar -> report before any training.

Run after ``build_pairs.py`` from the flashdreams repo root::

    .venv/bin/python integrations/omnidreams/drift_correction/gate_faithful.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import (
    build_pipeline,
    finish_probe_chunk,
    load_clip,
    predict_v,
    replay_history,
    reset_history,
    start_probe_chunk,
)
from torch import Tensor

## Gate configuration

PAIRS_DIR = Path(
    os.environ.get(
        "PAIRS_DIR", "integrations/omnidreams/drift_correction/outputs/pairs_v2"
    )
)
"""Clip files from ``build_pairs.py``."""

OUT_PATH = Path(
    os.environ.get(
        "GATE_OUT",
        "integrations/omnidreams/drift_correction/outputs/gate/gate_faithful_v2.json",
    )
)
"""Aggregated per-timestep and per-depth results."""

PROBE_CHUNKS = tuple(
    int(x) for x in os.environ.get("PROBE_CHUNKS", "24,39,54,69,79").split(",")
)
"""Probe chunks spanning ~6.6 s to ~21.4 s of rollout. Lap default: all
satisfy ``(k - 1) % LAP_CHUNKS >= 3`` so the 3-chunk KV window lies inside
one lap (no conditioning-teleport contamination), and all sit in laps >= 4
(convention: drifted side well past the lap-2 clean
reference). Fork pairs (v4) pass their own set with the window inside one
fork segment."""

PAIR_SCHEME = os.environ.get("PAIR_SCHEME", "lap")
"""``lap`` (v1-v3 loop pairs) or ``fork`` (v4 re-anchored fork pairs:
clean counterpart of chunk ``j`` = fork ``s`` with the largest
``fork_starts[s] <= j``, chunk ``j - fork_starts[s]`` of its latents)."""

WINDOW_CHUNKS = 3
"""KV-window span in chunks (``window_size_t=6`` / ``len_t=2``)."""

M_NOISE = 8
"""Noise seeds per (clip, chunk, timestep) cell."""

CLEAN_LAP = 2
"""Lap supplying the clean counterpart content (convention 2026-07-22:
lap 2, keeping the synthetic-seed transition tail of laps 0-1 out of the
clean reference; the 2026-07-22 GO gate ran with lap 1)."""


def lap_aligned(k: int, lap_chunks: int) -> int:
    """Map chunk ``k`` (>= lap 1) to its lap-:data:`CLEAN_LAP` counterpart."""
    assert k >= 1, "chunk 0 is image-anchored and never remapped"
    return 1 + CLEAN_LAP * lap_chunks + (k - 1) % lap_chunks


def window_indices(k: int) -> list[int]:
    """Chunk indices whose KV survives in the window when probing chunk ``k``."""
    return list(range(max(0, k - WINDOW_CHUNKS), k))


def histories(d: dict, k: int) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Build the (gen latents, clean latents, hdmaps) prefix lists for chunk ``k``.

    Both branches replay the full prefix ``0 .. k-1`` at original indices
    (RoPE positions preserved; pre-window chunks are evicted and only feed
    the roll). The clean branch swaps the surviving window chunks' content
    for their lap-aligned lap-:data:`CLEAN_LAP` counterparts.
    """
    lap_chunks = d["lap_chunks"]
    gen = [d["latents"][j] for j in range(k)]
    clean = list(gen)
    for j in window_indices(k):
        if j == 0:
            continue  # image-anchored chunk, never remapped
        if PAIR_SCHEME == "fork":
            starts = d["fork_starts"]
            s = max(i for i, cs in enumerate(starts) if cs <= j)
            clean[j] = d["fork_latents"][s][j - starts[s]].to(
                gen[0].device, gen[0].dtype
            )
        else:
            src = lap_aligned(j, lap_chunks)
            if src != j:
                clean[j] = d["latents"][src]
    hdmaps = [d["hdmaps"][j] for j in range(k)]
    return gen, clean, hdmaps


def main() -> None:
    torch.set_grad_enabled(False)
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"

    pipe = build_pipeline(with_oneshot_encoders=False)
    device = pipe.device
    dtype = torch.bfloat16
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps = scheduler.denoising_step_list
    sigmas = scheduler.denoising_sigmas
    n_steps = timesteps.shape[0]  # ty: ignore[not-subscriptable]
    ctx_t = torch.tensor(
        float(pipe.diffusion_model.config.context_noise), device=device, dtype=dtype
    )

    agg = {
        i: {"alphas": [], "alphas_ub": [], "rels": [], "rels_x0": []}
        for i in range(n_steps)
    }
    by_depth: dict[int, list[float]] = {}
    for clip_path in clips:
        d = load_clip(clip_path, device, dtype)
        emb = d["embeddings"]
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=emb["text_embeddings"],
            image_embeddings=emb["image_embeddings"],
        )
        tc = cache.transformer_cache

        for k in PROBE_CHUNKS:
            if k >= d["num_chunk"]:
                continue
            gen_hist, clean_hist, hdmaps = histories(d, k)
            hdmap_k = d["hdmaps"][k]
            x0 = d["latents"][k]

            z_ts: dict[int, list[Tensor]] = {}
            for t_idx in range(n_steps):
                sig = sigmas[t_idx].to(dtype)  # ty: ignore[not-subscriptable]
                z_ts[t_idx] = []
                for m in range(M_NOISE):
                    g = torch.Generator(device=device).manual_seed(
                        900_000 + 10_000 * k + 100 * t_idx + m
                    )
                    eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
                    z_ts[t_idx].append((1 - sig) * x0 + sig * eps)

            preds: dict[str, dict[int, list[Tensor]]] = {}
            for name, hist in (("gen", gen_hist), ("clean", clean_hist)):
                reset_history(tc)
                replay_history(
                    transformer,  # ty: ignore[invalid-argument-type]
                    tc,
                    latents=hist,
                    hdmaps=hdmaps,
                    context_timestep=ctx_t,
                    add_noise=scheduler.add_noise,
                )
                start_probe_chunk(tc, ar_idx=k)
                preds[name] = {
                    t_idx: [
                        predict_v(
                            transformer,  # ty: ignore[invalid-argument-type]
                            tc,
                            hdmap=hdmap_k,
                            z_t=z,
                            timestep=timesteps[t_idx].to(device=device, dtype=dtype),  # ty: ignore[not-subscriptable]
                        )
                        for z in z_list
                    ]
                    for t_idx, z_list in z_ts.items()
                }
                finish_probe_chunk(tc, ar_idx=k)

            line = [f"{clip_path.stem} k={k:2d} ({(5 + 8 * k) / 30:5.1f}s)"]
            for t_idx in range(n_steps):
                deltas = torch.stack(
                    [a - b for a, b in zip(preds["gen"][t_idx], preds["clean"][t_idx])]
                )
                m = deltas.shape[0]
                bias = deltas.mean(dim=0)
                bias_sq = bias.square().sum().item()
                sq_dev = (deltas - bias).square().sum(dim=tuple(range(1, deltas.ndim)))
                var = sq_dev.mean().item()
                var_ub = sq_dev.sum().item() / (m - 1)
                bias_sq_ub = max(0.0, bias_sq - var_ub / m)
                gen_norm = (
                    torch.stack(preds["gen"][t_idx]).flatten(1).norm(dim=1).mean()
                )
                rel = (deltas.flatten(1).norm(dim=1).mean() / (gen_norm + 1e-12)).item()
                # x0-space view: x0 = z_t - sigma * v with shared (z_t, sigma),
                # so delta_x0 = -sigma * delta_v and only the denominator changes.
                sig = float(sigmas[t_idx])  # ty: ignore[not-subscriptable]
                x0_norm = (
                    torch.stack(
                        [
                            z.float() - sig * v
                            for z, v in zip(z_ts[t_idx], preds["gen"][t_idx])
                        ]
                    )
                    .flatten(1)
                    .norm(dim=1)
                    .mean()
                )
                rel_x0 = (
                    sig * deltas.flatten(1).norm(dim=1).mean() / (x0_norm + 1e-12)
                ).item()
                a = bias_sq / (bias_sq + var + 1e-12)
                a_ub = bias_sq_ub / (bias_sq_ub + var_ub + 1e-12)
                agg[t_idx]["alphas"].append(a)
                agg[t_idx]["alphas_ub"].append(a_ub)
                agg[t_idx]["rels"].append(rel)
                agg[t_idx]["rels_x0"].append(rel_x0)
                by_depth.setdefault(k, []).append(rel)
                line.append(
                    f"t={int(timesteps[t_idx]):4d} a*={a:.3f}/{a_ub:.3f}"  # ty: ignore[not-subscriptable]
                    f" rel_v={rel:.4f} rel_x0={rel_x0:.4f}"
                )
            print(" | ".join(line), flush=True)
        del cache, tc

    summary = {
        "per_timestep": {
            str(int(timesteps[i])): {  # ty: ignore[not-subscriptable]
                "alpha_star": sum(v["alphas"]) / len(v["alphas"]),
                "alpha_star_unbiased": sum(v["alphas_ub"]) / len(v["alphas_ub"]),
                "rel_v": sum(v["rels"]) / len(v["rels"]),
                "rel_x0": sum(v["rels_x0"]) / len(v["rels_x0"]),
                "cells": len(v["alphas"]),
            }
            for i, v in agg.items()
            if v["alphas"]
        },
        "rel_v_by_depth": {
            str(k): {"seconds": (5 + 8 * k) / 30, "rel_v": sum(r) / len(r)}
            for k, r in sorted(by_depth.items())
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(
        "\n========== Omnidreams FAITHFUL gate (real drift pairs, v-space) =========="
    )
    for t, v in summary["per_timestep"].items():
        print(
            f"t={t:>4s}: alpha* {v['alpha_star']:.3f} (unbiased {v['alpha_star_unbiased']:.3f})"
            f" | rel_v {v['rel_v']:.4f} rel_x0 {v['rel_x0']:.4f} | {v['cells']} cells"
        )
    for k, v in summary["rel_v_by_depth"].items():
        print(f"depth k={k:>3s} ({v['seconds']:5.1f}s): rel_v {v['rel_v']:.4f}")
    all_ub = [a for v in agg.values() for a in v["alphas_ub"]]
    mean_rel = sum(r for v in agg.values() for r in v["rels"]) / max(
        1, sum(len(v["rels"]) for v in agg.values())
    )
    frac = sum(a >= 0.7 for a in all_ub) / len(all_ub)
    print(
        f"cells with unbiased alpha* >= 0.7: {frac:.0%} | mean rel_v gap {mean_rel:.4f}"
    )
    if mean_rel < 0.01:
        print("RESULT: drift gap ~zero -> nothing to correct. STOP.")
    elif frac >= 0.7:
        print("RESULT: real drift gap is systematic. Report headroom before training.")
    else:
        print(
            "RESULT: below the reference bar -- report before committing to training."
        )
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
