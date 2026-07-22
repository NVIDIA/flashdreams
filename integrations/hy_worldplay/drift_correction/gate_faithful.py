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

"""Faithful step-0 gate: alpha* on real drifted-vs-clean loop pairs.

Unlike :mod:`gate_systematicity` (proxy perturbations), this measures the
method's actual premise on :mod:`build_pairs` output: at a late chunk of a
strafe-loop rollout, the drifted history's memory frames are swapped for
their lap-aligned lap-``CLEAN_LAP`` counterparts -- same actions, same
viewmats, same RoPE positions, clean content -- and the x0 prediction gap is
decomposed over noise seeds into systematic bias vs variance. Also reports
the drift-gap magnitude ``rel`` (the training loss denominator); near-zero
means no signal to train on.

Run after ``build_pairs.py`` from the repo root::

    .venv/bin/python integrations/hy_worldplay/drift_correction/gate_faithful.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from _pairs import (
    chunk_x0,
    clean_counterfactual,
    history_of,
    load_clip,
    make_ctrl,
)
from _rollout import build_runner, finish_probe_chunk, predict_x0, start_probe_chunk
from hy_worldplay._action import HyWorldPlayWan21TransformerCache
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame

## Gate configuration

PAIRS_DIR = Path("integrations/hy_worldplay/drift_correction/outputs/pairs")
"""Clip files from ``build_pairs.py``."""

OUT_PATH = Path(
    "integrations/hy_worldplay/drift_correction/outputs/gate/gate_faithful.json"
)
"""Aggregated per-timestep results."""

PROBE_CHUNKS = (12, 16, 20, 23)
"""Late chunks probed per clip (laps 3+ of a 6-lap / 24-chunk rollout)."""

M_NOISE = 8
"""Noise seeds per (clip, chunk, timestep) cell."""

CLEAN_LAP = 1
"""Lap supplying the clean teacher content. Lap 0 is even cleaner but its
first frame is the stamped real-image latent (different distribution)."""


def main() -> None:
    torch.set_grad_enabled(False)
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"

    meta = torch.load(clips[0], map_location="cpu", weights_only=False)
    runner = build_runner(
        num_chunk=meta["num_chunk"], pose=meta["pose"], output_dir=PAIRS_DIR
    )
    pipe = runner.pipeline
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype
    cfg = runner.config
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps, sigmas = scheduler.timesteps, scheduler.sigmas
    n_steps = len(timesteps) - 1

    # Accumulate squared-bias / variance / norms per timestep across all
    # (clip, chunk) cells.
    agg = {i: {"alphas": [], "alphas_ub": [], "rels": []} for i in range(n_steps)}
    for clip_path in clips:
        d = load_clip(clip_path, device, dtype)
        lap = d["lap_latents"]
        for k in PROBE_CHUNKS:
            if k >= d["num_chunk"]:
                continue
            selected = d["memory_frame_indices"][k]
            if not selected:
                continue
            ctrl = make_ctrl(d, k, device=device, dtype=dtype)
            h_gen = history_of(d, k)
            h_clean = clean_counterfactual(
                h_gen, selected=selected, lap_latents=lap, clean_lap=CLEAN_LAP
            )
            x0 = chunk_x0(d, k)

            z_ts: dict[int, list[Tensor]] = {}
            for t_idx in range(n_steps):
                sig = sigmas[t_idx].to(dtype)
                z_ts[t_idx] = []
                for m in range(M_NOISE):
                    g = torch.Generator(device=device).manual_seed(
                        900_000 + 10_000 * k + 100 * t_idx + m
                    )
                    eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
                    z_ts[t_idx].append((1 - sig) * x0 + sig * eps)

            preds: dict[str, dict[int, list[Tensor]]] = {}
            for name, h in (("gen", h_gen), ("clean", h_clean)):
                start_probe_chunk(tc, ar_idx=k, history=h)
                preds[name] = {
                    t_idx: [
                        predict_x0(
                            transformer,
                            tc,
                            ctrl=ctrl,
                            z_t=z,
                            timestep=timesteps[t_idx].to(dtype),
                            sigma=float(sigmas[t_idx]),
                        )
                        for z in z_list
                    ]
                    for t_idx, z_list in z_ts.items()
                }
                finish_probe_chunk(tc, ar_idx=k)

            line = [f"{clip_path.stem} k={k:2d}"]
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
                a = bias_sq / (bias_sq + var + 1e-12)
                a_ub = bias_sq_ub / (bias_sq_ub + var_ub + 1e-12)
                agg[t_idx]["alphas"].append(a)
                agg[t_idx]["alphas_ub"].append(a_ub)
                agg[t_idx]["rels"].append(rel)
                line.append(
                    f"t={int(timesteps[t_idx]):4d} a*={a:.3f}/{a_ub:.3f} rel={rel:.3f}"
                )
            print(" | ".join(line), flush=True)

    summary = {
        str(int(timesteps[i])): {
            "alpha_star": sum(v["alphas"]) / len(v["alphas"]),
            "alpha_star_unbiased": sum(v["alphas_ub"]) / len(v["alphas_ub"]),
            "rel": sum(v["rels"]) / len(v["rels"]),
            "cells": len(v["alphas"]),
        }
        for i, v in agg.items()
        if v["alphas"]
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n========== HY-WorldPlay FAITHFUL gate (real drift pairs) ==========")
    for t, v in summary.items():
        print(
            f"t={t:>4s}: alpha* {v['alpha_star']:.3f} (unbiased {v['alpha_star_unbiased']:.3f})"
            f" | rel drift gap {v['rel']:.3f} | {v['cells']} cells"
        )
    all_ub = [a for v in agg.values() for a in v["alphas_ub"]]
    mean_rel = sum(r for v in agg.values() for r in v["rels"]) / max(
        1, sum(len(v["rels"]) for v in agg.values())
    )
    frac = sum(a >= 0.7 for a in all_ub) / len(all_ub)
    print(
        f"cells with unbiased alpha* >= 0.7: {frac:.0%} | mean rel gap {mean_rel:.3f}"
    )
    if mean_rel < 0.01:
        print("RESULT: drift gap ~zero -> nothing to correct. STOP.")
    elif frac >= 0.7:
        print("RESULT: real drift gap is systematic. GO for v1 training.")
    else:
        print(
            "RESULT: below the reference bar -- report before committing to training."
        )
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
