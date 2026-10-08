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

"""Style-specific alpha*(t) gate for the style-drift corrector.

``drift_correction/gate_faithful.py``'s measurement applied to the
``gen_style_drift_pairs.py`` branch corpus: the deploy hook
(``omnidreams/_drift_corrector.py``) currently gates the style corrector
with the PHOTOREAL profile (``GATE_ALPHA`` = {1000: 0.96, 803: 0.667},
measured on tiled-HDMap loop drift); this script measures the styled
counterpart so styled worlds can be gated on their own numbers.

Cells mirror ``train_style_corrector.py`` exactly: probe absolute chunk
``k`` with the DRIFTED branch swapped ``k - a + 1`` chunks earlier
(:data:`DEPTHS`, the +8..+24 long-hold blur regime) against the CLEAN
styled reference branch swapped ``1..REF_MAX`` chunks earlier — same
absolute index, same HDMap, same seed, matched RoPE by construction. Both
branches replay a 3-chunk KV window at deploy semantics (style LoRA scale
1 in-window, style text KV rebuilt at base weights) and are probed at the
SAME ``z_t`` (built from the drifted chunk's x0) over :data:`M_NOISE`
seeds; the velocity gap decomposes into systematic bias vs seed variance,
``alpha*(t) = bias^2 / (bias^2 + var)`` (gate_faithful's estimator, both
plug-in and unbiased). Probed timesteps: the 2-step solver schedule
(t=1000 and the 450-warped t=803) plus the ``finalize_kv_cache`` context
forward's t=128, which the deploy gate resolves by nearest-t lookup.

Outputs ``edit_sft/outputs/gate_style.json`` (per-timestep, per-hold-depth
breakdown, and a flat ``gate_alpha`` table consumable by the deploy hook's
``GATE_ALPHA_JSON`` override), prints the comparison against the deployed
photoreal profile and a one-line recommendation.

Env knobs: ``PAIRS_DIR``, ``GATE_OUT``, ``STYLE_LORA``, ``STYLE_RANK``,
``STYLES``, ``DEPTHS``, ``CELLS_PER_DEPTH``, ``REF_MAX``, ``M_NOISE``,
``SEED``.

Run from the flashdreams repo root (forward-only, fits the co-tenant
VRAM share)::

    .venv/bin/python integrations/omnidreams/edit_sft/gate_style.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _host import CONTEXT_NOISE_SEED, build_pipeline, reset_history
from _lora import apply_lora, load_lora, set_lora_scale, unwrap_compiled
from _train_attn import functional_attention, patch_functional_attention
from omnidreams._drift_corrector import GATE_ALPHA as PHOTOREAL_GATE
from omnidreams._drift_corrector import _nearest_alpha
from omnidreams._edit_lora import _LORA_TARGETS as LORA_TARGETS
from style_prompts import clip_key
from torch import Tensor

## Gate configuration

BASE = Path("integrations/omnidreams/edit_sft")
PAIRS_DIR = Path(os.environ.get("PAIRS_DIR", str(BASE / "outputs/style_drift_pairs")))
OUT_PATH = Path(os.environ.get("GATE_OUT", str(BASE / "outputs/gate_style.json")))

STYLE_LORA = Path(
    os.environ.get("STYLE_LORA", str(BASE / "outputs/lora_style_step1600.pt"))
)
"""Style-skin checkpoint (must be the one the branches were rolled with)."""

STYLE_RANK = int(os.environ.get("STYLE_RANK", "64"))
"""Rank of the style checkpoint (the eval RANK-must-match gotcha)."""

STYLES = frozenset(s for s in os.environ.get("STYLES", "").split(",") if s)
"""Optional slug filter; empty = every style found in the pairs dir."""

DEPTHS = tuple(int(x) for x in os.environ.get("DEPTHS", "8,12,16,20,24").split(","))
"""Style-hold depths (chunks since swap, 1-based) of the drifted branch to
probe — the +8..+24 regime the corrector deploys on (its training band is
+8..+20; +24 checks the tail)."""

CELLS_PER_DEPTH = int(os.environ.get("CELLS_PER_DEPTH", "1"))
"""Probe cells per (clip, style, depth); 1 keeps the sweep ~30 min."""

REF_MAX = int(os.environ.get("REF_MAX", "4"))
"""Max hold depth of the clean styled reference (+1..+4 = the manifold
``train_style_corrector.py`` teaches toward)."""

REPLAY_CHUNKS = 3
"""History chunks replayed before a probe = the full ``window_size_t=6`` /
``len_t=2`` KV window (the trainer's setting)."""

M_NOISE = int(os.environ.get("M_NOISE", "8"))
"""Noise seeds per (cell, timestep) — gate_faithful's setting."""

DEV_BAR = 0.05
"""|style alpha* - deployed photoreal alpha| that matters: at the shipped
gain 0.25 this is a >=7% relative change of the smaller effective per-step
LoRA scale."""

SEED = int(os.environ.get("SEED", "0"))


def load_pairs() -> list[dict]:
    """Load the branch corpus; per-branch full latent lists as the trainer."""
    bases: dict[str, dict] = {}
    pairs: list[dict] = []
    for path in sorted(PAIRS_DIR.glob("*__*.pt")):
        uuid, slug = path.stem.split("__", 1)
        if STYLES and slug not in STYLES:
            continue
        if uuid not in bases:
            base_path = PAIRS_DIR / f"{uuid}_base.pt"
            assert base_path.exists(), f"missing {base_path} for {path.name}"
            bases[uuid] = torch.load(base_path, map_location="cpu", weights_only=False)
        base = bases[uuid]
        n = int(base["n_chunks"])
        assert len(base["latents"]) == n and len(base["hdmaps"]) == n
        d = torch.load(path, map_location="cpu", weights_only=False)
        full = {
            int(s): base["latents"][: int(s)] + lat for s, lat in d["branches"].items()
        }
        pairs.append(
            {
                "uuid": uuid,
                "slug": slug,
                "n_chunks": n,
                "hdmaps": base["hdmaps"],
                "full": full,
                "offsets": sorted(full),
            }
        )
    assert pairs, f"no styled branch files under {PAIRS_DIR}"
    return pairs


def select_cells(p: dict, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    """Stratified probe cells ``(k, a, b, depth)`` for one (clip, style) pair.

    Per drift depth in :data:`DEPTHS`, up to :data:`CELLS_PER_DEPTH` chunks
    ``k`` with the drifted branch swapped at ``a = k - depth + 1`` and a
    clean reference branch ``b`` at hold depth ``1..REF_MAX`` — the
    trainer's cell rule, stratified by depth instead of pooled. With the
    corpus offsets spaced 4 apart, multiple-of-4 depths pin the reference
    at hold depth +4: the reference KV window is then fully styled content
    (no unstyled pre-swap chunks), the cleanest form of the premise.
    """
    offsets, n = p["offsets"], p["n_chunks"]
    cells: list[tuple[int, int, int, int]] = []
    for depth in DEPTHS:
        ks = [
            k
            for k in range(1 + REPLAY_CHUNKS, n)
            if (k - depth + 1) in offsets
            and any(1 <= k - b + 1 <= REF_MAX for b in offsets)
        ]
        if not ks:
            print(f"{p['uuid'][:8]}__{p['slug']}: no cells at depth {depth}")
            continue
        picks = rng.choice(len(ks), size=min(CELLS_PER_DEPTH, len(ks)), replace=False)
        for i in sorted(int(x) for x in picks):
            k = ks[i]
            refs = [b for b in offsets if 1 <= k - b + 1 <= REF_MAX]
            b = refs[int(rng.integers(len(refs)))]
            cells.append((k, k - depth + 1, b, depth))
    return cells


def main() -> None:
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(SEED)
    pairs = load_pairs()

    prompt_emb = torch.load(
        BASE / "outputs/style_embeddings.pt", map_location="cpu", weights_only=False
    )
    assets = torch.load(
        BASE / "outputs/style_clip_assets.pt", map_location="cpu", weights_only=False
    )

    pipe = build_pipeline(with_oneshot_encoders=False)
    device = pipe.device
    dtype = torch.bfloat16
    dm = pipe.diffusion_model
    transformer = dm.transformer
    scheduler = dm.scheduler
    ctx_t = torch.tensor(float(dm.config.context_noise), device=device, dtype=dtype)
    ctx_idx = torch.argmin((scheduler._full_timesteps - ctx_t.float()).abs())
    sigma_ctx = float(scheduler._full_sigmas[ctx_idx])
    # Probe points: the solver schedule (1000 and the 450-warped 803) plus
    # the context forward (t=128) the deploy gate also rescales.
    probe_ts: list[tuple[int, Tensor, float]] = [
        (
            int(t),
            scheduler.denoising_step_list[i].to(device=device, dtype=dtype),
            float(scheduler.denoising_sigmas[i]),
        )
        for i, t in enumerate(scheduler.denoising_step_list.tolist())
    ] + [(int(ctx_t), ctx_t, sigma_ctx)]

    # Style LoRA at deploy semantics (train_style_corrector's setup): frozen
    # weights, per-chunk scale toggling; NO corrector — the gate measures
    # the uncorrected styled drift gap the corrector is premised on.
    network = unwrap_compiled(transformer.network)
    network.requires_grad_(False)
    style_names = apply_lora(network, rank=STYLE_RANK, targets=LORA_TARGETS)
    load_lora(network, STYLE_LORA)
    patch_functional_attention()
    print(
        f"style r{STYLE_RANK} on {len(style_names)} projections | "
        f"{len(pairs)} (clip, style) pairs | depths {DEPTHS} x "
        f"{CELLS_PER_DEPTH}/depth, {M_NOISE} seeds, "
        f"timesteps {[t for t, _, _ in probe_ts]}",
        flush=True,
    )

    # One long-lived cache; probes never replay chunk 0 (start >= 1), so
    # the image/mask fields can come from any clip.
    set_lora_scale(network, 0.0)
    first_uuid = pairs[0]["uuid"]
    cache = pipe.initialize_cache_from_embeddings(
        text_embeddings=prompt_emb[clip_key(first_uuid)],
        image_embeddings=assets["image_embeddings"][first_uuid],
    )
    tc = cache.transformer_cache
    _cur_text = [clip_key(first_uuid)]

    def swap_text(key: str) -> None:
        """Rebuild the cross-attn text KV at BASE weights (deploy truth)."""
        if _cur_text[0] == key:
            return
        set_lora_scale(network, 0.0)
        pipe.replace_text_from_embeddings(cache, prompt_emb[key])
        _cur_text[0] = key

    def replay(
        latents: list[Tensor],
        hdmaps: list[Tensor],
        upto: int,
        swap_at: int,
        slug: str,
        uuid: str,
    ) -> None:
        """Rebuild the KV window for chunks ``upto - 3 .. upto - 1``.

        The trainer's semantics: pre-swap chunks commit at base weights
        under the clip's own prompt; chunks at/after ``swap_at`` commit at
        style scale 1 under the style prompt.
        """
        start = max(0, upto - REPLAY_CHUNKS)
        reset_history(tc)
        for bc in tc.network_cache.block_caches:
            bc.self_attn._prev_chunk_idx = start - 1
        swapped = swap_at <= start
        swap_text(slug if swapped else clip_key(uuid))
        for j in range(start, upto):
            if j == swap_at and not swapped:
                swap_text(slug)
                swapped = True
            set_lora_scale(network, 1.0 if j >= swap_at else 0.0)
            g = torch.Generator(device=device).manual_seed(CONTEXT_NOISE_SEED + j)
            noisy = scheduler.add_noise(latents[j], ctx_t, rng=g)
            tc.start(j)
            transformer.finalize_kv_cache(
                noisy_latent=noisy, timestep=ctx_t, cache=tc, input=hdmaps[j]
            )
            tc.finalize(j)

    agg = {
        t: {"alphas": [], "alphas_ub": [], "rels": [], "rels_x0": []}
        for t, _, _ in probe_ts
    }
    by_depth: dict[int, dict[int, dict[str, list[float]]]] = {}
    for p in pairs:
        uuid, slug = p["uuid"], p["slug"]
        for k, a, b, depth in select_cells(p, rng):
            end = k + 1
            lat_d = [x.to(device, dtype) for x in p["full"][a][:end]]
            lat_r = [x.to(device, dtype) for x in p["full"][b][:end]]
            hd = [x.to(device, dtype) for x in p["hdmaps"][:end]]
            x0 = lat_d[k]

            z_ts: dict[int, list[Tensor]] = {}
            for t_idx, (_, _, sig) in enumerate(probe_ts):
                z_ts[t_idx] = []
                for m in range(M_NOISE):
                    g = torch.Generator(device=device).manual_seed(
                        900_000 + 10_000 * k + 100 * t_idx + m
                    )
                    eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
                    z_ts[t_idx].append((1 - sig) * x0 + sig * eps)

            preds: dict[str, dict[int, list[Tensor]]] = {}
            for name, (lat, swap) in (
                ("drift", (lat_d, a)),
                ("clean", (lat_r, b)),
            ):
                replay(lat, hd, k, swap, slug, uuid)
                # The probed chunk is in-window on both branches (hold
                # depth >= 1): style text + scale 1, even when the swap
                # happened AT k (depth-1 references) and the replay loop
                # never reached it.
                swap_text(slug)
                set_lora_scale(network, 1.0)
                tc.start(k)
                with functional_attention():
                    preds[name] = {
                        t_idx: [
                            transformer.predict_flow(
                                noisy_latent=z, timestep=tt, cache=tc, input=hd[k]
                            ).float()
                            for z in z_ts[t_idx]
                        ]
                        for t_idx, (_, tt, _) in enumerate(probe_ts)
                    }
                tc.finalize(k)

            line = [f"{uuid[:8]}__{slug} k={k:2d} d=+{depth:2d} (ref +{k - b + 1})"]
            for t_idx, (t_lab, _, sig) in enumerate(probe_ts):
                deltas = torch.stack(
                    [
                        d_ - c_
                        for d_, c_ in zip(preds["drift"][t_idx], preds["clean"][t_idx])
                    ]
                )
                m = deltas.shape[0]
                bias = deltas.mean(dim=0)
                bias_sq = bias.square().sum().item()
                sq_dev = (deltas - bias).square().sum(dim=tuple(range(1, deltas.ndim)))
                var = sq_dev.mean().item()
                var_ub = sq_dev.sum().item() / (m - 1)
                bias_sq_ub = max(0.0, bias_sq - var_ub / m)
                drift_norm = (
                    torch.stack(preds["drift"][t_idx]).flatten(1).norm(dim=1).mean()
                )
                rel = (
                    deltas.flatten(1).norm(dim=1).mean() / (drift_norm + 1e-12)
                ).item()
                x0_norm = (
                    torch.stack(
                        [
                            z.float() - sig * v
                            for z, v in zip(z_ts[t_idx], preds["drift"][t_idx])
                        ]
                    )
                    .flatten(1)
                    .norm(dim=1)
                    .mean()
                )
                rel_x0 = (
                    sig * deltas.flatten(1).norm(dim=1).mean() / (x0_norm + 1e-12)
                ).item()
                a_hat = bias_sq / (bias_sq + var + 1e-12)
                a_ub = bias_sq_ub / (bias_sq_ub + var_ub + 1e-12)
                agg[t_lab]["alphas"].append(a_hat)
                agg[t_lab]["alphas_ub"].append(a_ub)
                agg[t_lab]["rels"].append(rel)
                agg[t_lab]["rels_x0"].append(rel_x0)
                slot = by_depth.setdefault(depth, {}).setdefault(
                    t_lab, {"alphas_ub": [], "rels": []}
                )
                slot["alphas_ub"].append(a_ub)
                slot["rels"].append(rel)
                line.append(f"t={t_lab:4d} a*={a_hat:.3f}/{a_ub:.3f} rel_v={rel:.4f}")
            print(" | ".join(line), flush=True)

    summary = {
        "style_lora": str(STYLE_LORA),
        "config": {
            "depths": list(DEPTHS),
            "cells_per_depth": CELLS_PER_DEPTH,
            "ref_max": REF_MAX,
            "m_noise": M_NOISE,
            "seed": SEED,
            "styles": sorted({p["slug"] for p in pairs}),
            "clips": len({p["uuid"] for p in pairs}),
        },
        "per_timestep": {
            str(t): {
                "alpha_star": sum(v["alphas"]) / len(v["alphas"]),
                "alpha_star_unbiased": sum(v["alphas_ub"]) / len(v["alphas_ub"]),
                "rel_v": sum(v["rels"]) / len(v["rels"]),
                "rel_x0": sum(v["rels_x0"]) / len(v["rels_x0"]),
                "cells": len(v["alphas"]),
            }
            for t, v in agg.items()
            if v["alphas"]
        },
        "per_depth": {
            str(d): {
                str(t): {
                    "alpha_star_unbiased": sum(s["alphas_ub"]) / len(s["alphas_ub"]),
                    "rel_v": sum(s["rels"]) / len(s["rels"]),
                    "cells": len(s["alphas_ub"]),
                }
                for t, s in sorted(slots.items())
            }
            for d, slots in sorted(by_depth.items())
        },
        "photoreal_gate_alpha": {str(int(t)): a for t, a in PHOTOREAL_GATE.items()},
    }
    # The deploy-facing profile (GATE_ALPHA_JSON consumes this entry):
    # unbiased alpha*, the same estimator the photoreal 0.96/0.667 came from.
    summary["gate_alpha"] = {
        t: round(v["alpha_star_unbiased"], 3)
        for t, v in summary["per_timestep"].items()
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n========== Omnidreams STYLE gate (styled drift pairs, v-space) ==========")
    for t, v in summary["per_timestep"].items():
        print(
            f"t={t:>4s}: alpha* {v['alpha_star']:.3f} (unbiased"
            f" {v['alpha_star_unbiased']:.3f}) | rel_v {v['rel_v']:.4f}"
            f" rel_x0 {v['rel_x0']:.4f} | {v['cells']} cells"
        )
    for d, slots in summary["per_depth"].items():
        parts = [
            f"t={t} a*_ub={s['alpha_star_unbiased']:.3f} rel_v={s['rel_v']:.3f}"
            for t, s in slots.items()
        ]
        print(f"depth +{d:>2s}: " + " | ".join(parts))

    print("\n========== style profile vs deployed photoreal GATE_ALPHA ==========")
    deviating: list[str] = []
    for t, a_style in summary["gate_alpha"].items():
        a_photo = _nearest_alpha(float(t))
        delta = a_style - a_photo
        if abs(delta) >= DEV_BAR:
            deviating.append(f"t={t} {a_photo:.3f}->{a_style:.3f} ({delta:+.3f})")
        print(
            f"t={t:>4s}: style {a_style:.3f} | photoreal deploy {a_photo:.3f}"
            f" | delta {delta:+.3f}"
        )
    mean_rel = sum(r for v in agg.values() for r in v["rels"]) / max(
        1, sum(len(v["rels"]) for v in agg.values())
    )
    if mean_rel < 0.01:
        rec = "styled drift gap ~zero -> nothing to gate differently. STOP."
    elif deviating:
        rec = (
            f"override for styled worlds: GATE_ALPHA_JSON={OUT_PATH} "
            f"({'; '.join(deviating)} exceed the {DEV_BAR} bar)"
        )
    else:
        rec = (
            f"keep the photoreal profile (every timestep within {DEV_BAR} "
            "of the deployed values)"
        )
    print(f"RECOMMENDATION: {rec}")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
