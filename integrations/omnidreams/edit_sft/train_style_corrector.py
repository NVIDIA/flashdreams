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

"""Style-drift corrector trainer (Issue 1 fix ladder item 1).

The Clean Forcing recipe (``drift_correction/train_v2.py``) retargeted at
the style-skin long-hold blur: a SEPARATE rank-16 LoRA on the self-attn
q/k/v/output projections, trained WITH the style LoRA active, that maps
late-window styled predictions back toward the early-window styled
manifold. Like the shipped drift corrector, it corrects the *flow
prediction under the live (drifted) KV history* — every denoise-step
``predict_flow`` and the KV-commit context forward — not the latent
post hoc, so the checkpoint deploys through the exact same hook
(``omnidreams/_drift_corrector.py``: same 4 targets, same rank, same
``{"lora": {2i: A_i, 2i+1: B_i}}`` load order).

Counterfactual pairs come from ``gen_style_drift_pairs.py`` branches: for
a probe chunk ``k``, the DRIFTED branch swapped ``DRIFT_MIN..DRIFT_MAX``
chunks earlier (its window has compounded into blur) and the REFERENCE
branch swapped ``1..REF_MAX`` chunks earlier (the model's own clean styled
output at the SAME absolute index, same HDMap, same seed — a reference
that never round-trips through JoyAI, sidestepping the VAE-blur bias).
Per pooled sample, at a matched noisy state ``z_t`` built from the drifted
chunk::

    v_clean = style-LoRA'd model | reference history, corrector 0
    v_base  = style-LoRA'd model | drifted history,   corrector 0
    v_corr  = style-LoRA'd model | drifted history,   corrector 1  (grad)
    L_dag   = ||v_corr - v_clean||^2 / ||v_clean - v_base||^2

plus the ``train_v2`` drift-contraction term (weight :data:`CW_LOSS`): the
corrected chunk-``k`` prediction is committed into chunk ``k+1``'s KV with
grad (``record_kv`` / ``inject_kv``) and chunk ``k+1``'s gap to its clean
styled teacher is penalized. No-op episodes (:data:`NOOP_PROB`) pin the
corrector to identity on unstyled and on early-window styled history, so
the always-on deploy hook is safe outside and at the start of style
windows.

Style-LoRA semantics mirror deploy (``_edit_lora`` / ``TextEditLoRA``):
text KV is always rebuilt at base weights; in-window forwards and commits
run at style scale 1; pre-swap replay chunks run at base. The corrector
trains at scale 1 and deploys at ``alpha*(t) * gain`` (the hook's gate;
sweep ``drift_corrector_gain`` at eval, as for the photoreal corrector).

Deploy note: compose with the pre-merged style LoRA via the corrector's
UNFUSED path (``DRIFT_CORRECTOR_UNFUSED=1``), attaching the
``TextEditLoRA`` BEFORE ``apply_drift_corrector``. The default pre-merged
corrector path and ``TextEditLoRA.set_active`` both assume exclusive
ownership of the projection ``weight.data`` and corrupt each other.

Run from the flashdreams repo root (resumable: re-run the same command)::

    STEPS=1000 .venv/bin/python \
        integrations/omnidreams/edit_sft/train_style_corrector.py
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
import torch.nn as nn
from _host import CONTEXT_NOISE_SEED, build_pipeline, reset_history
from _lora import (
    DEFAULT_TARGETS,
    LoRALinear,
    apply_lora,
    load_lora,
    lora_parameters,
    unwrap_compiled,
)
from _train_attn import (
    functional_attention,
    inject_kv,
    patch_functional_attention,
    record_kv,
)
from omnidreams._edit_lora import _LORA_TARGETS as LORA_TARGETS
from omnidreams.runner import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH
from style_prompts import clip_key
from torch import Tensor

## Training configuration

BASE = Path("integrations/omnidreams/edit_sft")
OUT_DIR = BASE / "outputs"
PAIRS_DIR = Path(os.environ.get("PAIRS_DIR", str(OUT_DIR / "style_drift_pairs")))

STYLE_LORA = Path(os.environ.get("STYLE_LORA", str(OUT_DIR / "lora_style_step1600.pt")))
"""Style-skin checkpoint the corrector trains under (must be the one whose
branches ``gen_style_drift_pairs.py`` rolled)."""

STYLE_RANK = int(os.environ.get("STYLE_RANK", "64"))
"""Rank of the style checkpoint (the eval RANK-must-match gotcha)."""

CKPT = Path(os.environ.get("CKPT", str(OUT_DIR / "lora_style_corrector.pt")))
"""Output checkpoint (corrector LoRA + optimizer + step); loads through
``omnidreams/_drift_corrector.py`` unchanged."""

INIT = os.environ.get("INIT", "")
"""Optional warm-start corrector checkpoint (same format; the photoreal
``drift_correction/outputs/lora_v2_v3_valpeak.pt`` is shape-compatible).
Empty = scratch."""

STEPS = int(os.environ.get("STEPS", "1000"))
LR = float(os.environ.get("LR", "2e-4"))
WARMUP = 40
GRAD_CLIP = 1.0
EVAL_EVERY = 100
SAVE_EVERY = 200

RANK = int(os.environ.get("RANK", "16"))
"""Corrector rank. The deploy hook hardcodes 16 — other values need a
matching ``_drift_corrector._LORA_RANK`` change."""

SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "0"))
"""When > 0: step-tagged snapshots every SNAP_EVERY steps plus a running
val-peak snapshot (``<ckpt>_valpeak.pt``), the ``train_v2`` convention."""

CW_LOSS = float(os.environ.get("CW_LOSS", "0.5"))
"""Drift-contraction weight (the ``train_v2`` paper-final value)."""

DRIFT_WEIGHT = os.environ.get("DRIFT_WEIGHT", "0") == "1"
"""Per-token loss weighting by drift magnitude. The style drift
concentrates in background tokens that the uniform DAG numerator
under-weights (roadside texture stays soft after correction); with the
knob on, each token's squared error in the DAG and contraction numerators
is multiplied by that token's clean-vs-base gap norm (mean-1 per frame,
clamped to [0.5, 4.0], detached). 0 = exact unweighted behavior."""

TOKENS_PER_FRAME = (DEFAULT_VIDEO_HEIGHT // 16) * (DEFAULT_VIDEO_WIDTH // 16)
"""Patch-token grid per latent frame (VAE /8 x patchify /2): 44*80."""

NOOP_PROB = float(os.environ.get("NOOP_PROB", "0.15"))
"""Probability of a no-op episode: corrector at scale 1 must match the
corrector-free prediction on unstyled history (50%) or on early-window
styled history (50%) — keeps the always-on deploy gate safe off the
drifted manifold."""

NOOP_WEIGHT = float(os.environ.get("NOOP_WEIGHT", "1.0"))

DRIFT_MIN = int(os.environ.get("DRIFT_MIN", "8"))
DRIFT_MAX = int(os.environ.get("DRIFT_MAX", "20"))
"""Style-hold depth (chunks since swap, 1-based) of the drifted branch:
+8..+20 spans "structure still holds" through "washed out" on the v3
skin."""

REF_MAX = int(os.environ.get("REF_MAX", "4"))
"""Max hold depth of the reference branch (+1..+4 = the clean manifold)."""

REPLAY_CHUNKS = int(os.environ.get("REPLAY_CHUNKS", "3"))
"""History chunks replayed before a probe. 3 chunks = the full
``window_size_t=6`` / ``len_t=2`` KV window, so the visible state equals a
full-history replay (the ``train_style_sft`` mid-stream-start setting)."""

REL_V_EXCLUDE = 0.8
"""Drop cells whose clean-vs-base gap exceeds this fraction of ``|v_base|``
(unpredictable content difference, the ``train_v2`` convention)."""

ALPHA_STAR = tuple(
    float(x) for x in os.environ.get("ALPHA_STAR", "0.96,0.667").split(",")
)
"""Per-timestep sampling weights (the deploy gate's relative profile)."""

N_VAL_CLIPS = int(os.environ.get("N_VAL_CLIPS", "1"))
"""Clips (last of the sources manifest with pairs) held out for val."""

STYLES = frozenset(s for s in os.environ.get("STYLES", "").split(",") if s)
"""Optional slug filter; empty = every style found in the pairs dir."""

SEED = int(os.environ.get("SEED", "0"))


def drift_weights(gap: Tensor) -> Tensor:
    """Per-token weights from a clean-vs-base flow gap ``[..., T*HW, C]``.

    Weight = the token's channel-norm of ``gap``, normalized to mean 1
    over each frame's tokens, clamped to [0.5, 4.0], detached — so tokens
    the drift actually moved dominate the numerator without changing the
    loss scale or blowing up on near-uniform frames.
    """
    w = gap.detach().norm(dim=-1, keepdim=True)
    assert w.shape[-2] % TOKENS_PER_FRAME == 0, tuple(gap.shape)
    lead = w.shape[:-2]
    w = w.reshape(*lead, -1, TOKENS_PER_FRAME, 1)
    w = w / (w.mean(dim=-2, keepdim=True) + 1e-8)
    return w.reshape(*lead, -1, 1).clamp(0.5, 4.0)


def load_pairs() -> list[dict]:
    """Load the branch corpus and precompute the probe cells per pair.

    A cell ``(k, a, b)`` probes absolute chunk ``k`` with the drifted
    branch swapped at ``a`` (depth ``k - a + 1`` in
    ``[DRIFT_MIN, DRIFT_MAX]``) against the reference branch swapped at
    ``b`` (depth in ``[1, REF_MAX]``); ``k + 1`` must exist for the
    contraction successor.
    """
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
        offsets = sorted(full)
        cells: list[tuple[int, int, int]] = []
        for k in range(1 + REPLAY_CHUNKS, n - 1):
            refs = [b for b in offsets if 1 <= k - b + 1 <= REF_MAX]
            drifts = [a for a in offsets if DRIFT_MIN <= k - a + 1 <= DRIFT_MAX]
            cells += [(k, a, b) for a in drifts for b in refs]
        if not cells:
            print(f"skip {path.name}: no probe cells in range", flush=True)
            continue
        pairs.append(
            {
                "uuid": uuid,
                "slug": slug,
                "n_chunks": n,
                "base": base["latents"],
                "hdmaps": base["hdmaps"],
                "full": full,
                "offsets": offsets,
                "cells": cells,
            }
        )
    assert pairs, f"no styled branch files under {PAIRS_DIR}"
    return pairs


def main() -> None:
    """Run the counterfactual style-corrector training loop."""
    rng = np.random.default_rng(SEED)
    pairs = load_pairs()
    manifest = json.loads((OUT_DIR / "sources" / "manifest.json").read_text())
    uuids_in_order = [
        e["uuid"] for e in manifest if any(p["uuid"] == e["uuid"] for p in pairs)
    ]
    assert len(uuids_in_order) > N_VAL_CLIPS, (
        f"{len(uuids_in_order)} clips can't spare {N_VAL_CLIPS} for val"
    )
    val_uuids = set(uuids_in_order[-N_VAL_CLIPS:])
    train_ids = [i for i, p in enumerate(pairs) if p["uuid"] not in val_uuids]
    val_ids = [i for i, p in enumerate(pairs) if p["uuid"] in val_uuids]

    prompt_emb = torch.load(
        OUT_DIR / "style_embeddings.pt", map_location="cpu", weights_only=False
    )
    assets = torch.load(
        OUT_DIR / "style_clip_assets.pt", map_location="cpu", weights_only=False
    )

    pipe = build_pipeline(with_oneshot_encoders=False)
    assert pipe.V_group is None, "single-GPU trainer; run without CP"
    device = pipe.device
    dtype = torch.bfloat16
    dm = pipe.diffusion_model
    transformer = dm.transformer
    scheduler = dm.scheduler
    timesteps = scheduler.denoising_step_list  # [1000, 450] on the chunk2 host
    sigmas = scheduler.denoising_sigmas
    n_steps = int(timesteps.shape[0])
    t_probs = np.array(ALPHA_STAR[:n_steps]) / sum(ALPHA_STAR[:n_steps])
    ctx_t = torch.tensor(float(dm.config.context_noise), device=device, dtype=dtype)
    ctx_idx = torch.argmin((scheduler._full_timesteps - ctx_t.float()).abs())
    sigma_ctx = scheduler._full_sigmas[ctx_idx].to(dtype)

    # Style LoRA (frozen, deploy-window semantics) + corrector on top. The
    # corrector wraps the inner base linears of the 4 self-attn targets in
    # the SAME named_modules walk order as the deploy hook's _apply_lora,
    # so the saved index order loads through _drift_corrector unchanged.
    network = unwrap_compiled(transformer.network)
    network.requires_grad_(False)
    style_names = apply_lora(network, rank=STYLE_RANK, targets=LORA_TARGETS)
    load_lora(network, STYLE_LORA)
    for p in lora_parameters(network):  # style-only at this point
        p.requires_grad_(False)
    style_mods = [network.get_submodule(n) for n in style_names]

    corr_mods: list[LoRALinear] = []
    for mname, module in list(network.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        if not any(mname.endswith(t) for t in DEFAULT_TARGETS):
            continue
        inner = module.base
        assert isinstance(inner, nn.Linear)
        corr = LoRALinear(inner, rank=RANK).to(inner.weight.device)
        corr.scale = 0.0
        module.base = corr
        corr_mods.append(corr)
    assert 2 * len(corr_mods) == len(style_mods), (
        f"{len(corr_mods)} corrector wraps vs {len(style_mods)} style wraps"
    )
    corr_params = [p for m in corr_mods for p in (m.A.weight, m.B.weight)]
    if RANK != 16:
        print(
            f"WARNING: RANK={RANK} will not load through the deploy hook "
            "(_drift_corrector._LORA_RANK is 16).",
            flush=True,
        )
    print(
        f"style r{STYLE_RANK} on {len(style_mods)} projections (frozen) | "
        f"corrector r{RANK} on {len(corr_mods)} projections, "
        f"{sum(p.numel() for p in corr_params) / 1e6:.2f}M params | "
        f"drift-weight {'on' if DRIFT_WEIGHT else 'off'} | "
        f"{len(pairs)} pairs ({len(val_ids)} val), cells/pair "
        f"{[len(p['cells']) for p in pairs]}",
        flush=True,
    )
    patch_functional_attention()
    opt = torch.optim.AdamW(corr_params, lr=LR)

    def set_scales(style: float, corr: float) -> None:
        """Independent runtime gains for the two LoRA stacks."""
        for m in style_mods:
            m.scale = style
        for m in corr_mods:
            m.scale = corr

    # One long-lived cache; probes never replay chunk 0 with content that
    # matters (start >= 1), so the image/mask fields can come from any clip.
    set_scales(0.0, 0.0)
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
        set_scales(0.0, 0.0)
        pipe.replace_text_from_embeddings(cache, prompt_emb[key])
        _cur_text[0] = key

    start_step = 0
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        for i, p in enumerate(corr_params):
            p.data.copy_(state["lora"][i].to(p.device, p.dtype))
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        rng = np.random.default_rng(state["rng_seed"])
        print(f"RESUMED from {CKPT} at step {start_step}", flush=True)
    elif INIT:
        state = torch.load(INIT, map_location="cpu", weights_only=False)
        for i, p in enumerate(corr_params):
            p.data.copy_(state["lora"][i].to(p.device, p.dtype))
        print(f"warm start from {INIT}", flush=True)

    def save(step: int, path: Path = CKPT) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "lora": {i: p.detach().cpu() for i, p in enumerate(corr_params)},
                "opt": opt.state_dict(),
                "step": step,
                "rng_seed": SEED + step,
            },
            tmp,
        )
        tmp.replace(path)

    def replay(
        latents: list[Tensor],
        hdmaps: list[Tensor],
        upto: int,
        swap_at: int | None,
        slug: str | None,
        uuid: str,
        corr: float,
    ) -> None:
        """Rebuild the KV window for chunks ``max(0, upto - 3) .. upto - 1``.

        Pre-swap chunks commit at base weights under the clip's own prompt;
        chunks at/after ``swap_at`` commit at style scale 1 under the style
        prompt (the deploy window semantics), with the corrector at ``corr``
        (1 for the corrected branch — deploy commits corrected — else 0).
        """
        start = max(0, upto - REPLAY_CHUNKS)
        reset_history(tc)
        for bc in tc.network_cache.block_caches:
            bc.self_attn._prev_chunk_idx = start - 1
        swapped = swap_at is not None and swap_at <= start
        assert slug is not None or swap_at is None
        swap_text(slug if (swapped and slug is not None) else clip_key(uuid))
        with torch.no_grad():
            for j in range(start, upto):
                if swap_at is not None and j == swap_at and not swapped:
                    assert slug is not None
                    swap_text(slug)
                    swapped = True
                in_win = swap_at is not None and j >= swap_at
                set_scales(1.0 if in_win else 0.0, corr if in_win else 0.0)
                g = torch.Generator(device=device).manual_seed(CONTEXT_NOISE_SEED + j)
                noisy = scheduler.add_noise(latents[j], ctx_t, rng=g)
                tc.start(j)
                transformer.finalize_kv_cache(
                    noisy_latent=noisy, timestep=ctx_t, cache=tc, input=hdmaps[j]
                )
                tc.finalize(j)

    def predict_v(z_t: Tensor, t_idx: int, hdmap: Tensor) -> Tensor:
        """One functional-attention (non-writing) forward -> fp32 flow."""
        t = timesteps[t_idx].to(device=device, dtype=dtype)
        with functional_attention():
            flow = transformer.predict_flow(
                noisy_latent=z_t, timestep=t, cache=tc, input=hdmap
            )
        return flow.float()

    def make_zt(x0: Tensor, t_idx: int, rng_: np.random.Generator) -> Tensor:
        sig = sigmas[t_idx].to(dtype)
        g = torch.Generator(device=device).manual_seed(int(rng_.integers(2**31)))
        return (1 - sig) * x0 + sig * torch.randn(
            x0.shape, device=device, dtype=dtype, generator=g
        )

    excl = {c: {"tested": 0, "excluded": 0} for c in range(len(pairs))}

    def styled_losses(
        c: int, grad: bool, rng_: np.random.Generator
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """One pooled cell -> (dag, contraction, r_target sq, unweighted dag).

        Under :data:`DRIFT_WEIGHT` the dag/contraction numerators are
        drift-weighted per token; the unweighted dag ratio is returned
        alongside (equal to dag when off) so val stays comparable across
        weighted and unweighted runs. Returns ``None`` for excluded
        (unpredictable) cells; callers redraw.
        """
        p = pairs[c]
        uuid, slug = p["uuid"], p["slug"]
        k, a, b = p["cells"][int(rng_.integers(len(p["cells"])))]
        t_idx = int(rng_.choice(n_steps, p=t_probs))
        t2_idx = int(rng_.choice(n_steps, p=t_probs))
        end = k + 2
        lat_d = [x.to(device, dtype) for x in p["full"][a][:end]]
        lat_r = [x.to(device, dtype) for x in p["full"][b][:end]]
        hd = [x.to(device, dtype) for x in p["hdmaps"][:end]]
        z_t = make_zt(lat_d[k], t_idx, rng_)
        z_t2 = make_zt(lat_d[k + 1], t2_idx, rng_)

        with torch.no_grad():
            # Teacher: the clean styled manifold at k and (for the
            # contraction target) at k+1 — reference-branch history.
            replay(lat_r, hd, k, b, slug, uuid, corr=0.0)
            set_scales(1.0, 0.0)
            tc.start(k)
            v_clean = predict_v(z_t, t_idx, hd[k])
            tc.finalize(k)
            replay(lat_r, hd, k + 1, b, slug, uuid, corr=0.0)
            set_scales(1.0, 0.0)
            tc.start(k + 1)
            v_clean2 = predict_v(z_t2, t2_idx, hd[k + 1])
            tc.finalize(k + 1)
            # Base: the drifted branch without corrector.
            replay(lat_d, hd, k + 1, a, slug, uuid, corr=0.0)
            set_scales(1.0, 0.0)
            tc.start(k + 1)
            v_base2 = predict_v(z_t2, t2_idx, hd[k + 1])
            tc.finalize(k + 1)
            replay(lat_d, hd, k, a, slug, uuid, corr=0.0)
            set_scales(1.0, 0.0)
            tc.start(k)
            v_base = predict_v(z_t, t_idx, hd[k])
            tc.finalize(k)
        r_sq = (v_clean - v_base).square().sum()
        r2_sq = (v_clean2 - v_base2).square().sum()
        w = drift_weights(v_clean - v_base) if DRIFT_WEIGHT else None
        w2 = drift_weights(v_clean2 - v_base2) if DRIFT_WEIGHT else None

        excl[c]["tested"] += 1
        rel_v = (r_sq.sqrt() / (v_base.norm() + 1e-9)).item()
        if rel_v > REL_V_EXCLUDE:
            excl[c]["excluded"] += 1
            return None

        # Corrected branch: drifted history committed WITH the corrector
        # (deploy commits corrected), probe + commit + successor probe.
        replay(lat_d, hd, k, a, slug, uuid, corr=1.0)
        set_scales(1.0, 1.0)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        tc.start(k)
        with ctx:
            v_corr = predict_v(z_t, t_idx, hd[k])
            err = (v_corr - v_clean).square()
            dag_raw = err.sum() / (r_sq + 1e-8)
            dag = (err * w).sum() / (r_sq + 1e-8) if w is not None else dag_raw
            # Commit chunk k's corrected prediction with grad: recorded
            # functional KV + a numerically identical no-grad buffer twin
            # (the train_v2 contraction machinery).
            sig = sigmas[t_idx].to(dtype)
            x0_corr = (z_t.float() - float(sig) * v_corr).to(dtype)
            g = torch.Generator(device=device).manual_seed(CONTEXT_NOISE_SEED + k)
            eps = torch.randn(x0_corr.shape, device=device, dtype=dtype, generator=g)
            noisy_corr = (1 - sigma_ctx) * x0_corr + sigma_ctx * eps
            recorded: list = []
            with record_kv(recorded), functional_attention():
                transformer.predict_flow(
                    noisy_latent=noisy_corr, timestep=ctx_t, cache=tc, input=hd[k]
                )
        with torch.no_grad():  # buffer twin write + index advance
            transformer.finalize_kv_cache(
                noisy_latent=noisy_corr.detach(), timestep=ctx_t, cache=tc, input=hd[k]
            )
        tc.finalize(k)
        tc.start(k + 1)
        with ctx:
            with inject_kv(recorded):
                v_con = predict_v(z_t2, t2_idx, hd[k + 1])
            err2 = (v_con - v_clean2).square()
            con = (err2 * w2 if w2 is not None else err2).sum() / (r2_sq + 1e-8)
        tc.finalize(k + 1)
        return dag, con, r_sq, dag_raw

    def noop_loss(c: int, grad: bool, rng_: np.random.Generator) -> Tensor:
        """Corrector-at-1 vs corrector-at-0 gap off the drifted manifold."""
        p = pairs[c]
        uuid, slug, n = p["uuid"], p["slug"], p["n_chunks"]
        styled = bool(rng_.random() < 0.5)
        if styled:
            b = int(p["offsets"][int(rng_.integers(len(p["offsets"])))])
            depth = int(rng_.integers(1, REF_MAX + 1))
            k = min(b + depth - 1, n - 1)
            latents, swap_at, key = p["full"][b], b, slug
        else:
            k = int(rng_.integers(1 + REPLAY_CHUNKS, n))
            latents, swap_at, key = p["base"], None, None
        t_idx = int(rng_.choice(n_steps, p=t_probs))
        lat = [x.to(device, dtype) for x in latents[: k + 1]]
        hd = [x.to(device, dtype) for x in p["hdmaps"][: k + 1]]
        z_t = make_zt(lat[k], t_idx, rng_)
        sty = 1.0 if styled else 0.0
        replay(lat, hd, k, swap_at, key, uuid, corr=0.0)
        tc.start(k)
        with torch.no_grad():
            set_scales(sty, 0.0)
            v0 = predict_v(z_t, t_idx, hd[k])
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            set_scales(sty, 1.0)
            v1 = predict_v(z_t, t_idx, hd[k])
            loss = (v1 - v0).square().sum() / (v0.square().sum() + 1e-8)
        tc.finalize(k)
        return loss

    def draw_styled(
        ids: list[int], grad: bool, rng_: np.random.Generator
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Redraw until a non-excluded cell is sampled."""
        while True:
            out = styled_losses(int(rng_.choice(ids)), grad, rng_)
            if out is not None:
                return out

    @torch.no_grad()
    def val_metrics(n: int = 12) -> tuple[float, float, float, float]:
        vrng = np.random.default_rng(1234)
        s_dag = s_con = s_noop = s_raw = 0.0
        for _ in range(n):
            dag, con, _, raw = draw_styled(val_ids, False, vrng)
            s_dag += dag.item()
            s_con += con.item()
            s_raw += raw.item()
        n_noop = max(1, n // 2)
        for _ in range(n_noop):
            s_noop += noop_loss(int(vrng.choice(val_ids)), False, vrng).item()
        return 1 - s_dag / n, 1 - s_con / n, s_noop / n_noop, 1 - s_raw / n

    torch.set_grad_enabled(True)
    best_vd = float("-inf")
    for step in range(start_step + 1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        if rng.random() < NOOP_PROB:
            noop = noop_loss(int(rng.choice(train_ids)), True, rng)
            loss = NOOP_WEIGHT * noop
            terms = f"noop {noop.item():.4f}"
        else:
            dag, con, r_sq, _ = draw_styled(train_ids, True, rng)
            loss = dag + CW_LOSS * con
            terms = f"dag {dag.item():.4f} con {con.item():.4f} |r|^2 {r_sq.item():.1f}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corr_params, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            vd, vc, vn, vraw = val_metrics()
            excl_str = " ".join(
                f"c{c}:{v['excluded']}/{v['tested']}" for c, v in excl.items()
            )
            raw_str = f" (unw {vraw:+.3f})" if DRIFT_WEIGHT else ""
            print(
                f"step {step:5d} | {terms} | val dag-R^2 {vd:+.3f}{raw_str} "
                f"con-R^2 {vc:+.3f} noop {vn:.4f} | excluded {excl_str}",
                flush=True,
            )
            if SNAP_EVERY and vd > best_vd:
                best_vd = vd
                save(step, CKPT.with_name(f"{CKPT.stem}_valpeak.pt"))
                print(
                    f"val-peak snapshot at step {step} (dag-R^2 {vd:+.3f})",
                    flush=True,
                )
        if step % SAVE_EVERY == 0 or step == STEPS:
            save(step)
        if SNAP_EVERY and step % SNAP_EVERY == 0:
            save(step, CKPT.with_name(f"{CKPT.stem}_step{step}.pt"))

    vd, vc, vn, vraw = val_metrics(24)
    raw_str = f" (unw {vraw:+.3f})" if DRIFT_WEIGHT else ""
    print(
        f"TRAIN-STYLE-CORRECTOR-DONE | final val dag-R^2 {vd:+.3f}{raw_str} "
        f"con-R^2 {vc:+.3f} noop {vn:.4f} | saved {CKPT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
