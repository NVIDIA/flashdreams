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

"""Edit-timestamped style-SFT trainer (Tier-2b of the live-edit hack).

Teaches the realtime distilled student (2-step causal Cosmos DiT) to apply
a global visual style mid-stream when the text prompt is swapped,
imitating the offline JoyAI restyles (``PLAN.md`` recipe A, restricted to
the global-style instruction bank). Supervision is teacher-forced
flow matching in latent space on edit-timestamped clips:

- **Context build**: chunks before the sampled swap chunk ``k`` are the
  SOURCE rollout's own latents (``generate_sources.py``), replayed through
  ``finalize_kv_cache`` at the host's ``context_noise`` (the
  ``drift_correction`` ``replay_history`` machinery) under base weights —
  at deploy the LoRA window opens only at the swap. RoPE is absolute and
  the KV window holds ``REPLAY_CHUNKS`` chunks, so replaying just the
  window prefix reproduces a full rollout's visible state exactly.
- **Edit supervision**: at ``k`` the text KV is rebuilt to the style
  prompt (plain ``replace_text_from_embeddings``, exactly the deploy
  swap); chunks ``[k, k + SPAN)`` are then supervised toward the STYLED
  latents (the JoyAI output encoded by ``precompute_style.py``). States
  stay on-policy within each chunk (the student's own detached flow
  advances the two-step trajectory, mirroring ``scheduler.sample``); the
  v-target at every state is ``(z_t - x0_styled) / sigma``. A 0.5-weighted
  context term (t=128) supervises the same forward whose K/V the commit
  writes, and the STYLED latent is committed into the rolling KV — teacher
  forcing proceeds on edited history, merged-weight commits matching the
  deploy window semantics (``_edit_lora``).
- **No-op regularization**: ``PRE_CHUNKS`` supervised chunks right before
  ``k`` target the SOURCE latents under the clip's own prompt (keeps
  non-edit behavior identical), and with ``NOOP_PROB`` the whole window is
  a no-op (swap to the clip's own prompt, source targets) so the style
  must come from the text KV, not the weights.
- **v4 self-consistency maintenance** (:data:`V4_MAINT`, default on): the
  v3 maintenance episodes (``MAINT_PROB``: styled history at scale 1,
  styled targets) taught a SOFT steady state — their JoyAI targets are RGB
  re-encoded through the VAE, so the fixed point the LoRA converges to is
  blurred and the drift corrector can only pull back to that soft point
  (``issues_and_fixes.md`` Issue 1). v4 replaces the JoyAI latents in
  maintenance episodes ONLY with the model's OWN early-window styled
  rollouts — the ``gen_style_drift_pairs.py`` clean reference branches:
  history and targets both come from a branch segment held at most
  :data:`V4_EARLY` chunks past its swap (the pre-drift styled manifold),
  so the steady state supervised is the model's own sharp styled
  continuation. Swap-transition and no-op episodes keep their JoyAI /
  source targets unchanged — they still define the style identity at the
  transition. Pairs without a usable branch segment fall back to the v3
  maintenance path (counted, reported in the DONE line).

Data is gated by the style-mode VLM filter
(``style_pairs/filter_report.json``) when present: ``passed`` pairs train
anywhere, ``early_window_ok`` pairs only within the first
:data:`EARLY_CHUNKS` chunks (~4 s — the window before streaming layout
drift sets in on heavy styles). Without a report every pair on disk
trains, with a warning.

Host mechanics are the ``guidance_distill/train_guidance.py`` ones: eager
pipeline, LoRA r64 on the 8 attention projections (checkpoint loads
through ``omnidreams/_edit_lora.py`` unchanged), functional self-attention
on grad forwards, per-block checkpointing, and every term's backward
immediately after its forward inside ``functional_attention()`` (the
recompute must retake the non-writing path, and later stock forwards
in-place mutate buffers the tape saved). No teacher forward exists here —
the target is a fixed latent — so steps are lighter than Tier-2a's.

VRAM (fits the ~65 GB co-tenant share): bf16 2B DiT ~4 GB + per-episode
KV caches ~19 GB (28 blocks x 6-latent-frame window at 88x160 latents) +
fp32 r64 LoRA/AdamW ~0.5 GB + one grad forward's checkpointed block
inputs ~1.6 GB (immediate per-term backward keeps only one tape alive)
— ~30 GB peak, eager.

Run from the flashdreams repo root (after ``precompute_style.py``)::

    STEPS=1000 .venv/bin/python \
        integrations/omnidreams/edit_sft/train_style_sft.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drift_correction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guidance_distill"))

import numpy as np
import torch
from _host import CONTEXT_NOISE_SEED, build_pipeline
from _lora import (
    apply_lora,
    lora_parameters,
    save_lora,
    set_lora_scale,
    unwrap_compiled,
)
from _train_attn import functional_attention, patch_functional_attention
from style_prompts import DEFAULT_SKIP_STYLES, STYLE_PROMPTS, clip_key
from torch import Tensor
from train_guidance import LORA_TARGETS, checkpoint_blocks

## Training configuration

BASE = Path("integrations/omnidreams/edit_sft")
OUT_DIR = BASE / "outputs"
SRC_DIR = OUT_DIR / "sources"
STYLE_DIR = Path(os.environ.get("STYLE_DIR", str(OUT_DIR / "style_pairs")))
"""Styled-target corpus; ``style_pairs_teacher`` = the teacher-regenerated
targets (``gen_teacher_styled_targets.py``, v5 recipe)."""
LAT_DIR = Path(os.environ.get("LAT_DIR", str(OUT_DIR / "latents")))
"""Latents matching STYLE_DIR (``precompute_style.py`` with the same env)."""
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", "lora_style")
"""Checkpoint filename prefix (``<prefix>_step<N>.pt``)."""
PAIRS_DIR = Path(os.environ.get("PAIRS_DIR", str(OUT_DIR / "style_drift_pairs")))
"""Clean-branch corpus (``gen_style_drift_pairs.py``) for v4 maintenance."""

STEPS = int(os.environ.get("STEPS", "1000"))
LR = float(os.environ.get("LR", "2e-4"))
RANK = int(os.environ.get("RANK", "64"))
"""r64 = the Tier-2a capacity that passed its gate; style is a stronger
transform than weather, so start at the proven upper rank."""

SEED = int(os.environ.get("SEED", "0"))
HOLDOUT = int(os.environ.get("HOLDOUT", "2"))
"""Clips (last of the sources manifest) reserved for the eval gate."""

WARMUP = 40
GRAD_CLIP = 1.0
SAVE_EVERY = 100
LOG_EVERY = 10
EMA_DECAY = 0.98

NOOP_PROB = float(os.environ.get("NOOP_PROB", "0.1"))
MAINT_PROB = float(os.environ.get("MAINT_PROB", "0.3"))
"""Probability of a no-op window (swap to the clip's own prompt, source
targets): anchors the restyle to the text KV instead of the weights."""

V4_MAINT = os.environ.get("V4_MAINT", "1") == "1"
"""Maintenance episodes replay/target the model's OWN early-window styled
branches (self-consistency) instead of the JoyAI re-encodes; 0 = exact v3
maintenance behavior. ``MAINT_PROB`` semantics are unchanged."""

V4_EARLY = int(os.environ.get("V4_EARLY", "7"))
"""Max style-hold depth (chunks since swap, 1-based) a v4 episode may
touch: history ``[k - REPLAY_CHUNKS, k)`` and targets ``[k, k + SPAN)``
must fit inside branch depths ``1..V4_EARLY``. 7 is the tightest feasible
with ``REPLAY_CHUNKS=3`` / ``SPAN=4`` (one k per branch, ``k = s + 3``);
Issue 1 puts "excellent" at +1..+4 and structure-holds at +7..+12, so 7
keeps targets on the sharp side of the drift. Raising it trades target
sharpness for more (s, k) cells."""

PRE_CHUNKS = int(os.environ.get("PRE_CHUNKS", "1"))
"""Supervised chunks immediately before ``k`` (source targets, original
prompt) — no-op regularization at the swap boundary."""

SPAN = int(os.environ.get("SPAN", "4"))
"""Supervised chunks per window, ``[k, k + SPAN)`` (the deploy edit-window
width; Tier-2a shipped 6-chunk windows, styled pairs support any width)."""

SWAP_MIN, SWAP_MAX = (
    int(os.environ.get("SWAP_MIN", "4")),
    int(os.environ.get("SWAP_MAX", "20")),
)
"""Swap chunk ``k`` range — past the 3-chunk KV window fill; the upper end
is clamped per pair so the window fits the pair's usable range."""

EARLY_CHUNKS = int(os.environ.get("EARLY_CHUNKS", "15"))
"""Usable chunks for ``early_window_ok``-only pairs: chunks 0..14 cover
frames 0..116 (~3.9 s) — the pre-drift window the style filter certified."""

REPLAY_CHUNKS = int(os.environ.get("REPLAY_CHUNKS", "3"))
"""Source chunks replayed before the first supervised chunk. 3 chunks =
the full ``window_size_t=6`` / ``len_t=2`` KV window, so the visible state
equals a full-history replay (``train_v2.py`` mid-stream-start machinery);
raise only if the host's window grows."""

CTX_WEIGHT = 0.5
"""Weight of the finalize/context-forward (t=128) term."""

SKIP_STYLES = frozenset(
    s
    for s in os.environ.get("SKIP_STYLES", ",".join(DEFAULT_SKIP_STYLES)).split(",")
    if s
)
"""Slugs excluded from training regardless of the filter verdict."""


def load_pairs(train_uuids: set[str], n_chunks: int) -> list[tuple[str, str, int]]:
    """Collect trainable ``(uuid, slug, max_chunk)`` styled pairs.

    ``max_chunk`` is the exclusive bound on supervised chunk indices:
    the clip length for filter-``passed`` pairs, :data:`EARLY_CHUNKS` for
    ``early_window_ok``-only pairs. Pairs missing precomputed latents, on
    held-out clips, in :data:`SKIP_STYLES`, or rejected by the filter are
    dropped. Without a filter report every remaining pair trains at full
    range (with a warning) — rerun once the filter job lands.
    """
    report_path = STYLE_DIR / "filter_report.json"
    report: dict[str, dict] = {}
    if report_path.exists():
        for entry in json.loads(report_path.read_text()):
            if "output" in entry:
                report[entry["output"]] = entry
    else:
        print(
            f"WARNING: {report_path} not found — training on ALL styled "
            "pairs unfiltered.",
            flush=True,
        )

    pairs: list[tuple[str, str, int]] = []
    for path in sorted(STYLE_DIR.glob("*__*.mp4")):
        uuid, slug = path.stem.split("__", 1)
        if uuid not in train_uuids or slug in SKIP_STYLES:
            continue
        if slug not in STYLE_PROMPTS:
            print(f"skip {path.name}: slug {slug!r} not in STYLE_PROMPTS", flush=True)
            continue
        if not (LAT_DIR / f"{path.stem}_latents.pt").exists():
            print(f"skip {path.name}: no precomputed latents", flush=True)
            continue
        if report_path.exists():
            entry = report.get(path.name)
            if entry is None or "error" in entry:
                continue
            if entry.get("passed"):
                max_chunk = n_chunks
            elif entry.get("early_window_ok"):
                max_chunk = EARLY_CHUNKS
            else:
                continue
        else:
            max_chunk = n_chunks
        if min(SWAP_MAX, max_chunk - SPAN) < SWAP_MIN:
            print(f"skip {path.name}: usable range < SWAP_MIN", flush=True)
            continue
        pairs.append((uuid, slug, max_chunk))
    return pairs


def load_v4_branches(pair_keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """Load the clean-branch segments usable for v4 maintenance episodes.

    A ``gen_style_drift_pairs.py`` branch file stores per swap offset ``s``
    the styled rollout's chunks ``s..n_chunks-1``. A maintenance swap chunk
    ``k`` is usable when the whole episode — history ``[k - REPLAY_CHUNKS,
    k)`` plus targets ``[k, k + SPAN)`` — lies at hold depth <=
    :data:`V4_EARLY` inside the branch::

        s + REPLAY_CHUNKS <= k <= min(s + V4_EARLY, n_chunks) - SPAN

    Returns ``{(uuid, slug): {"cells": [(s, k), ...], "branches":
    {s: latents for chunks s..}}}`` for the trainable ``pair_keys``,
    branch latents trimmed to the deepest chunk any usable ``k`` touches.
    Files or offsets too short for any ``k`` are skipped (reported).
    """
    corpus: dict[tuple[str, str], dict] = {}
    for path in sorted(PAIRS_DIR.glob("*__*.pt")):
        uuid, slug = path.stem.split("__", 1)
        if (uuid, slug) not in pair_keys:
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        n = int(d["n_chunks"])
        cells: list[tuple[int, int]] = []
        branches: dict[int, list[Tensor]] = {}
        for s, lats in sorted((int(s), lat) for s, lat in d["branches"].items()):
            assert len(lats) == n - s, f"{path.name} s={s}: {len(lats)} chunks"
            ks = list(range(s + REPLAY_CHUNKS, min(s + V4_EARLY, n) - SPAN + 1))
            if not ks:
                continue
            cells += [(s, k) for k in ks]
            branches[s] = lats[: max(ks) + SPAN - s]
        if not cells:
            print(f"v4: skip {path.name} — no usable k at V4_EARLY={V4_EARLY}")
            continue
        corpus[(uuid, slug)] = {"cells": cells, "branches": branches}
        print(
            f"v4: {uuid[:8]}__{slug} usable k {[k for _, k in cells]}",
            flush=True,
        )
    return corpus


def main() -> None:
    """Run the teacher-forced style-SFT loop."""
    rng = np.random.default_rng(SEED)
    manifest = json.loads((SRC_DIR / "manifest.json").read_text())
    uuids: list[str] = [e["uuid"] for e in manifest]
    n_chunks = int(manifest[0]["n_chunks"])
    assert len(uuids) > HOLDOUT, f"{len(uuids)} clips can't spare {HOLDOUT} held out"
    train_uuids = uuids[: len(uuids) - HOLDOUT]
    pairs = load_pairs(set(train_uuids), n_chunks)
    assert pairs, (
        f"no trainable styled pairs under {STYLE_DIR} (run precompute_style.py "
        "first; check filter_report.json / SKIP_STYLES)."
    )
    v4_corpus = load_v4_branches({(u, s) for u, s, _ in pairs}) if V4_MAINT else {}
    v4_fallbacks = [0]  # maintenance draws without a branch -> v3 path
    if V4_MAINT:
        n_cells = sum(len(e["cells"]) for e in v4_corpus.values())
        print(
            f"v4 maintenance ON: {n_cells} self-consistency cells over "
            f"{len(v4_corpus)}/{len(pairs)} pairs (V4_EARLY={V4_EARLY}); "
            "pairs without branches fall back to v3 maintenance.",
            flush=True,
        )

    prompt_emb = torch.load(
        OUT_DIR / "style_embeddings.pt", map_location="cpu", weights_only=False
    )
    assets = torch.load(
        OUT_DIR / "style_clip_assets.pt", map_location="cpu", weights_only=False
    )

    # Latents are small (~0.9 MB per chunk) — front-load everything on CPU.
    src_lat: dict[str, list[Tensor]] = {}
    hd_lat: dict[str, list[Tensor]] = {}
    for uuid in sorted({uuid for uuid, _, _ in pairs}):
        src = torch.load(
            SRC_DIR / f"{uuid}_latents.pt", map_location="cpu", weights_only=False
        )["latents"]
        hd = torch.load(
            LAT_DIR / f"{uuid}_hdmap_latents.pt",
            map_location="cpu",
            weights_only=False,
        )["hdmaps"]
        assert len(src) == n_chunks and len(hd) == n_chunks, (
            f"clip {uuid}: {len(src)} source / {len(hd)} hdmap chunks != {n_chunks}"
        )
        src_lat[uuid], hd_lat[uuid] = src, hd
    tgt_lat: dict[tuple[str, str], list[Tensor]] = {}
    for uuid, slug, _ in pairs:
        tgt = torch.load(
            LAT_DIR / f"{uuid}__{slug}_latents.pt",
            map_location="cpu",
            weights_only=False,
        )["latents"]
        # Teacher-regenerated targets cover one chunk48 window (24 student
        # chunks) — shorter than the clip; clamp the pair's usable range.
        assert SWAP_MIN + SPAN <= len(tgt) <= n_chunks, (
            f"pair {uuid}__{slug}: {len(tgt)} styled chunks "
            f"(need {SWAP_MIN + SPAN}..{n_chunks})"
        )
        tgt_lat[(uuid, slug)] = tgt
    pairs = [
        (uuid, slug, min(max_chunk, len(tgt_lat[(uuid, slug)])))
        for uuid, slug, max_chunk in pairs
    ]

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
    ctx_t = torch.tensor(float(dm.config.context_noise), device=device, dtype=dtype)
    # Context-noise sigma from the warped training table (the add_noise
    # snap, resolved once so the v-target can be built from a known eps).
    ctx_idx = torch.argmin((scheduler._full_timesteps - ctx_t.float()).abs())
    sigma_ctx = scheduler._full_sigmas[ctx_idx]

    network = unwrap_compiled(transformer.network)
    network.requires_grad_(False)  # frozen base; only the LoRA A/B path trains
    wrapped = apply_lora(network, rank=RANK, targets=LORA_TARGETS)
    params = lora_parameters(network)
    n_pass = sum(mc == n_chunks for _, _, mc in pairs)
    print(
        f"LoRA r{RANK} on {len(wrapped)} projections | "
        f"{sum(p.numel() for p in params) / 1e6:.2f}M params | "
        f"{len(pairs)} pairs ({n_pass} full + {len(pairs) - n_pass} early-window) "
        f"over {len(src_lat)} clips ({HOLDOUT} held out) | "
        f"styles {sorted({slug for _, slug, _ in pairs})}",
        flush=True,
    )
    patch_functional_attention()
    checkpoint_blocks(network)
    opt = torch.optim.AdamW(params, lr=LR)

    def predict_v(tc, z_t: Tensor, timestep: Tensor, hd_p: Tensor) -> Tensor:
        """One functional-attention (non-writing) forward -> fp32 flow."""
        with functional_attention():
            flow = transformer.predict_flow(
                noisy_latent=z_t, timestep=timestep, cache=tc, input=hd_p
            )
        return flow.float()

    def train_chunk(
        tc,
        j: int,
        x0_tgt: Tensor,
        hd_p: Tensor,
        weight: float,
        in_window: bool,
        model_rng: torch.Generator,
    ) -> dict[str, float]:
        """Supervise chunk ``j`` toward ``x0_tgt`` and commit it to the KV.

        Mirrors ``scheduler.sample``: the student's own (detached) flow
        advances the two-step trajectory, and each state's v-target points
        at the teacher-forced ``x0_tgt`` (``(z_t - x0) / sigma``; at
        t=1000, sigma=1, that is exactly ``eps - x0``). Each term backwards
        immediately inside ``functional_attention()``. The KV commit is
        the stock finalize on ``x0_tgt`` re-noised at t=128 — the same
        state the 0.5-weighted context term supervises — run at LoRA
        scale 1 inside the edit window (deploy commits merged) and scale 0
        before it (deploy commits base pre-swap).

        Args:
            tc: Transformer cache, positioned at chunk ``j - 1``.
            j: Supervised AR chunk index.
            x0_tgt: fp32 patchified target latent (styled or source).
            hd_p: Patchified HDMap conditioning for chunk ``j``.
            weight: Per-chunk loss weight (1 / supervised chunks).
            in_window: Whether ``j`` is inside the edit window (>= k).
            model_rng: The rollout's noise generator.

        Returns:
            Unweighted per-term losses (t1000 / t450 / ctx floats).
        """
        tc.start(j)
        losses: dict[str, float] = {}
        noisy = torch.randn(
            transformer.latent_shape, device=device, dtype=dtype, generator=model_rng
        )
        clean: Tensor | None = None
        for i in range(n_steps):
            sigma = sigmas[i]
            timestep = timesteps[i].to(dtype=dtype)
            if i > 0:
                assert clean is not None
                noise = torch.empty_like(noisy).normal_(generator=model_rng)
                noisy = ((1.0 - sigma) * clean + sigma * noise).to(dtype)
            v_student = predict_v(tc, noisy, timestep, hd_p)
            v_target = (noisy.float() - x0_tgt) / sigma
            term = (v_student - v_target).square().mean()
            with functional_attention():
                (weight * term / n_steps).backward()
            losses[f"t{int(timesteps[i].item())}"] = float(term)
            clean = noisy - sigma * v_student.detach()  # fp32 via promotion

        # Context term at the state the commit below writes: for a known
        # eps, the v-target is exactly eps - x0.
        eps = torch.randn(
            transformer.latent_shape,
            device=device,
            dtype=torch.float32,
            generator=model_rng,
        )
        z_ctx = ((1.0 - sigma_ctx) * x0_tgt + sigma_ctx * eps).to(dtype)
        v_ctx = predict_v(tc, z_ctx, ctx_t, hd_p)
        term = (v_ctx - (eps - x0_tgt)).square().mean()
        with functional_attention():
            (weight * CTX_WEIGHT * term).backward()
        losses["ctx"] = float(term)

        # All backwards done -> the stock finalize (buffer write + index
        # advance) commits the teacher-forced target into the history.
        if not in_window:
            set_lora_scale(network, 0.0)
        with torch.no_grad():
            transformer.finalize_kv_cache(
                noisy_latent=z_ctx.detach(), timestep=ctx_t, cache=tc, input=hd_p
            )
        if not in_window:
            set_lora_scale(network, 1.0)
        tc.finalize(j)
        return losses

    def train_step() -> tuple[dict[str, float], str]:
        """One edit-timestamped episode -> (losses, episode).

        Backwards happen inside (per term); the caller owns ``zero_grad``
        / clip / ``opt.step``.
        """
        uuid, slug, max_chunk = pairs[int(rng.integers(len(pairs)))]
        no_op = bool(rng.random() < NOOP_PROB)
        # Steady-state ("maintenance") episode: the swap happened long ago —
        # styled history, style text, LoRA on, styled targets. Without this
        # the LoRA only ever sees unstyled history and learns a constant
        # style PUSH; deployed past the trained span the push compounds
        # through the KV commits and the world blurs out (observed at
        # +7-10 chunks on the step-1600 run).
        maint = (not no_op) and bool(rng.random() < MAINT_PROB)
        # v4: self-consistency maintenance — replay/target a clean
        # early-window branch of the model's OWN styled rollout instead of
        # the (VAE-soft) JoyAI latents. Only maintenance changes; swap and
        # no-op episodes keep JoyAI/source targets.
        v4 = v4_corpus.get((uuid, slug)) if maint else None
        if maint and V4_MAINT and v4 is None:
            v4_fallbacks[0] += 1
        if v4 is not None:
            s, k = v4["cells"][int(rng.integers(len(v4["cells"])))]
        else:
            k_hi = min(SWAP_MAX, max_chunk - SPAN)
            k = int(rng.integers(SWAP_MIN, k_hi + 1))
        n_pre = 0 if maint else min(PRE_CHUNKS, k)
        j0 = k - n_pre

        dm._rng = torch.Generator(device=device).manual_seed(int(rng.integers(2**31)))
        # Build the original prompt's text KV at base weights: at deploy the
        # LoRA window is closed until the swap, and a scale left at 1 from
        # the previous episode would bake LoRA deltas into the pre-swap
        # cache (reviewer-caught).
        set_lora_scale(network, 0.0)
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=prompt_emb[clip_key(uuid)],
            image_embeddings=assets["image_embeddings"][uuid],
        )
        tc = cache.transformer_cache
        end = k + SPAN
        hd = [x.to(device, dtype) for x in hd_lat[uuid][:end]]
        if v4 is not None:
            # Absolute-indexed view of the branch (chunks < s are never
            # touched: history starts at k - REPLAY_CHUNKS >= s and
            # maintenance has no pre-window / no-op chunks).
            blat = v4["branches"][s]
            tgt = [None] * s + [x.to(device, dtype) for x in blat[: end - s]]
            src = tgt
        else:
            src = [x.to(device, dtype) for x in src_lat[uuid][:end]]
            tgt = [x.to(device, dtype) for x in tgt_lat[(uuid, slug)][:end]]

        if maint:
            # In-window steady state: text already swapped, history styled,
            # commits at LoRA scale 1 (deploy's use_lora window semantics).
            pipe.replace_text_from_embeddings(cache, prompt_emb[slug])

        # Context build: replay the KV-window prefix (SOURCE latents at base
        # weights for swap episodes — deploy runs pre-window chunks on the
        # unmodified network; STYLED latents at scale 1 for maintenance
        # episodes). Per-chunk seeded context noise = the replay_history
        # contract; the mid-stream start is the train_v2 machinery.
        replay = tgt if maint else src
        set_lora_scale(network, 1.0 if maint else 0.0)
        start = max(0, j0 - REPLAY_CHUNKS)
        with torch.no_grad():
            for bc in tc.network_cache.block_caches:
                bc.self_attn._prev_chunk_idx = start - 1
            for j in range(start, j0):
                g = torch.Generator(device=device).manual_seed(CONTEXT_NOISE_SEED + j)
                noisy = scheduler.add_noise(replay[j], ctx_t, rng=g)
                tc.start(j)
                transformer.finalize_kv_cache(
                    noisy_latent=noisy, timestep=ctx_t, cache=tc, input=hd[j]
                )
                tc.finalize(j)
        set_lora_scale(network, 1.0)

        model_rng = dm.rng
        assert model_rng is not None
        weight = 1.0 / (n_pre + SPAN)
        sums: dict[str, float] = {}
        total = 0.0
        for j in range(j0, end):
            if j == k and not maint:
                # The deploy swap: plain cross-attn KV rebuild (no-op
                # windows swap to the clip's own prompt — same machinery,
                # identical KV — so style must be read from the text).
                emb = prompt_emb[clip_key(uuid) if no_op else slug]
                pipe.replace_text_from_embeddings(cache, emb)
            x0_tgt = (src[j] if (no_op or (not maint and j < k)) else tgt[j]).float()
            losses = train_chunk(
                tc, j, x0_tgt, hd[j], weight, maint or j >= k, model_rng
            )
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + weight * value
            total += weight * (
                sum(v for key, v in losses.items() if key != "ctx") / n_steps
                + CTX_WEIGHT * losses["ctx"]
            )

        del cache
        sums["total"] = total
        if no_op:
            tag = "no_op:"
        elif v4 is not None:
            tag = f"maintv4[s={s}]:"
        elif maint:
            tag = "maintfb:" if V4_MAINT else "maint:"
        else:
            tag = ""
        episode = (
            f"{uuid[:8]} {tag}{slug[:16]} k={k} pre={n_pre} span={SPAN} max={max_chunk}"
        )
        return sums, episode

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(True)
    ema: float | None = None
    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        losses, episode = train_step()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        torch.cuda.empty_cache()  # the episode's KV caches died with its frame
        ema = (
            losses["total"]
            if ema is None
            else EMA_DECAY * ema + (1.0 - EMA_DECAY) * losses["total"]
        )
        if step % LOG_EVERY == 0 or step == 1:
            terms = " ".join(f"{k} {v:.4f}" for k, v in losses.items() if k != "total")
            print(
                f"step {step:5d} | loss {losses['total']:.4f} (ema {ema:.4f})"
                f" | {terms} | {episode}",
                flush=True,
            )
        if step % SAVE_EVERY == 0 or step == STEPS:
            path = OUT_DIR / f"{CKPT_PREFIX}_step{step}.pt"
            save_lora(network, path)
            print(f"saved {path}", flush=True)

    fb = f" | v4 fallbacks {v4_fallbacks[0]}" if V4_MAINT else ""
    print(f"TRAIN-STYLE-SFT-DONE | final loss ema {ema:.4f}{fb}", flush=True)


if __name__ == "__main__":
    main()
