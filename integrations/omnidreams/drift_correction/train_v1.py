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

"""v1 corrector trainer: counterfactual clean-history teacher in velocity space.

Trains the LoRA corrector r_phi on the frozen distilled Omnidreams model so
its velocity prediction under drifted history matches the frozen model's
prediction under the lap-aligned clean history at the same ``z_t``::

    L = ||v_{theta+LoRA}(z_t, h_gen, t) - v_theta(z_t, h_clean, t)||^2
        / ||v_theta(z_t, h_clean, t) - v_theta(z_t, h_gen, t)||^2

The drift-gap denominator is required (raw MSE diverges; reference
finding); ``R^2 = 1 - L``. Both passes share the exact ``z_t`` re-noised
from the rollout's own x0 (native anchoring). Timesteps are sampled from
the 2 distillation steps with probability proportional to the measured
``alpha*(t)`` (pairs-v2 gate: 0.96 @ t=1000, 0.667 @ t=803); fully-collapsed
cells (``rel_v > 0.8``) are excluded at draw time per convention.

Host adaptations vs the HY reference (``hy_worldplay`` ``train_v1.py``):
velocity-space targets; history rebuilt per sample by a truncated
forged-index replay (:data:`REPLAY_CHUNKS` warmup chunks -- deep-layer KV
entangles history, so a bare window replay is NOT equivalent; checked
numerically at startup); functional self-attention toggle
(:mod:`_train_attn`) for the grad-carrying probe. Conventions (2026-07-22): clean reference = lap 2, training cells in laps >= 4,
checkpoints every <= 200 steps with RESUME.

Run from the repo root (resumable: re-run the same command)::

    STEPS=1500 .venv/bin/python integrations/omnidreams/drift_correction/train_v1.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from _host import build_pipeline, load_clip, reset_history, swap_text_kv
from _lora import apply_lora, lora_parameters, set_lora_scale
from _train_attn import functional_attention, patch_functional_attention
from torch import Tensor

## Training configuration

BASE = Path("integrations/omnidreams/drift_correction")

PAIRS_DIR = Path(os.environ.get("PAIRS_DIR", str(BASE / "outputs/pairs_v2")))
"""Clip files from ``build_pairs.py`` (v2: real-photo-seeded rollouts)."""

CKPT = Path(os.environ.get("CKPT", str(BASE / "outputs/lora_v1.pt")))
"""Output checkpoint (LoRA + optimizer + step); saved every ``SAVE_EVERY``."""

STEPS = int(os.environ.get("STEPS", "1500"))
LR = float(os.environ.get("LR", "5e-4"))
WARMUP = 60
"""Linear LR warmup steps; required for stability (reference finding)."""

GRAD_CLIP = 1.0
RANK = 16
EVAL_EVERY = 100
SAVE_EVERY = 200
"""Partial checkpoints every <= 200 steps (box wedges)."""

CLEAN_LAP = 2
"""Lap 2 supplies the clean reference (keeps the
synthetic-seed transition tail of laps 0-1 out of it)."""

MIN_LAP = 4
"""Training cells live in laps >= 4 (drifted side well past the clean lap)."""

REPLAY_CHUNKS = 15
"""History chunks replayed before a probe. The KV cache holds only 3
chunks, but deep-layer K/V entangle earlier history (each replayed chunk's
projections depend on what that forward attended to), so a bare window
replay diverges from the rollout state (rel 0.034 measured). The error
decays with warmup depth; on the photoreal pairs-v2 rollouts the floor is
higher than on the render-regime pairs (drift couples longer-range), and
15 chunks reaches rel ~0.009-0.014 = 3-5% of the drift-gap signal
(measured on this host 2026-07-22)."""

ALPHA_STAR = tuple(
    float(x) for x in os.environ.get("ALPHA_STAR", "0.96,0.667").split(",")
)
"""Measured unbiased alpha* per solver timestep (t=1000, t=803); used as
the timestep sampling weights. Default = the pairs-v2 (photoreal) gate;
override via env when a pair set gets its own gate run."""

REL_V_EXCLUDE = 0.8
"""Drop fully-collapsed cells. A sample whose
velocity-space drift gap ``||v_clean - v_base|| / ||v_base||`` exceeds this
is a degenerate state (e.g. the clip-0 late-horizon collapse) and is
rejected at draw time (the probes needed for the check are computed anyway;
exclusions are logged per clip)."""

PAIR_SCHEME = os.environ.get("PAIR_SCHEME", "lap")
"""``lap`` (v1-v3 loop pairs) or ``fork`` (pairs-v4, ``build_pairs_v4.py``):
the clean counterpart of chunk ``j`` comes from the re-anchored fork
covering ``j`` instead of a lap-aligned revisit. Training cells then live
past the first fork segment (drifted side deep, counterpart anchored)."""

UW = os.environ.get("UW", "0") == "1"
"""Uncertainty-weighted loss (analysis arm 2026-07-24): estimate the
per-token systematic share of the drift residual from ``UW_DRAWS`` noise
draws at the same cell — ``w = bias^2 / (bias^2 + var)``, the per-token
alpha* — and weight both loss numerator and denominator by ``w`` so the
LoRA never learns to chase unpredictable content (foliage). Config flag:
``UW=0`` keeps the unweighted objective as the ablation arm."""

UW_DRAWS = 2
"""Noise draws for the weight estimate (adds 2 no-grad teacher forwards)."""

N_VAL_CLIPS = 1
SEED = int(os.environ.get("SEED", "0"))

TBIN_EVAL = int(os.environ.get("TBIN_EVAL", "0"))
"""When > 0: no training — load CKPT, evaluate val R^2 binned per solver
timestep (TBIN_EVAL cells per bin) and print ``TBIN t=<t> R2=<r>`` lines
(the reliability factor rho(t) for the gain-prediction analysis, analysis arm 2026-07-24), then exit."""


def lap_aligned(c: int, lap_chunks: int) -> int:
    """Map chunk ``c`` (>= lap 1) to its lap-:data:`CLEAN_LAP` counterpart."""
    assert c >= 1, "chunk 0 is image-anchored and never remapped"
    return 1 + CLEAN_LAP * lap_chunks + (c - 1) % lap_chunks


def training_ks(num_chunk: int, lap_chunks: int, laps: int) -> list[int]:
    """All probe chunks in laps >= MIN_LAP.

    Every lap position is usable: the clean swap maps each replayed chunk
    to its lap-2 counterpart *by position*, so both branches share the lap
    cycle's boundary structure (including the conditioning teleport) and
    differ only in content cleanliness.
    """
    return [k for k in range(1 + MIN_LAP * lap_chunks, num_chunk)]


def training_ks_fork(num_chunk: int, fork_starts: list[int]) -> list[int]:
    """Fork-scheme cells: past the first segment (drifted side is deep
    while the counterpart is at most SEG_CHUNKS from its anchor)."""
    return [k for k in range(int(fork_starts[1]), num_chunk)]


def clean_chunk(d: dict, gen: list[Tensor], j: int, lap_chunks: int) -> Tensor:
    """Clean counterpart content for replayed chunk ``j`` (>= 1)."""
    if PAIR_SCHEME == "fork":
        starts = d["fork_starts"]
        s = max(i for i, cs in enumerate(starts) if cs <= j)
        return d["fork_latents"][s][j - int(starts[s])]
    return gen[lap_aligned(j, lap_chunks)]


def main() -> None:
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    rng = np.random.default_rng(SEED)

    pipe = build_pipeline(with_oneshot_encoders=False)
    device = pipe.device
    dtype = torch.bfloat16
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps = scheduler.denoising_step_list
    sigmas = scheduler.denoising_sigmas
    n_steps = timesteps.shape[0]  # ty: ignore[not-subscriptable]
    t_probs = np.array(ALPHA_STAR[:n_steps]) / sum(ALPHA_STAR[:n_steps])
    ctx_t = torch.tensor(
        float(pipe.diffusion_model.config.context_noise), device=device, dtype=dtype
    )

    datas = [load_clip(p, "cpu", dtype) for p in clips]
    # Per-clip lap geometry: pairs v3 mixes lap lengths across clips (the
    # repeat-prior fix), so nothing may assume a shared lap_chunks.
    if PAIR_SCHEME == "fork":
        ks_by_clip = [
            training_ks_fork(int(d["num_chunk"]), d["fork_starts"]) for d in datas
        ]
    else:
        ks_by_clip = [
            training_ks(int(d["num_chunk"]), int(d["lap_chunks"]), int(d["laps"]))
            for d in datas
        ]
    train_ids = list(range(len(datas) - N_VAL_CLIPS))
    val_ids = list(range(len(datas) - N_VAL_CLIPS, len(datas)))
    print(
        f"{len(datas)} clips ({len(val_ids)} val) | lap_chunks "
        f"{[int(d['lap_chunks']) for d in datas]} | cells/clip "
        f"{[len(ks) for ks in ks_by_clip]} (laps >= {MIN_LAP}, "
        f"clean lap {CLEAN_LAP})",
        flush=True,
    )

    # One long-lived cache (rope, masks, rolling self-attn buffers) serves
    # every clip; probes never touch chunk 0 (image), and the per-clip
    # prompt is handled by swapping the cross-attn text KV on clip switch.
    emb = datas[0]["embeddings"]
    cache = pipe.initialize_cache_from_embeddings(
        text_embeddings=emb["text_embeddings"],
        image_embeddings=emb["image_embeddings"],
    )
    tc = cache.transformer_cache
    _text_clip = [0]

    def use_clip_text(c: int) -> None:
        """Swap the cross-attn text KV when the sampled clip changes."""
        if _text_clip[0] != c:
            swap_text_kv(network, tc, datas[c]["embeddings"]["text_embeddings"])
            _text_clip[0] = c

    network = transformer.network
    wrapped = apply_lora(network, rank=RANK)  # ty: ignore[invalid-argument-type]
    params = lora_parameters(network)  # ty: ignore[invalid-argument-type]
    print(
        f"LoRA on {len(wrapped)} projections | "
        f"{sum(p.numel() for p in params) / 1e6:.2f}M params",
        flush=True,
    )
    patch_functional_attention()
    opt = torch.optim.AdamW(params, lr=LR)

    start_step = 0
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        for i, p in enumerate(params):
            p.data.copy_(state["lora"][i].to(p.device, p.dtype))
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        rng = np.random.default_rng(state["rng_seed"])
        print(f"RESUMED from {CKPT} at step {start_step}", flush=True)

    def save(step: int) -> None:
        tmp = CKPT.with_suffix(".tmp")
        torch.save(
            {
                "lora": {i: p.detach().cpu() for i, p in enumerate(params)},
                "opt": opt.state_dict(),
                "step": step,
                "rng_seed": SEED + step,  # fresh stream on resume
            },
            tmp,
        )
        tmp.replace(CKPT)

    _device_clips: dict[int, dict] = {}

    def to_device(c: int) -> dict:
        """Device-resident view of clip ``c`` (memoized; ~0.3 GB per clip)."""
        if c not in _device_clips:

            def move(v):
                if isinstance(v, list):
                    return [move(x) for x in v]
                return v.to(device, dtype) if isinstance(v, Tensor) else v

            _device_clips[c] = {
                key: move(v)
                for key, v in datas[c].items()
                if key
                in (
                    "latents",
                    "hdmaps",
                    "lap_chunks",
                    "num_chunk",
                    "fork_starts",
                    "fork_latents",
                )
            }
        return _device_clips[c]

    def replay_window(latents: list[Tensor], hdmaps: list[Tensor], k: int) -> None:
        """Rebuild the KV state from chunks ``k - REPLAY_CHUNKS .. k - 1``.

        Forged-index truncated replay at original absolute indices (RoPE
        preserved) on a reset cache whose ``_prev_chunk_idx`` is set so the
        ``BlockKVCache`` contiguity assert passes. Context-noise eps is
        seeded per absolute index, shared across branches.
        """
        start = max(0, k - REPLAY_CHUNKS)
        reset_history(tc)
        for bc in tc.network_cache.block_caches:
            bc.self_attn._prev_chunk_idx = start - 1
        for j in range(start, k):
            g = torch.Generator(device=device).manual_seed(77_000 + j)
            noisy = scheduler.add_noise(latents[j], ctx_t, rng=g)
            tc.start(j)
            transformer.finalize_kv_cache(
                noisy_latent=noisy, timestep=ctx_t, cache=tc, input=hdmaps[j]
            )
            tc.finalize(j)

    def predict_v(z_t: Tensor, t_idx: int, hdmap: Tensor) -> Tensor:
        t = timesteps[t_idx].to(device=device, dtype=dtype)  # ty: ignore[not-subscriptable]
        with functional_attention():
            flow = transformer.predict_flow(
                noisy_latent=z_t, timestep=t, cache=tc, input=hdmap
            )
        return flow.float()

    excl = {c: {"tested": 0, "excluded": 0} for c in range(len(datas))}

    def sample_losses(
        c: int, grad: bool, rng_: np.random.Generator, t_forced: int | None = None
    ) -> tuple[Tensor, Tensor] | None:
        """One drift-pair sample -> (normalized v-space loss, r_target sq-norm).

        Returns ``None`` when the cell is degenerate (fully-collapsed state,
        ``rel_v > REL_V_EXCLUDE``); callers redraw.
        """
        use_clip_text(c)
        d = to_device(c)
        lap_chunks = int(d["lap_chunks"])
        k = int(rng_.choice(ks_by_clip[c]))
        t_idx = int(rng_.choice(n_steps, p=t_probs)) if t_forced is None else t_forced
        gen = d["latents"]
        clean = list(gen)
        for j in range(max(1, k - REPLAY_CHUNKS), k):
            clean[j] = clean_chunk(d, gen, j, lap_chunks)
        x0 = gen[k]
        sig = sigmas[t_idx].to(dtype)  # ty: ignore[not-subscriptable]
        n_draws = UW_DRAWS if UW else 1
        z_ts = []
        for _ in range(n_draws):
            g = torch.Generator(device=device).manual_seed(int(rng_.integers(2**31)))
            z_ts.append(
                (1 - sig) * x0
                + sig * torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
            )

        with torch.no_grad():
            set_lora_scale(network, 0.0)  # ty: ignore[invalid-argument-type]
            replay_window(clean, d["hdmaps"], k)
            tc.start(k)
            v_cleans = [predict_v(z, t_idx, d["hdmaps"][k]) for z in z_ts]
            tc.finalize(k)
            replay_window(gen, d["hdmaps"], k)
            tc.start(k)
            v_bases = [predict_v(z, t_idx, d["hdmaps"][k]) for z in z_ts]
            tc.finalize(k)
        v_clean, v_base = v_cleans[0], v_bases[0]
        r_target_sq = (v_clean - v_base).square().sum()

        w = None
        if UW:
            rs = torch.stack([vc - vb for vc, vb in zip(v_cleans, v_bases)])
            mean_r = rs.mean(0)
            var = rs.var(0, unbiased=True)
            bias2 = (mean_r.square() - var / n_draws).clamp_min(0.0)
            w = (bias2 / (bias2 + var + 1e-12)).detach()

        excl[c]["tested"] += 1
        rel_v = (r_target_sq.sqrt() / (v_base.norm() + 1e-9)).item()
        if rel_v > REL_V_EXCLUDE:
            excl[c]["excluded"] += 1
            return None

        set_lora_scale(network, 1.0)  # ty: ignore[invalid-argument-type]
        with torch.no_grad():
            replay_window(gen, d["hdmaps"], k)  # LoRA-scaled replay, no grad
        tc.start(k)
        with torch.enable_grad() if grad else torch.no_grad():
            v_corr = predict_v(z_ts[0], t_idx, d["hdmaps"][k])
            err = (v_corr - v_clean).square()
            gap = (v_clean - v_base).square()
            if w is not None:
                err, gap = err * w, gap * w
            loss = err.sum() / (gap.sum() + 1e-8)
        tc.finalize(k)
        return loss, r_target_sq

    def draw_losses(
        ids: list[int],
        grad: bool,
        rng_: np.random.Generator,
        t_forced: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Redraw until a non-degenerate cell is sampled."""
        while True:
            out = sample_losses(int(rng_.choice(ids)), grad, rng_, t_forced)
            if out is not None:
                return out

    @torch.no_grad()
    def val_r2(n: int = 12) -> float:
        vrng = np.random.default_rng(1234)  # fixed cells/noise across evals
        s = 0.0
        for _ in range(n):
            loss, _ = draw_losses(val_ids, False, vrng)
            s += loss.item()
        return 1 - s / n

    def replay_equivalence_check() -> None:
        """Assert the forged-index window replay matches a full-prefix replay."""
        d = to_device(0)
        k = ks_by_clip[0][0]
        g = torch.Generator(device=device).manual_seed(1)
        z_t = (1 - sigmas[0].to(dtype)) * d["latents"][k] + sigmas[0].to(  # ty: ignore[not-subscriptable]
            dtype
        ) * torch.randn(d["latents"][k].shape, device=device, dtype=dtype, generator=g)
        set_lora_scale(network, 0.0)  # ty: ignore[invalid-argument-type]
        with torch.no_grad():
            replay_window(d["latents"], d["hdmaps"], k)
            tc.start(k)
            v_win = predict_v(z_t, 0, d["hdmaps"][k])
            tc.finalize(k)
            reset_history(tc)
            for j in range(k):
                gg = torch.Generator(device=device).manual_seed(77_000 + j)
                noisy = scheduler.add_noise(d["latents"][j], ctx_t, rng=gg)
                tc.start(j)
                transformer.finalize_kv_cache(
                    noisy_latent=noisy, timestep=ctx_t, cache=tc, input=d["hdmaps"][j]
                )
                tc.finalize(j)
            tc.start(k)
            v_full = predict_v(z_t, 0, d["hdmaps"][k])
            tc.finalize(k)
        rel = ((v_win - v_full).norm() / (v_full.norm() + 1e-9)).item()
        print(f"replay equivalence: rel diff {rel:.2e}", flush=True)
        assert rel < 2e-2, "truncated replay too far from full-prefix replay"

    replay_equivalence_check()

    if TBIN_EVAL:
        # rho(t) for the gain-prediction analysis: val R^2 per solver step.
        for t_idx in range(n_steps):
            vrng = np.random.default_rng(1234)
            s = 0.0
            for _ in range(TBIN_EVAL):
                loss, _ = draw_losses(val_ids, False, vrng, t_idx)
                s += loss.item()
            r2 = 1 - s / TBIN_EVAL
            print(f"TBIN t={int(timesteps[t_idx])} R2={r2:+.4f}", flush=True)  # ty: ignore[not-subscriptable]
        print("TBIN-DONE", flush=True)
        return

    torch.set_grad_enabled(True)
    for step in range(start_step + 1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss, rt_sq = draw_losses(train_ids, True, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            excl_str = " ".join(
                f"c{c}:{v['excluded']}/{v['tested']}" for c, v in excl.items()
            )
            print(
                f"step {step:5d} | loss {loss.item():.4f}"
                f" (train R^2 {1 - loss.item():+.3f})"
                f" | val R^2 {val_r2():+.3f} | |r_t|^2 {rt_sq.item():.1f}"
                f" | excluded {excl_str}",
                flush=True,
            )
        if step % SAVE_EVERY == 0 or step == STEPS:
            save(step)

    print(f"TRAIN-V1-DONE | final val R^2 {val_r2(24):+.3f} | saved {CKPT}", flush=True)


if __name__ == "__main__":
    main()
