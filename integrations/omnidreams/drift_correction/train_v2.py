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

"""v2 corrector trainer: DAgger pool + drift-contraction (Omnidreams).

The paper-final recipe on this host: the v1 counterfactual loss over the
aggregated pair pool (round-0 pairs-v2 + corrected-rollout DAgger round-1),
plus a drift-contraction term — the corrected chunk-``k`` prediction is
*committed* into chunk ``k+1``'s KV history WITH gradient and chunk
``k+1``'s gap to its clean teacher is penalized (weight ``CW_LOSS``)::

    x0_corr(k)   = z_t - sigma * v_corr(k)                     (grad)
    commit       : chunk-k context forward on noisy(x0_corr)   (grad KV,
                   recorded functionally; buffer gets a no-grad twin)
    L_con        = ||v_inject(k+1) - v_clean(k+1)||^2
                   / ||v_clean(k+1) - v_base(k+1)||^2
    L            = L_dag + CW_LOSS * L_con

Host mechanics: the commit's grad-carrying per-block K/V are captured by
``record_kv`` and swapped over their numerically identical buffered twins
by ``inject_kv`` during the chunk-``k+1`` probe (``_train_attn``).

Run from the repo root (resumable: re-run the same command)::

    STEPS=1000 .venv/bin/python integrations/omnidreams/drift_correction/train_v2.py
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
from _train_attn import (
    functional_attention,
    inject_kv,
    patch_functional_attention,
    record_kv,
)
from torch import Tensor

## Training configuration

BASE = Path("integrations/omnidreams/drift_correction")

POOLS = [
    BASE / "outputs" / p
    for p in os.environ.get("POOLS", "pairs_v2,pairs_dagger1").split(",")
]
"""Pair pools, aggregated (round-0 + corrected-rollout DAgger rounds)."""

INIT = os.environ.get("INIT", str(BASE / "outputs/lora_v1.pt"))
"""Warm-start checkpoint (v1)."""

CKPT = Path(os.environ.get("CKPT", str(BASE / "outputs/lora_v2.pt")))
"""Output checkpoint (LoRA + optimizer + step); saved every ``SAVE_EVERY``."""

STEPS = int(os.environ.get("STEPS", "1000"))
LR = float(os.environ.get("LR", "2e-4"))
WARMUP = 40
GRAD_CLIP = 1.0
RANK = 16
EVAL_EVERY = 100
SAVE_EVERY = 200
CW_LOSS = float(os.environ.get("CW_LOSS", "0.5"))
"""Drift-contraction weight (paper-final value)."""

SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "0"))
"""When > 0: also keep step-tagged snapshot copies (``<ckpt>_stepN.pt``)
every SNAP_EVERY steps plus a running val-peak snapshot
(``<ckpt>_valpeak.pt``, refreshed whenever val dag-R^2 improves at an eval
point), so the sweep can compare checkpoints without retraining (the
checkpoint-selection comparison)."""

CLEAN_LAP = 2
MIN_LAP = 4
REPLAY_CHUNKS = 15
ALPHA_STAR = tuple(
    float(x) for x in os.environ.get("ALPHA_STAR", "0.96,0.667").split(",")
)
REL_V_EXCLUDE = 0.8
N_VAL_CLIPS = 1
"""Same data conventions as ``train_v1`` (same conventions as train_v1); the
val clip is the last clip of the FIRST pool (round-0 held-out scene)."""

SEED = int(os.environ.get("SEED", "0"))

PAIR_SCHEME = os.environ.get("PAIR_SCHEME", "lap")
"""``lap`` (v1-v3 loop pairs) or ``fork`` (pairs-v4 re-anchored forks);
see ``train_v1.py`` / ``build_pairs_v4.py``."""

UW = os.environ.get("UW", "0") == "1"
"""Uncertainty-weighted DAgger term (per-token alpha* weights from
``UW_DRAWS`` noise draws; see ``train_v1.py``). The contraction term stays
unweighted: it penalizes accumulation of whatever residual remains, while
the weights' job is to keep unpredictable content out of the *target* —
recorded design choice 2026-07-24. ``UW=0`` = ablation arm."""

UW_DRAWS = 2


def lap_aligned(c: int, lap_chunks: int) -> int:
    """Map chunk ``c`` (>= lap 1) to its lap-:data:`CLEAN_LAP` counterpart."""
    assert c >= 1, "chunk 0 is image-anchored and never remapped"
    return 1 + CLEAN_LAP * lap_chunks + (c - 1) % lap_chunks


def training_ks(num_chunk: int, lap_chunks: int, laps: int) -> list[int]:
    """Probe chunks in laps >= MIN_LAP with a successor chunk available."""
    return [k for k in range(1 + MIN_LAP * lap_chunks, num_chunk - 1)]


def training_ks_fork(num_chunk: int, fork_starts: list[int]) -> list[int]:
    """Fork-scheme cells past the first segment, successor available."""
    return [k for k in range(int(fork_starts[1]), num_chunk - 1)]


def clean_chunk(d: dict, gen: list[Tensor], j: int, lap_chunks: int) -> Tensor:
    """Clean counterpart content for replayed chunk ``j`` (>= 1)."""
    if PAIR_SCHEME == "fork":
        starts = d["fork_starts"]
        s = max(i for i, cs in enumerate(starts) if cs <= j)
        return d["fork_latents"][s][j - int(starts[s])]
    return gen[lap_aligned(j, lap_chunks)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    clip_paths: list[Path] = []
    val_paths: list[Path] = []
    for i, pool in enumerate(POOLS):
        clips = sorted(pool.glob("clip_*.pt"))
        assert clips, f"no clips under {pool}"
        if i == 0:
            val_paths = clips[-N_VAL_CLIPS:]
            clips = clips[:-N_VAL_CLIPS]
        clip_paths += clips
    datas = [load_clip(p, "cpu", torch.bfloat16) for p in clip_paths + val_paths]
    train_ids = list(range(len(clip_paths)))
    val_ids = list(range(len(clip_paths), len(datas)))

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
    print(
        f"{len(datas)} clips ({len(val_ids)} val) from {len(POOLS)} pools | "
        f"lap_chunks {[int(d['lap_chunks']) for d in datas]} | cells/clip "
        f"{[len(ks) for ks in ks_by_clip]} | contraction w={CW_LOSS}",
        flush=True,
    )

    emb = datas[0]["embeddings"]
    cache = pipe.initialize_cache_from_embeddings(
        text_embeddings=emb["text_embeddings"],
        image_embeddings=emb["image_embeddings"],
    )
    tc = cache.transformer_cache
    _text_clip = [0]

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

    def use_clip_text(c: int) -> None:
        """Swap the cross-attn text KV when the sampled clip changes."""
        if _text_clip[0] != c:
            swap_text_kv(network, tc, datas[c]["embeddings"]["text_embeddings"])
            _text_clip[0] = c

    start_step = 0
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        for i, p in enumerate(params):
            p.data.copy_(state["lora"][i].to(p.device, p.dtype))
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        rng = np.random.default_rng(state["rng_seed"])
        print(f"RESUMED from {CKPT} at step {start_step}", flush=True)
    elif INIT and INIT != "scratch":
        state = torch.load(INIT, map_location="cpu", weights_only=False)
        for i, p in enumerate(params):
            p.data.copy_(state["lora"][i].to(p.device, p.dtype))
        print(f"warm start from {INIT}", flush=True)

    def save(step: int, path: Path = CKPT) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "lora": {i: p.detach().cpu() for i, p in enumerate(params)},
                "opt": opt.state_dict(),
                "step": step,
                "rng_seed": SEED + step,
            },
            tmp,
        )
        tmp.replace(path)

    _device_clips: dict[int, dict] = {}

    def to_device(c: int) -> dict:
        """Device-resident view of clip ``c`` (memoized)."""
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
        """Rebuild the KV state from chunks ``k - REPLAY_CHUNKS .. k - 1``."""
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

    def make_zt(x0: Tensor, t_idx: int, rng_: np.random.Generator) -> Tensor:
        sig = sigmas[t_idx].to(dtype)  # ty: ignore[not-subscriptable]
        g = torch.Generator(device=device).manual_seed(int(rng_.integers(2**31)))
        return (1 - sig) * x0 + sig * torch.randn(
            x0.shape, device=device, dtype=dtype, generator=g
        )

    excl = {c: {"tested": 0, "excluded": 0} for c in range(len(datas))}

    def sample_losses(
        c: int, grad: bool, rng_: np.random.Generator
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """One pooled sample -> (dag loss, contraction loss, r_target sq).

        Returns ``None`` for degenerate (collapsed) cells; callers redraw.
        """
        use_clip_text(c)
        d = to_device(c)
        lap_chunks = int(d["lap_chunks"])
        k = int(rng_.choice(ks_by_clip[c]))
        t_idx = int(rng_.choice(n_steps, p=t_probs))
        t2_idx = int(rng_.choice(n_steps, p=t_probs))
        gen = d["latents"]
        clean = list(gen)
        for j in range(max(1, k - REPLAY_CHUNKS), k + 1):
            clean[j] = clean_chunk(d, gen, j, lap_chunks)
        n_draws = UW_DRAWS if UW else 1
        z_ts = [make_zt(gen[k], t_idx, rng_) for _ in range(n_draws)]
        z_t = z_ts[0]
        z_t2 = make_zt(gen[k + 1], t2_idx, rng_)

        with torch.no_grad():
            set_lora_scale(network, 0.0)  # ty: ignore[invalid-argument-type]
            # Teacher/base at k (clean swap over the replay span).
            replay_window(clean, d["hdmaps"], k)
            tc.start(k)
            v_cleans = [predict_v(z, t_idx, d["hdmaps"][k]) for z in z_ts]
            tc.finalize(k)
            # Teacher/base at k+1 for the contraction target (the clean
            # history now includes the counterpart chunk k).
            replay_window(clean, d["hdmaps"], k + 1)
            tc.start(k + 1)
            v_clean2 = predict_v(z_t2, t2_idx, d["hdmaps"][k + 1])
            tc.finalize(k + 1)
            replay_window(gen, d["hdmaps"], k + 1)
            tc.start(k + 1)
            v_base2 = predict_v(z_t2, t2_idx, d["hdmaps"][k + 1])
            tc.finalize(k + 1)
            replay_window(gen, d["hdmaps"], k)
            tc.start(k)
            v_bases = [predict_v(z, t_idx, d["hdmaps"][k]) for z in z_ts]
            tc.finalize(k)
        v_clean, v_base = v_cleans[0], v_bases[0]
        r_sq = (v_clean - v_base).square().sum()
        r2_sq = (v_clean2 - v_base2).square().sum()

        w = None
        if UW:
            rs = torch.stack([vc - vb for vc, vb in zip(v_cleans, v_bases)])
            mean_r = rs.mean(0)
            var = rs.var(0, unbiased=True)
            bias2 = (mean_r.square() - var / n_draws).clamp_min(0.0)
            w = (bias2 / (bias2 + var + 1e-12)).detach()

        excl[c]["tested"] += 1
        rel_v = (r_sq.sqrt() / (v_base.norm() + 1e-9)).item()
        if rel_v > REL_V_EXCLUDE:
            excl[c]["excluded"] += 1
            return None

        set_lora_scale(network, 1.0)  # ty: ignore[invalid-argument-type]
        with torch.no_grad():
            replay_window(gen, d["hdmaps"], k)  # LoRA-scaled replay, no grad
        ctx = torch.enable_grad() if grad else torch.no_grad()
        tc.start(k)
        with ctx:
            v_corr = predict_v(z_t, t_idx, d["hdmaps"][k])
            err = (v_corr - v_clean).square()
            gap = (v_clean - v_base).square()
            if w is not None:
                err, gap = err * w, gap * w
            dag = err.sum() / (gap.sum() + 1e-8)
            # Commit chunk k's corrected prediction with grad: recorded
            # functional KV + a numerically identical no-grad buffer twin.
            sig = sigmas[t_idx].to(dtype)  # ty: ignore[not-subscriptable]
            x0_corr = (z_t.float() - float(sig) * v_corr).to(dtype)
            g = torch.Generator(device=device).manual_seed(77_000 + k)
            eps = torch.randn(x0_corr.shape, device=device, dtype=dtype, generator=g)
            idx = torch.argmin(
                (scheduler._full_timesteps - ctx_t.float()).abs()  # ty: ignore[unsupported-operator]
            ).reshape(1)
            sig_ctx = scheduler._full_sigmas.index_select(0, idx).reshape(()).to(dtype)  # ty: ignore[call-non-callable]
            noisy_corr = (1 - sig_ctx) * x0_corr + sig_ctx * eps
            recorded: list = []
            with record_kv(recorded), functional_attention():
                transformer.predict_flow(
                    noisy_latent=noisy_corr,
                    timestep=ctx_t,
                    cache=tc,
                    input=d["hdmaps"][k],
                )
        with torch.no_grad():  # buffer twin write + index advance
            transformer.finalize_kv_cache(
                noisy_latent=noisy_corr.detach(),
                timestep=ctx_t,
                cache=tc,
                input=d["hdmaps"][k],
            )
        tc.finalize(k)
        tc.start(k + 1)
        with ctx:
            with inject_kv(recorded):
                v_con = predict_v(z_t2, t2_idx, d["hdmaps"][k + 1])
            con = (v_con - v_clean2).square().sum() / (r2_sq + 1e-8)
        tc.finalize(k + 1)
        return dag, con, r_sq

    def draw_losses(
        ids: list[int], grad: bool, rng_: np.random.Generator
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Redraw until a non-degenerate cell is sampled."""
        while True:
            out = sample_losses(int(rng_.choice(ids)), grad, rng_)
            if out is not None:
                return out

    @torch.no_grad()
    def val_r2(n: int = 12) -> tuple[float, float]:
        vrng = np.random.default_rng(1234)
        s_dag = s_con = 0.0
        for _ in range(n):
            dag, con, _ = draw_losses(val_ids, False, vrng)
            s_dag += dag.item()
            s_con += con.item()
        return 1 - s_dag / n, 1 - s_con / n

    torch.set_grad_enabled(True)
    best_vd = float("-inf")
    for step in range(start_step + 1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        dag, con, r_sq = draw_losses(train_ids, True, rng)
        loss = dag + CW_LOSS * con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            vd, vc = val_r2()
            excl_str = " ".join(
                f"c{c}:{v['excluded']}/{v['tested']}" for c, v in excl.items()
            )
            print(
                f"step {step:5d} | dag {dag.item():.4f} con {con.item():.4f}"
                f" | val dag-R^2 {vd:+.3f} con-R^2 {vc:+.3f}"
                f" | |r|^2 {r_sq.item():.1f} | excluded {excl_str}",
                flush=True,
            )
            if SNAP_EVERY and vd > best_vd:
                best_vd = vd
                save(step, CKPT.with_name(f"{CKPT.stem}_valpeak.pt"))
                print(
                    f"val-peak snapshot at step {step} (dag-R^2 {vd:+.3f})", flush=True
                )
        if step % SAVE_EVERY == 0 or step == STEPS:
            save(step)
        if SNAP_EVERY and step % SNAP_EVERY == 0:
            save(step, CKPT.with_name(f"{CKPT.stem}_step{step}.pt"))

    vd, vc = val_r2(24)
    print(
        f"TRAIN-V2-DONE | final val dag-R^2 {vd:+.3f} con-R^2 {vc:+.3f} | saved {CKPT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
