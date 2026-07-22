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

"""v2 corrector trainer: DAgger pool + drift-contraction (the av2s recipe).

Initialized from the v1 LoRA, with the reference's two closed-loop upgrades
(``wan_train_v2.py``):

- **DAgger** -- train on the aggregated pool of round-0 pairs (base-rollout
  histories) and round-1 pairs regenerated with the v1 corrector active
  (``CORRECTOR_LORA`` in ``build_pairs.py``), fixing the train/deploy
  covariate shift.
- **Drift contraction** -- commit the corrected one-step
  ``x0_hat = z_t - sigma * flow_corr`` *with gradient* as the newest history
  frames for chunk ``k + 1`` and penalize that chunk's x0 gap against its
  clean teacher (weight ``CW_LOSS``), optimizing the accumulation mechanism
  directly.

Run from the repo root::

    POOLS=outputs/pairs,outputs/pairs_dagger1 INIT=outputs/lora_v1.pt \
        .venv/bin/python integrations/hy_worldplay/drift_correction/train_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from _lora import apply_lora, load_lora, lora_parameters, save_lora, set_lora_scale
from _pairs import (
    TOKENS_PER_FRAME,
    chunk_x0,
    clean_counterfactual,
    history_of,
    load_clip,
    make_ctrl,
)
from _rollout import build_runner, finish_probe_chunk, start_probe_chunk
from _train_attn import patch_functional_attention
from hy_worldplay._action import HyWorldPlayWan21TransformerCache
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame
from train_v1 import checkpoint_blocks

## Training configuration

_BASE = Path("integrations/hy_worldplay/drift_correction")

POOLS = [
    _BASE / p
    for p in os.environ.get("POOLS", "outputs/pairs,outputs/pairs_dagger1").split(",")
]
"""DAgger aggregation: round-0 (base rollouts) + round-1 (corrector-active
rollouts). Samples draw uniformly over pools."""

INIT = os.environ.get("INIT", str(_BASE / "outputs/lora_v1.pt"))
"""v1 LoRA checkpoint to initialize from (required for v2)."""

CKPT = Path(os.environ.get("CKPT", str(_BASE / "outputs/lora_v2.pt")))

STEPS = int(os.environ.get("STEPS", "600"))
LR = float(os.environ.get("LR", "2e-4"))
"""Continue-training LR (below v1's 5e-4, per the reference)."""

WARMUP = 40
CW_LOSS = float(os.environ.get("CW_LOSS", "0.5"))
"""Drift-contraction term weight."""

T_WEIGHTS = [float(w) for w in os.environ.get("T_WEIGHTS", "1,1,1,1").split(",")]
"""Sampling weights over the 4 distillation timesteps (t=1000 first).
Oversampling t=1000 concentrates training where the faithful gate measured
the drift gap as most systematic (alpha* 0.81 vs ~0.55 elsewhere)."""

FID_W = float(os.environ.get("FID_W", "0"))
"""Content-fidelity (trust-region) weight: penalizes
``||x0_corr - x0_base||^2`` (drift-gap normalized) so the corrector moves
*statistics* toward the clean teacher without rewriting *content* — the
anti-repeat / anti-hallucination seatbelt. Ramped linearly over
``FID_WARMUP`` steps."""

FID_WARMUP = 100

TARGETS = tuple(
    t.strip()
    for t in os.environ.get(
        "TARGETS", "self_attn.q,self_attn.k,self_attn.v,self_attn.o"
    ).split(",")
)
"""LoRA target projections. ``self_attn.q,self_attn.k`` gives the
attention-routing-only variant (no content-writing v/o adapters); note a
narrowed target set cannot init from a full-target checkpoint."""

GRAD_CLIP = 1.0
RANK = 16
EVAL_EVERY = 100
SAVE_EVERY = 200
CLEAN_LAP = 1
SEED = int(os.environ.get("SEED", "0"))


def main() -> None:
    pools = [sorted(p.glob("clip_*.pt")) for p in POOLS]
    pools = [p for p in pools if p]
    assert pools, f"no clips under any of {POOLS}"
    rng = np.random.default_rng(SEED)

    t_probs = np.array(T_WEIGHTS, dtype=np.float64)
    t_probs = t_probs / t_probs.sum()

    meta = torch.load(pools[0][0], map_location="cpu", weights_only=False)
    runner = build_runner(
        num_chunk=meta["num_chunk"],
        pose=meta["pose"],
        output_dir=CKPT.parent,
        compile_network=False,
    )
    pipe = runner.pipeline
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps, sigmas = scheduler.timesteps, scheduler.sigmas
    n_steps = len(timesteps) - 1

    cfg = runner.config
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)

    network = transformer.network
    if hasattr(network, "_orig_mod"):
        network = network._orig_mod
    apply_lora(network, rank=RANK, targets=TARGETS)
    if INIT and INIT != "scratch":
        load_lora(network, INIT)
    patch_functional_attention()
    checkpoint_blocks(network)
    params = lora_parameters(network)
    print(f"v2 init from {INIT} | pools {[len(p) for p in pools]} clips", flush=True)
    opt = torch.optim.AdamW(params, lr=LR)

    datas = [[load_clip(p, "cpu", dtype) for p in pool] for pool in pools]
    n0 = len(datas[0])
    nval = max(2, n0 // 8)
    train_ids = list(range(n0 - nval))
    val_ids = list(range(n0 - nval, n0))

    def ks_for(d: dict) -> list[int]:
        """Chunks whose memory window is fully past the clean lap (per-clip:
        mixed-geometry pools carry different lap sizes / rollout lengths).
        Leaves room for the k+1 contraction chunk."""
        lap_chunks = d["lap_latents"] // 4
        return list(range((CLEAN_LAP + 2) * lap_chunks, d["num_chunk"] - 1))

    def predict(cachet, ctrl, z_t: Tensor, t_idx: int) -> Tensor:
        flow = transformer.predict_flow(
            noisy_latent=z_t,
            timestep=timesteps[t_idx].to(dtype),
            cache=cachet,
            input=ctrl,
        )
        return z_t.float() - float(sigmas[t_idx]) * flow.float()

    def prefill_only(cachet, ctrl, history: Tensor, k: int) -> None:
        start_probe_chunk(cachet, ar_idx=k, history=history)
        with torch.no_grad():
            if ctrl.memory_frame_indices:
                transformer.prefill_memory_kv_cache(
                    cache=cachet, input=ctrl, timestep=timesteps[0].to(dtype)
                )
                cachet.prefill_completed_for_chunk = k

    def teacher_and_base(cachet, ctrl, h_clean, h_gen, z_t, t_idx, k):
        """Frozen-model x0 under clean and drifted history (no grad)."""
        with torch.no_grad():
            set_lora_scale(network, 0.0)
            prefill_only(cachet, ctrl, h_clean, k)
            x0_clean = predict(cachet, ctrl, z_t, t_idx)
            finish_probe_chunk(cachet, ar_idx=k)
            prefill_only(cachet, ctrl, h_gen, k)
            x0_base = predict(cachet, ctrl, z_t, t_idx)
            finish_probe_chunk(cachet, ar_idx=k)
        return x0_clean, x0_base

    def sample_losses(
        pool_id: int, c: int, grad: bool, fid_w: float = 0.0
    ) -> tuple[Tensor, Tensor]:
        """One sample -> (dagger loss, contraction loss)."""
        # Pools differ in size (round-1 regenerates a train-split prefix);
        # fold the pool-0 index into the smaller pool. Val ids exist only in
        # pool 0, so this cannot leak validation clips into training.
        c = c % len(datas[pool_id])
        d = {
            key: (v.to(device) if isinstance(v, Tensor) else v)
            for key, v in datas[pool_id][c].items()
        }
        lap = d["lap_latents"]
        k = int(rng.choice(ks_for(d)))
        t_idx = int(rng.choice(n_steps, p=t_probs))
        ctrl = make_ctrl(d, k, device=device, dtype=dtype)
        h_gen = history_of(d, k)
        h_clean = clean_counterfactual(
            h_gen,
            selected=d["memory_frame_indices"][k],
            lap_latents=lap,
            clean_lap=CLEAN_LAP,
        )
        x0 = chunk_x0(d, k)
        sig = sigmas[t_idx].to(dtype)
        z_t = (1 - sig) * x0 + sig * torch.randn(x0.shape, device=device, dtype=dtype)

        x0_clean, x0_base = teacher_and_base(tc, ctrl, h_clean, h_gen, z_t, t_idx, k)
        rt_sq = (x0_clean - x0_base).square().sum()

        if not grad:
            set_lora_scale(network, 1.0)
            prefill_only(tc, ctrl, h_gen, k)
            with torch.no_grad():
                x0_corr = predict(tc, ctrl, z_t, t_idx)
            finish_probe_chunk(tc, ar_idx=k)
            l_dag = (x0_corr - x0_clean).square().sum() / (rt_sq + 1e-8)
            return l_dag, torch.zeros((), device=device)

        # Grad path: two-phase backward so only one chunk's graph is ever
        # live (a joint k / k+1 graph OOMs next to co-tenant jobs, and
        # checkpoint recomputation forbids sharing one cache across both).
        #
        # Phase 1 -- contraction: run the student at k WITHOUT grad, make
        # its x0 a leaf, splice it into chunk k+1's history, run the
        # grad-capable committed prefill + k+1 forward, and backward
        # ``CW_LOSS * l_con`` immediately. This accumulates the k+1-path
        # LoRA grads and yields ``leaf.grad``.
        set_lora_scale(network, 1.0)
        prefill_only(tc, ctrl, h_gen, k)
        with torch.no_grad():
            x0_ng = predict(tc, ctrl, z_t, t_idx)
        finish_probe_chunk(tc, ar_idx=k)
        leaf = x0_ng.detach().requires_grad_(True)

        k2 = k + 1
        sel2 = d["memory_frame_indices"][k2] or []
        touches_k = any(k * 4 <= idx < (k + 1) * 4 for idx in sel2)
        l_con = torch.zeros((), device=device)
        if touches_k:
            ctrl2 = make_ctrl(d, k2, device=device, dtype=dtype)
            t2_idx = int(rng.choice(n_steps, p=t_probs))
            x0_next = chunk_x0(d, k2)
            sig2 = sigmas[t2_idx].to(dtype)
            z2 = (1 - sig2) * x0_next + sig2 * torch.randn(
                x0_next.shape, device=device, dtype=dtype
            )
            h2_gen = history_of(d, k2)
            h2_clean = clean_counterfactual(
                h2_gen,
                selected=sel2,
                lap_latents=lap,
                clean_lap=CLEAN_LAP,
            )
            x0_clean2, x0_base2 = teacher_and_base(
                tc, ctrl2, h2_clean, h2_gen, z2, t2_idx, k2
            )
            rt2_sq = (x0_clean2 - x0_base2).square().sum()

            h2_committed = h2_gen.clone()
            s = slice(k * 4 * TOKENS_PER_FRAME, (k + 1) * 4 * TOKENS_PER_FRAME)
            h2_committed[..., s, :] = leaf.to(dtype)
            set_lora_scale(network, 1.0)
            start_probe_chunk(tc, ar_idx=k2, history=h2_committed)
            with torch.enable_grad():
                if ctrl2.memory_frame_indices:
                    transformer.prefill_memory_kv_cache(
                        cache=tc, input=ctrl2, timestep=timesteps[0].to(dtype)
                    )
                    tc.prefill_completed_for_chunk = k2
                x0_corr2 = predict(tc, ctrl2, z2, t2_idx)
                l_con = (x0_corr2 - x0_clean2).square().sum() / (rt2_sq + 1e-8)
            (CW_LOSS * l_con).backward()
            finish_probe_chunk(tc, ar_idx=k2)
            l_con = l_con.detach()

        # Phase 2 -- dagger + chain rule: rerun the student at k WITH grad
        # (deterministic, so it reproduces ``x0_ng`` exactly) and route the
        # contraction gradient through it via the dot-product trick:
        # d/dtheta [x0_corr . leaf.grad] == (dl_con/dx0) (dx0/dtheta).
        set_lora_scale(network, 1.0)
        prefill_only(tc, ctrl, h_gen, k)
        with torch.enable_grad():
            x0_corr = predict(tc, ctrl, z_t, t_idx)
            l_dag = (x0_corr - x0_clean).square().sum() / (rt_sq + 1e-8)
            loss_k = l_dag
            if fid_w > 0:
                # Trust region: correct statistics without rewriting content.
                loss_k = loss_k + fid_w * (
                    (x0_corr - x0_base).square().sum() / (rt_sq + 1e-8)
                )
            if leaf.grad is not None:
                loss_k = loss_k + (x0_corr * leaf.grad.detach()).sum()
        loss_k.backward()
        finish_probe_chunk(tc, ar_idx=k)
        return l_dag.detach(), l_con

    @torch.no_grad()
    def val_loss(n: int = 8) -> float:
        s = 0.0
        for _ in range(n):
            l_dag, _ = sample_losses(0, int(rng.choice(val_ids)), grad=False)
            s += l_dag.item()
        return s / n

    torch.set_grad_enabled(True)
    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        pool_id = int(rng.integers(len(pools)))
        c = int(rng.choice(train_ids))
        # Backwards happen inside sample_losses (two-phase; grads
        # accumulate into .grad).
        fid_w = FID_W * min(1.0, step / FID_WARMUP)
        l_dag, l_con = sample_losses(pool_id, c, grad=True, fid_w=fid_w)
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            vl = val_loss()
            print(
                f"step {step:4d} | dag {l_dag.item():.4f} | con {float(l_con):.4f} | "
                f"val dag-loss {vl:.4f} (R^2 {1 - vl:+.3f})",
                flush=True,
            )
        if step % SAVE_EVERY == 0 or step == STEPS:
            save_lora(network, CKPT)

    vl = val_loss(16)
    print(
        f"v2 done | final val dag-loss {vl:.4f} (R^2 {1 - vl:+.3f}) | saved {CKPT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
