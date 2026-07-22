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

"""v1 corrector trainer: counterfactual clean-history teacher in x0 space.

Trains the LoRA corrector r_phi on the frozen distilled HY-WorldPlay model
so its x0 prediction under drifted history matches the frozen model's
prediction under the lap-aligned clean history at the same ``z_t``::

    L = ||x0_{theta+LoRA}(z_t, h_gen, t) - x0_theta(z_t, h_clean, t)||^2
        / ||x0_theta(z_t, h_clean, t) - x0_theta(z_t, h_gen, t)||^2

The drift-gap denominator is required (raw MSE diverges); ``R^2 = 1 - L``.
Both passes share the exact ``z_t`` re-noised from the rollout's own x0
(native anchoring). The student's memory-KV prefill runs LoRA-scaled but
under ``no_grad`` (gradients flow through the current-chunk forward only;
deploy-time behaviour is unaffected since the weights are shared).

Host adaptations vs the Wan2.1 reference (``wan_train_synth.py``):
x0-space predictions, t sampled from the 4 distillation timesteps, eager
network (``compile_network=False``), functional dual-branch attention
(:mod:`_train_attn` -- restores k/v gradients, enables checkpointing), and
per-block gradient checkpointing (the fp32/dual-branch tape is ~40 GiB
without it).

Run from the repo root (resumable via ``INIT``)::

    STEPS=1500 .venv/bin/python integrations/hy_worldplay/drift_correction/train_v1.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from _lora import apply_lora, load_lora, lora_parameters, save_lora, set_lora_scale
from _pairs import chunk_x0, clean_counterfactual, history_of, load_clip, make_ctrl
from _rollout import build_runner, finish_probe_chunk, start_probe_chunk
from _train_attn import patch_functional_attention
from hy_worldplay._action import HyWorldPlayWan21TransformerCache
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame

## Training configuration

PAIRS_DIR = Path(
    os.environ.get(
        "PAIRS_DIR", "integrations/hy_worldplay/drift_correction/outputs/pairs"
    )
)
"""Clip files from ``build_pairs.py``."""

CKPT = Path(
    os.environ.get(
        "CKPT", "integrations/hy_worldplay/drift_correction/outputs/lora_v1.pt"
    )
)
"""Output LoRA checkpoint; saved every ``SAVE_EVERY`` steps."""

INIT = os.environ.get("INIT", "")
"""Optional LoRA checkpoint to resume/init from."""

STEPS = int(os.environ.get("STEPS", "1500"))
LR = float(os.environ.get("LR", "5e-4"))
WARMUP = 60
"""Linear LR warmup steps; required for stability (reference finding)."""

GRAD_CLIP = 1.0
RANK = 16
EVAL_EVERY = 100
SAVE_EVERY = 250
CLEAN_LAP = 1
SEED = int(os.environ.get("SEED", "0"))


def checkpoint_blocks(network) -> None:
    """Route block ``forward`` and ``prefill_memory_kv`` through checkpointing.

    Per-instance overrides (not wrapper modules) so the network loop's
    ``isinstance(block, Block)`` assertion keeps passing. Requires the
    functional-attention patch: forward recomputation must be side-effect
    free. The prefill's ``write_rope`` / ``write_prope`` side effects are
    plain re-assignments of deterministically recomputed tensors, so its
    recompute is value-stable. No-op under ``no_grad`` passes.
    """
    from torch.utils.checkpoint import checkpoint

    def wrap(fn):
        def ckpt_fn(*args, _inner=fn, **kwargs):
            if not torch.is_grad_enabled():
                return _inner(*args, **kwargs)
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        return ckpt_fn

    for block in network.blocks:
        block.forward = wrap(block.forward)
        block.prefill_memory_kv = wrap(block.prefill_memory_kv)


def main() -> None:
    clips = sorted(PAIRS_DIR.glob("clip_*.pt"))
    assert clips, f"no clips under {PAIRS_DIR}; run build_pairs.py first"
    rng = np.random.default_rng(SEED)

    meta = torch.load(clips[0], map_location="cpu", weights_only=False)
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
    wrapped = apply_lora(network, rank=RANK)
    if INIT:
        load_lora(network, INIT)
        print(f"LoRA init from {INIT}", flush=True)
    # Functional attention: restores k/v gradients (the stock path severs
    # them at the rolling-cache write) and makes per-block checkpointing
    # sound (no mutable storage in the tape -- the fp32 residual/dual-branch
    # tape is ~40 GiB unchunked).
    patch_functional_attention()
    checkpoint_blocks(network)
    params = lora_parameters(network)
    n_params = sum(p.numel() for p in params)
    print(
        f"LoRA on {len(wrapped)} projections | {n_params / 1e6:.2f}M params", flush=True
    )
    opt = torch.optim.AdamW(params, lr=LR)

    # Clip latents stay on CPU; a draw moves one clip's tensors to device.
    datas = [load_clip(p, "cpu", dtype) for p in clips]
    nval = max(2, len(datas) // 8)
    train_ids = list(range(len(datas) - nval))
    val_ids = list(range(len(datas) - nval, len(datas)))
    lap_chunks = datas[0]["lap_latents"] // 4
    num_chunk = datas[0]["num_chunk"]
    # Windows fully past the clean lap: selected recent-16 frames all drifted.
    ks = list(range((CLEAN_LAP + 2) * lap_chunks, num_chunk))
    print(
        f"{len(datas)} clips ({len(val_ids)} val) | k in {ks[0]}..{ks[-1]}", flush=True
    )

    def predict(ctrl, z_t: Tensor, t_idx: int) -> Tensor:
        t = timesteps[t_idx].to(dtype)
        flow = transformer.predict_flow(
            noisy_latent=z_t, timestep=t, cache=tc, input=ctrl
        )
        return z_t.float() - float(sigmas[t_idx]) * flow.float()

    def prefill_only(ctrl, history: Tensor, k: int) -> None:
        """Open the chunk bracket and run the memory prefill without grad."""
        start_probe_chunk(tc, ar_idx=k, history=history)
        with torch.no_grad():
            if ctrl.memory_frame_indices:
                transformer.prefill_memory_kv_cache(
                    cache=tc, input=ctrl, timestep=timesteps[0].to(dtype)
                )
                tc.prefill_completed_for_chunk = k

    def sample_losses(c: int, grad: bool) -> tuple[Tensor, Tensor]:
        """One drift-pair sample -> (normalized loss, r_target sq-norm)."""
        d = {
            key: (v.to(device) if isinstance(v, Tensor) else v)
            for key, v in datas[c].items()
        }
        k = int(rng.choice(ks))
        t_idx = int(rng.integers(n_steps))
        selected = d["memory_frame_indices"][k]
        ctrl = make_ctrl(d, k, device=device, dtype=dtype)
        h_gen = history_of(d, k)
        h_clean = clean_counterfactual(
            h_gen, selected=selected, lap_latents=d["lap_latents"], clean_lap=CLEAN_LAP
        )
        x0 = chunk_x0(d, k)
        sig = sigmas[t_idx].to(dtype)
        z_t = (1 - sig) * x0 + sig * torch.randn(x0.shape, device=device, dtype=dtype)

        with torch.no_grad():
            set_lora_scale(network, 0.0)
            prefill_only(ctrl, h_clean, k)
            x0_clean = predict(ctrl, z_t, t_idx)
            finish_probe_chunk(tc, ar_idx=k)
            prefill_only(ctrl, h_gen, k)
            x0_base = predict(ctrl, z_t, t_idx)
            finish_probe_chunk(tc, ar_idx=k)
        r_target_sq = (x0_clean - x0_base).square().sum()

        set_lora_scale(network, 1.0)
        prefill_only(ctrl, h_gen, k)  # LoRA-scaled prefill, no grad
        with torch.enable_grad() if grad else torch.no_grad():
            x0_corr = predict(ctrl, z_t, t_idx)
            loss = (x0_corr - x0_clean).square().sum() / (r_target_sq + 1e-8)
        if not grad:
            finish_probe_chunk(tc, ar_idx=k)
        return loss, r_target_sq

    @torch.no_grad()
    def val_r2(n: int = 8) -> float:
        s = 0.0
        for _ in range(n):
            loss, _ = sample_losses(int(rng.choice(val_ids)), grad=False)
            s += loss.item()
        return 1 - s / n

    torch.set_grad_enabled(True)
    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, step / WARMUP)
        opt.zero_grad()
        loss, rt_sq = sample_losses(int(rng.choice(train_ids)), grad=True)
        loss.backward()
        # Backward precedes the bracket close: checkpoint recomputation
        # needs the chunk's KV state alive.
        finish_probe_chunk(tc, ar_idx=tc.autoregressive_index)
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            print(
                f"step {step:5d} | loss {loss.item():.4f} (train R^2 {1 - loss.item():+.3f})"
                f" | val R^2 {val_r2():+.3f} | |r_t|^2 {rt_sq.item():.1f}",
                flush=True,
            )
        if step % SAVE_EVERY == 0 or step == STEPS:
            save_lora(network, CKPT)

    print(f"final val R^2 {val_r2(16):+.3f} | saved {CKPT}", flush=True)


if __name__ == "__main__":
    main()
