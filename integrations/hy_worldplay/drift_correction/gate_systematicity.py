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

"""Step-0 go/no-go gate: history-sensitivity + systematicity (alpha*) on HY-WorldPlay.

Counterfactual Forcing assumes the prediction error induced by a perturbed
history at a matched noisy state ``z_t`` is *systematic* -- reproducible
across the noise seeds of ``z_t`` -- so a deterministic corrector can remove
it. This gate measures, per distillation timestep ``t`` and chunk ``k``::

    delta_m   = x0(z_t^m, h_gen) - x0(z_t^m, h_alt)      m = 1..M noise seeds
    alpha*(t) = ||mean_m delta_m||^2 / (||mean_m delta_m||^2 + var_m)
    rel(t)    = mean_m ||delta_m|| / mean_m ||x0(z_t^m, h_gen)||

for three history perturbations ``h_alt``: a condition-matched history from an
independent same-pose rollout (``cross_seed``), and drift-like per-channel
scale/shift corruptions at two strengths. ``rel`` near zero means the
prediction ignores history (corrector cannot help -- STOP); ``alpha*`` >= 0.7
at most cells means the history-induced gap is systematic (GO). Reference
host measured 0.91-0.99.

Resumable: rollout captures are cached under ``outputs/gate/`` and reused.
Run from the repo root::

    .venv/bin/python integrations/hy_worldplay/drift_correction/gate_systematicity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from _rollout import (
    ChunkSnapshot,
    build_runner,
    capture_rollout,
    finish_probe_chunk,
    load_rollout,
    predict_x0,
    start_probe_chunk,
)
from hy_worldplay._action import HyWorldPlayWan21TransformerCache
from hy_worldplay.runner import _resolve_prompt, preprocess_first_frame
from torch import Tensor

## Gate configuration

NUM_CHUNK = 12
"""Rollout horizon in AR chunks (48 latent frames); long enough that late
chunks condition on FOV-selected memory with visible accumulated drift."""

POSE = "w-20, right-2, w-10, left-2, w-13"
"""47 motion steps -> 48 latents, exactly ``NUM_CHUNK * 4``. Mostly-forward
with two gentle turns so the FOV selector sees a production-like trajectory."""

ROLLOUT_SEEDS = (42, 1042)
"""Diffusion-noise seeds for the two independent rollouts; rollout A's history
is ``h_gen``, rollout B's supplies the condition-matched ``cross_seed``
counterfactual."""

PROBE_CHUNKS = (4, 8, 11)
"""Chunks probed: first FOV-selected chunk (frame 16), mid, and last."""

M_NOISE = 8
"""Noise seeds per (chunk, timestep) cell for the bias/variance decomposition.
The naive alpha* estimator inflates by ``(1 - alpha*) / M``; 8 seeds plus the
unbiased estimator below keep borderline cells out of threshold noise."""

CORRUPT_STRENGTHS = (0.15, 0.30)
"""Per-channel scale/shift corruption strengths (drift-like proxy, matches the
reference gate's ``drift_corrupt``)."""

LATENT_CHANNELS = 48
"""Wan 2.2 TI2V-5B latent channels; the patchified feature axis is laid out
``(c kt kh kw)``, so channel ``c`` owns the contiguous 4-slot span ``4c:4c+4``."""

OUT_DIR = Path("integrations/hy_worldplay/drift_correction/outputs/gate")
"""Gate artifacts: rollout captures, MP4s, and the results JSON."""


## History perturbations


def corrupt_history(history: Tensor, strength: float, seed: int) -> Tensor:
    """Apply drift-like per-latent-channel scale + shift to a patchified history.

    Args:
        history: Patchified history ``[..., L, C * kt * kh * kw]``.
        strength: Relative scale/shift magnitude.
        seed: Generator seed so each (chunk, strength) cell perturbs identically
            across noise seeds.

    Returns:
        Corrupted history with the same shape/dtype.
    """
    g = torch.Generator(device=history.device).manual_seed(seed)
    spatial = history.shape[-1] // LATENT_CHANNELS
    h = history.view(*history.shape[:-1], LATENT_CHANNELS, spatial)
    scale = 1 + strength * torch.randn(
        LATENT_CHANNELS, 1, device=history.device, dtype=history.dtype, generator=g
    )
    shift = (
        strength
        * history.float().std().to(history.dtype)
        * torch.randn(
            LATENT_CHANNELS, 1, device=history.device, dtype=history.dtype, generator=g
        )
    )
    return (h * scale + shift).view_as(history)


## Probe sweep


@torch.no_grad()
def probe_cell(
    transformer,
    tc: HyWorldPlayWan21TransformerCache,
    snap: ChunkSnapshot,
    *,
    ar_idx: int,
    history: Tensor,
    z_ts: dict[int, list[Tensor]],
    timesteps: Tensor,
    sigmas: Tensor,
    dtype: torch.dtype,
) -> dict[int, list[Tensor]]:
    """Predict x0 for every (timestep, noise-seed) probe under one history.

    One ``start_probe_chunk`` per history keeps a single memory-KV prefill
    for the whole sweep, matching how the sampler reuses it across steps.

    Returns:
        ``{t_idx: [x0 per noise seed]}`` fp32 tensors.
    """
    start_probe_chunk(tc, ar_idx=ar_idx, history=history)
    out: dict[int, list[Tensor]] = {}
    for t_idx, z_list in z_ts.items():
        t = timesteps[t_idx].to(dtype)
        sigma = float(sigmas[t_idx])
        out[t_idx] = [
            predict_x0(transformer, tc, ctrl=snap.ctrl, z_t=z, timestep=t, sigma=sigma)
            for z in z_list
        ]
    finish_probe_chunk(tc, ar_idx=ar_idx)
    return out


def main() -> None:
    # The gate is inference-only; predict_flow is called outside the
    # pipeline's no_grad-decorated entry points, and a taped 30-block
    # forward retains ~50 GiB of activations.
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = build_runner(num_chunk=NUM_CHUNK, pose=POSE, output_dir=OUT_DIR)
    pipe = runner.pipeline
    device = next(pipe.parameters()).device
    dtype = next(pipe.parameters()).dtype

    # Capture (or reload) the two independent rollouts.
    rollouts: dict[int, list[ChunkSnapshot]] = {}
    for seed in ROLLOUT_SEEDS:
        path = OUT_DIR / f"rollout_seed{seed}.pt"
        if path.exists():
            print(f"reusing capture {path}", flush=True)
            rollouts[seed] = load_rollout(path, device)
        else:
            print(f"capturing rollout seed={seed} ...", flush=True)
            rollouts[seed] = capture_rollout(
                runner,
                noise_seed=seed,
                save_path=path,
                mp4_path=OUT_DIR / f"rollout_seed{seed}.mp4",
            )
    snaps_a, snaps_b = rollouts[ROLLOUT_SEEDS[0]], rollouts[ROLLOUT_SEEDS[1]]

    # Fresh cache for probing (text embeddings + layout only; rolling caches
    # are reset per probe chunk).
    cfg = runner.config
    assert cfg.image_path is not None
    image = preprocess_first_frame(
        cfg.image_path, cfg.pixel_height, cfg.pixel_width
    ).to(device=device, dtype=dtype)
    cache = pipe.initialize_cache(text=[_resolve_prompt(cfg.prompt)], image=image)
    tc = cache.transformer_cache
    assert isinstance(tc, HyWorldPlayWan21TransformerCache)
    transformer = pipe.diffusion_model.transformer
    scheduler = pipe.diffusion_model.scheduler
    timesteps = cast(Tensor, scheduler.timesteps)
    sigmas = cast(Tensor, scheduler.sigmas)
    n_steps = len(timesteps) - 1  # trailing entry is the terminal t=0

    results: dict[str, dict] = {}
    for k in PROBE_CHUNKS:
        h_gen = snaps_a[k].history
        assert h_gen is not None
        variants = {
            "cross_seed": snaps_b[k].history,
            **{
                f"corrupt_{s:g}": corrupt_history(h_gen, s, seed=1000 * k)
                for s in CORRUPT_STRENGTHS
            },
        }

        # Shared z_t per (t, m): the anchor state comes from rollout A's own
        # x0 for this chunk, re-noised with the host's convention.
        x0 = snaps_a[k].clean_latent
        z_ts: dict[int, list[Tensor]] = {}
        for t_idx in range(n_steps):
            sig = sigmas[t_idx].to(dtype)
            z_ts[t_idx] = []
            for m in range(M_NOISE):
                g = torch.Generator(device=device).manual_seed(
                    10_000 * k + 100 * t_idx + m
                )
                eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=g)
                z_ts[t_idx].append((1 - sig) * x0 + sig * eps)

        x0_gen = probe_cell(
            transformer,
            tc,
            snaps_a[k],
            history=h_gen,
            ar_idx=k,
            z_ts=z_ts,
            timesteps=timesteps,
            sigmas=sigmas,
            dtype=dtype,
        )
        results[str(k)] = {}
        for name, h_alt in variants.items():
            assert h_alt is not None
            x0_alt = probe_cell(
                transformer,
                tc,
                snaps_a[k],
                history=h_alt,
                ar_idx=k,
                z_ts=z_ts,
                timesteps=timesteps,
                sigmas=sigmas,
                dtype=dtype,
            )
            alphas, alphas_ub, rels = [], [], []
            for t_idx in range(n_steps):
                deltas = torch.stack(
                    [a - b for a, b in zip(x0_gen[t_idx], x0_alt[t_idx])]
                )
                m = deltas.shape[0]
                bias = deltas.mean(dim=0)
                bias_sq = bias.square().sum().item()
                sq_dev = (deltas - bias).square().sum(dim=tuple(range(1, deltas.ndim)))
                var = sq_dev.mean().item()
                # Unbiased decomposition: E||mean||^2 = ||bias||^2 + var/M and
                # E[mean sq dev] = var (M-1)/M, so correct both before the ratio.
                var_ub = sq_dev.sum().item() / (m - 1)
                bias_sq_ub = max(0.0, bias_sq - var_ub / m)
                gen_norm = torch.stack(x0_gen[t_idx]).flatten(1).norm(dim=1).mean()
                alphas.append(bias_sq / (bias_sq + var + 1e-12))
                alphas_ub.append(bias_sq_ub / (bias_sq_ub + var_ub + 1e-12))
                rels.append(
                    (deltas.flatten(1).norm(dim=1).mean() / (gen_norm + 1e-12)).item()
                )
            results[str(k)][name] = {
                "t": [float(timesteps[i]) for i in range(n_steps)],
                "alpha_star": alphas,
                "alpha_star_unbiased": alphas_ub,
                "rel": rels,
            }
            print(
                f"k={k:2d} {name:14s} | "
                + " | ".join(
                    f"t={int(timesteps[i]):4d} a*={alphas[i]:.3f}"
                    f"/{alphas_ub[i]:.3f} rel={rels[i]:.3f}"
                    for i in range(n_steps)
                ),
                flush=True,
            )

    (OUT_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    # GO / NO-GO summary over every (chunk, variant, timestep) cell.
    cells = [
        (a, r)
        for per_k in results.values()
        for v in per_k.values()
        for a, r in zip(v["alpha_star_unbiased"], v["rel"])
    ]
    frac_go = sum(a >= 0.7 for a, _ in cells) / len(cells)
    mean_alpha = sum(a for a, _ in cells) / len(cells)
    mean_rel = sum(r for _, r in cells) / len(cells)
    print("\n================ HY-WorldPlay systematicity gate ================")
    print(
        f"cells with unbiased alpha* >= 0.7: {frac_go:.0%} | mean alpha* "
        f"{mean_alpha:.3f} | mean rel gap: {mean_rel:.3f}"
    )
    if mean_rel < 0.01:
        print("RESULT: x0 ~insensitive to history -> corrector cannot help. STOP.")
    elif frac_go >= 0.7:
        print("RESULT: history-induced gap is systematic at most steps. GO.")
    else:
        print(
            "RESULT: gap is noise-dominated at many steps. Investigate before training."
        )
    print(f"saved {OUT_DIR / 'gate_results.json'}")


if __name__ == "__main__":
    main()
