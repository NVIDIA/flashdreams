<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Clean Forcing drift corrector for Omnidreams

Long autoregressive Omnidreams rollouts drift: each chunk conditions on
self-generated history and small errors compound (color/saturation runaway,
texture mush, content repetition). This module ships a trained corrector — a
frozen-base LoRA (r16 on the self-attention q/k/v/o projections, ~7.3M
params) trained from the model's own closed-loop rollouts with a
counterfactual clean-history teacher — plus the full recipe that produced it.

## Deploy

Point the runner at a corrector checkpoint; everything else is automatic:

```python
config = OmnidreamsRunnerConfig(
    ...,
    drift_corrector=Path("lora_v2_v3_valpeak.pt"),  # None (default) = exact base behavior
    # drift_corrector_gain defaults to the shipped 0.25
)
```

The hook (`omnidreams/_drift_corrector.py`) wraps the transformer's
self-attention projections with the LoRA and rescales it to
``alpha*(t) x gain`` before every denoise step, where ``alpha*(t)`` is the
measured per-timestep systematic share of the drift error (0.96 at t=1000,
0.667 at t=803 on this host). The LoRA stays unfused (the per-timestep gate
cannot merge into one weight set); ``drift_corrector=None`` skips module
surgery entirely.

The trained checkpoint (~88 MB ``.pt``,
md5 ``84beb8c014346833ce182310373b5d32``) is distributed separately; see the
PR / release notes for the download link.

Measured on held-out 21.5 s rollouts (704x1280, 81 chunks): drift
Delta-MUSIQ +0.99 vs base +2.44 (60% cut), late-window MUSIQ +5% over base,
overall MUSIQ above base, RAFT dynamics −17%.

## Reproduce the corrector

All scripts run from the repo root on one GPU (~1 GPU-day end to end; every
stage checkpoints and resumes). The pair rollouts need the
``nvidia/omni-dreams-samples`` HF dataset (real dashcam first frames + 80 s
HDMaps; gated — export ``HF_TOKEN``):

```bash
python drift_correction/build_pairs.py      # tiled-HDMap loop rollouts -> pair clips
                                            # (LAP_MIX=4,5,6 = shipped mixed-lap layout)
python drift_correction/gate_faithful.py    # step-0 alpha*(t) go/no-go diagnostic
python drift_correction/train_v1.py         # counterfactual-teacher LoRA (velocity space)
python drift_correction/build_pairs.py      # DAgger round: CORRECTOR_LORA=<v1 ckpt>
python drift_correction/train_v2.py         # DAgger pool + drift-contraction round
                                            # (SNAP_EVERY=200 keeps the val-peak snapshot
                                            #  — the shipped checkpoint is a val-peak)
python drift_correction/eval_rollouts.py    # dial-grid sweep on held-out scenes
```

Train only if the gate passes (the drift gap is systematic: ``alpha*``
high); the measured ``alpha*(t)`` profile doubles as the deploy gate in
``omnidreams/_drift_corrector.py``. Correctors encode a checkpoint-specific
error map — re-instantiation on a new base checkpoint is ~1 GPU-day with
this recipe.

Ablation flags kept in the scripts: ``PAIR_SCHEME=fork`` +
``build_pairs_v4.py`` (loop-free re-anchored fork counterparts; the pair
design gate-validates at ``alpha* 0.89``), ``UW=1`` (per-token
``alpha*``-weighted loss), ``TBIN_EVAL`` (per-timestep val R², feeds
``gain_predict.py``).

## Known issues (shipped config)

- Some repeated rows of buildings/shops and repeated trucks can appear in
  long rollouts (partial content-repetition prior from the loop-structured
  training pairs; the mixed-lap layout removes the fixed-period component).
- Mild softness vs base on some scenes (owner-reviewed and accepted; the
  variance-of-Laplacian sharpness ratio is NOT a reliable proxy on this
  host — judge with frame stacks).
- RAFT dynamic degree −17% vs base.

## Design notes and background

- Method, gate math, result tables, and the paper-host reproduction:
  https://gitlab-master.nvidia.com/wenqingw/clean_forcing
- The HY-WorldPlay port of the same recipe (content-keyed deploy):
  ``integrations/hy_worldplay/drift_correction/``.
