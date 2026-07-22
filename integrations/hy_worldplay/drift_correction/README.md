<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Clean Forcing drift corrector for HY-WorldPlay

Long autoregressive rollouts drift: each chunk conditions on self-generated
history and small errors compound (saturation runaway, texture mush). This
module ships a trained corrector — a frozen-base LoRA (r16 on the
self-attention q/k/v/o projections, ~0.3% params) trained from the model's
own closed-loop rollouts with a counterfactual clean-history teacher, zero
real videos — plus the full recipe that produced it.

## Deploy

Point the runner at a corrector checkpoint; everything else is automatic:

```python
config = dataclasses.replace(
    RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
    drift_corrector=Path("lora_v2.pt"),  # None (default) = exact base behavior
    drift_corrector_gain=0.5,            # composed with the per-step alpha*(t) gate
)
```

Selection is content-keyed per job: static (locked-off camera) trajectories
run the untouched base weights — static scenes measure *negative* drift, so
correction there is pure artifact cost — while commanded-motion trajectories
apply the LoRA at ``alpha*(t) x gain`` per denoise step. The LoRA stays
unfused on motion jobs (a single-scale weight merge cannot express the
per-timestep gate); static jobs skip module surgery entirely.

The trained v2 checkpoint (~30 MB ``.pt``) is distributed separately; see
the PR / release notes for the download link.

## Reproduce the corrector

All scripts run from the repo root on one GPU (~2.5 GPU-days end to end):

```bash
python drift_correction/gen_first_frames.py     # optional T2V seeds (PROMPTS_FILE=...)
python drift_correction/build_pairs.py          # strafe-loop rollouts -> pair clips
python drift_correction/gate_faithful.py        # step-0 alpha*(t) go/no-go diagnostic
python drift_correction/train_v1.py             # counterfactual-teacher LoRA
python drift_correction/train_v2.py             # DAgger pool + drift-contraction round
python drift_correction/eval_rollouts.py        # gain-sweep rollouts
python drift_correction/score_drift.py          # drift / dynamics / progression / seams
python drift_correction/demo_static.py          # static-scene suite (+ make_sbs.py)
```

Train only if the gate passes (the drift gap is systematic: ``alpha*`` high);
the measured ``alpha*(t)`` profile doubles as the deploy gate in
``hy_worldplay/_drift_corrector.py``.

## Design notes and background

- Library shape (host adapter vs host-agnostic core), gate math, and the
  cross-host playbook: ``LIBRARY_DESIGN.md`` on the working branch
  (`wenqingw-nv/flashdreams-wq` @ ``hy-worldplay-counterfactual-forcing``).
- Method, result tables, and paper-host reproduction:
  https://gitlab-master.nvidia.com/wenqingw/clean_forcing
