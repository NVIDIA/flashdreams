<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint 1.5 V2 validation

Validated on 2026-08-25 against FlashDreams main `8fd97fa3`, source PR #464
`0f178234`, and the official `world_engine` implementation at
`b3f1e725dedac17ccbfaf9ee37f5e068bb44bed4`.

## Artifacts and hardware

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GiB), driver
  595.84.
- PyTorch: 2.12.1+cu130, CUDA 13.0.
- Waypoint revision: `391f92827075edcf4a8b3c8a2ddae010698f8636`.
- Waypoint SHA-256:
  `b872ad07968bae082a120a29072e61a13565086f042384ad7fdb79a7b0c50994`.
- Waypoint inventory: 393 BF16 tensors and 1,860,823,096 elements.
- TAEHV revision: `a0253886b13b9c4c3bd224bd479be03f5988a3df`.
- TAEHV SHA-256:
  `806f78a06266ba58bb982d8680add1a249aefcc96237e8a0aa60298617744682`.

Large outputs remain in the ignored local directory `artifacts/waypoint-pr464/`.
The initial 40-action MP4 decodes as exactly 164 frames (four seed frames plus
40 four-frame actions), with no duplicate or dropped frames. Additional scene
and long-rollout outputs are under its `additional-inference/` directory.

## Automated gates

- Ruff 0.12.7 check and format: passed.
- ty 0.0.53 for both integration packages: passed.
- CPU model, TAEHV, V2 lifecycle, reset, input, and MP4 tests: 51 passed and
  7 deselected.
- CUDA fixed-cache FlexAttention equivalence, local and global layers over
  eight autoregressive frames: 2 passed.
- Built-wheel entry-point discovery: `waypoint-1-5-1b` resolves to
  `WaypointApplication`; all 118 bundled actions are available.

## Real V2 inference

The real application ran with `--example-data --actions 40 --seed 464
--device cuda --profile`, using V2 `ApplicationRunner`, `Mp4ClientWindow`, and
`MetricsOutputSink`. It loaded the published checkpoints, established action
zero from the pinned seed image, generated actions 1 through 40, finalized each
cache entry, and encoded every presented frame.

| AR action | Diffuse | Decode | Finalize | Total | Peak allocated |
|---:|---:|---:|---:|---:|---:|
| 20 | 70.383 ms | 1.889 ms | 17.199 ms | 89.487 ms | 6.049 GiB |
| 40 | 71.909 ms | 1.863 ms | 17.623 ms | 91.413 ms | 6.049 GiB |

Actions 20 through 40 average 91.712 ms per four generated frames, or 43.61
generated frames/s. The session's 60 FPS value is playback timing, not measured
generation throughput. For context, the PR discussion reported 286.04 ms at
action 20 and 235.39 ms at action 40 on an RTX 5090; this comparison is not
hardware-normalized.

## Official implementation parity

Parity uses the same BF16 checkpoint and inputs in the integrated transformer
and the pinned official `world_engine`. The official reference must use its
inference patches and compiled FlexAttention path; its eager FlexAttention
fallback warns that it materializes dense scores and does not produce the
runtime kernel's result. Before attention, patchify, noise and control
embeddings, RoPE, adaptive projections, normalized tokens, Q/K/V, rotated Q/K,
cache tensors, and active mask blocks are bit-identical.

Two independently committed flow-prediction frames reached cosine similarity
of at least 0.999902, mean absolute error at most 0.011621, and max absolute
error at most 0.09375. A stronger seeded rollout comparison established action
zero in both caches and ran one controlled action through the complete official
four-step BF16 Euler schedule:

| Comparison | Mean absolute error | Max absolute error | Cosine similarity |
|---|---:|---:|---:|
| Seed cache flow | 0.010793 | 0.218750 | 0.999586 |
| Clean latent after four steps | 0.005682 | 0.033691 | 0.999869 |
| Final cache flow | 0.010511 | 0.265625 | 0.999487 |

The remaining BF16 difference is consistent with whole-model fusion in the
official compiled graph versus separately compiled attention in FlashDreams.

## Additional scene and long-rollout validation

All 15 seed images from the pinned Biome checkout were each run for 40 actions.
The original pinned example was also run through all 118 bundled actions. One
loaded application and model served fresh sequential sessions, exercising both
model reuse and per-session image cache/RNG isolation.

- 16 runs completed, totaling 718 generated actions and 2,936 decoded frames.
- Every metrics sequence was complete, ordered, and finite.
- Every 40-action MP4 decoded as exactly 164 frames; the 118-action MP4 decoded
  as exactly 476 frames.
- Downsampled seed and final frames were nonblank. Seed-to-final mean absolute
  pixel difference ranged from 21.09 to 51.73 on the 0-255 scale.
- Action-40 latency across the 15 scenes averaged 90.986 ms, ranging from
  89.029 to 92.622 ms.
- Action 118 completed in 91.479 ms. Maximum measured peak allocation across
  the batch was 6.355 GiB.

All scenes used seed 464 and the same bundled control timeline. This validates
execution, cache longevity, scene diversity, output encoding, and stable
performance; it is not a qualitative gameplay ranking of the scenes.
