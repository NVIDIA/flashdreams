<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot-VA GPU and parity evidence

This record was produced on 2026-08-25 from FlashDreams baseline
`8fd97fa38f04bc32c288760fa0fbf5da52464cea` and this integration worktree. It
is validation evidence, not a general performance claim.

## Fixed inputs

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB.
- Driver: 595.84.
- PyTorch/CUDA: 2.12.1+cu130 / CUDA 13.0.
- Upstream source: `robbyant/lingbot-va` at
  `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`.
- Checkpoint: `robbyant/lingbot-va-posttrain-robotwin` at
  `8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`.
- Input PNGs: the upstream Robotwin example files and hashes recorded in
  `assets/example_data/lingbot-va/robotwin/README.md`.
- Precision/device: BF16, one CUDA device, seed 42.

The real checkpoint contained 841 transformer entries. Two obsolete
`patch_embedding.*` entries are intentionally dropped; all 839 remaining keys
mapped bijectively to the 839 native network entries. Strict loading produced a
5,088,872,670-parameter transformer.

## Upstream flow parity

`tools/compare_upstream.py` loads the pinned upstream and native transformers
sequentially, then runs the same first-chunk video followed by action tensors
through the same cache lifecycle. The explicit acceptance gates are maximum
absolute error <= 0.07 and mean absolute error <= 0.012 for both streams.

| Native mode | Stream | Maximum absolute error | Mean absolute error | RMS error |
| --- | --- | ---: | ---: | ---: |
| eager | video | 0.04296875 | 0.00751040 | 0.00951632 |
| eager | action | 0.06250000 | 0.00875314 | 0.01267146 |
| compiled | video | 0.05468750 | 0.00970979 | 0.01229867 |
| compiled | action | 0.06250000 | 0.01116651 | 0.01560142 |

The comparison caught and then regression-tested a defect where the native
action block loop received current-video KV but omitted it from attention. The
pre-fix action maximum/mean errors were 1.015625/0.203186. The table contains
the post-fix measurements.

Reproduce the compiled bound check from the repository root:

```bash
PYTHONPATH=flashdreams:integrations/lingbot_va \
python integrations/lingbot_va/tools/compare_upstream.py \
    --checkpoint-root /path/to/resolved/snapshot \
    --upstream-root /path/to/robbyant-lingbot-va-7c6ffa9 \
    --compile-native \
    --maximum-video-error 0.07 --mean-video-error 0.012 \
    --maximum-action-error 0.07 --mean-action-error 0.012
```

## Real multi-chunk V2 run

Matched GPU-resident and offloaded runs used two chunks, default CFG (video 5,
action 1), 25 video steps, 50 action steps, and compilation disabled to isolate
model correctness. Both produced:

- video `[13, 3, 256, 320]`, finite BF16;
- actions `[64, 16]`, finite float32;
- different stable chunk means: 0.179792 and 0.333868;
- a valid 13-frame, 320x256, 10 FPS H.264 MP4;
- byte-identical resident/offload video and action artifacts.

| Artifact | SHA-256 |
| --- | --- |
| `demo.mp4` | `df0c193137a673f4f8d6b2372b4bf7afc01a9937c4280b7fa5a51912b5e93c1a` |
| `actions.npy` | `463b307b667c1ca13a47bbbc5a17f68604621dfe3c3a10fc5860077216928d95` |

Fresh-process engine measurements were:

| Mode | Prompt | Observation | Denoise | Decode | Total | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resident | 0.240 s | 0.221 s | 4.735 s | 0.260 s | 33.220 s | 43,329,760,768 B (40.35 GiB) |
| offload | 5.055 s | 0.776 s | 4.744 s | 0.429 s | 34.009 s | 39,804,415,488 B (37.07 GiB) |

After `close`, only the process CUDA allocator context remained (about 0.031
GiB); model components and caches were released. Compilation was separately
exercised through a complete one-chunk engine run and returned finite
`[5, 3, 256, 320]` video and `[32, 16]` actions. Cold compilation is excluded
from the table and no speedup claim is made.

## Remaining experimental limits

- One GPU only; no FSDP or context-parallel claim.
- The engine returns one complete rollout step after destructive teardown and
  decode; it does not promise per-chunk interactive presentation.
- The VAE integration intentionally depends on Diffusers 0.38 private
  streaming state and must be retested before the dependency window is widened.
- The official checkpoint currently carries two obsolete patch-embedding keys;
  both upstream and native loaders ignore/drop the same entries.
