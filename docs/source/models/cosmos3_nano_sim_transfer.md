---
title: 'Cosmos3 Nano Sim Transfer'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer">Checkpoint and model card</a>
  <a class="fd-button" href="https://github.com/NVIDIA/cosmos-framework">Native reference implementation</a>
</div>

Cosmos3 Nano Sim Transfer generates 832 × 480 RGB video from a caption and a
control video. Each session uses **one** control kind: depth, segmentation,
edges, or blur. The model emits one opening frame, then four frames per
autoregressive step. Its history cache remains bounded during long sessions.

Use the recorded-clip application below or embed `TransferStream` in a host
that supplies new control frames. The default quality baseline is eager
inference with the released four-step schedule and Wan 2.2 decoding.

## Quick Start

Use Linux, Python 3.12, a CUDA-capable NVIDIA GPU, and `ffmpeg` on `PATH`.
The original integration was measured on an RTX PRO 6000 with about 37 GiB
peak allocated GPU memory for the eager 30/3 history profile. Allow additional
memory for the CUDA context and other processes. The BF16 model weights are
about 30.3 GB; the cache, VAE, environment, and outputs need additional disk
space.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[clone FlashDreams](../quickstart/index.md#install), then run from its root:

```bash
uv sync --frozen --inexact --python 3.12 \
  --package flashdreams-cosmos3-nano-sim-transfer --package flashdreams --group cuda13
uv run --no-sync flashdreams-run-v2 control2v-cosmos3-nano-sim-transfer \
  --output-path output.mp4 --timeout unbound -- \
  --control segmentation.mp4 --hint seg \
  --prompt "A city street seen from behind a car." \
  --seed 0 --history long-rollout --decoder wan2.2 --timing-path timing.json
```

Replace `segmentation.mp4` with your control clip. Weights and the tokenizer
are fetched from a pinned release revision on first use. `--release` selects
a local release snapshot; `--vae` supplies a local Wan 2.2 VAE checkpoint.
Inspect all application options without loading the model:

```bash
uv run --no-sync flashdreams-run-v2 control2v-cosmos3-nano-sim-transfer -- --help
```

The application takes the frame rate from the input file unless `--fps` is
set. Clips whose length is not `1 + 4n` are padded for encoding and the output
is trimmed to the requested length. `--max-frames` bounds a first run.

## Control input

Supply RGB-encoded control frames matching the checkpoint's training format.
Raw metric-depth arrays and segmentation class IDs must be converted by the
caller. The integration does not estimate depth or segmentation from an RGB
scene, and it does not combine multiple control modalities in one request.

| `--hint` | Resize to 832 × 480 |
| --- | --- |
| `seg` | Nearest neighbor, preserving palette values. |
| `depth` | Bilinear with antialiasing, preserving fractional resized values. |
| `edge` | Nearest neighbor. |
| `blur` | Bicubic with antialiasing, clipped and rounded to pixel values. |

## Streaming API

```python
from cosmos3_nano_sim_transfer import TransferStream

stream = TransferStream.load(hint="seg", decoder="wan2.2", accel="eager")
try:
    stream.start(prompt="A city street seen from behind a car.", seed=0)
    first_rgb = stream.step(first_control)  # uint8 [1, H, W, 3]
    next_rgb = stream.step(next_controls)   # uint8 [4, H, W, 3]
finally:
    stream.close()
```

`step` accepts NumPy or PyTorch control frames and returns uint8 RGB frames
at 832 × 480. `step_tensor` keeps output on the GPU in `[T, 3, 480, 832]`
layout. `start` begins a fresh session; `reset` restarts the current one and
clears the control encoder, history, and decoder state.

| Setting | Default | Options |
| --- | --- | --- |
| History | `long-rollout`: 30 latent frames, including 3 pinned sinks | `export`: 51/1, or explicit `W/S`. |
| Decoder | `wan2.2` | `taehv-super` is a faster approximation. |
| Acceleration | `eager` | `graphs` or `compile` through `TransferStream.load`. Call `warm_up(prompt)` before `start`. |
| Sampling | Four released denoising steps | `steps=` or `t_list=` through `TransferStream.load`; shorter schedules require separate quality validation. |

The recorded-clip application exposes history and decoder selection. Acceleration
and custom schedules are currently streaming API options. Graph and compiled
modes require CUDA and at most 512 prompt tokens. Eager mode accepts longer
prompts.

## Validation and limits

The [integration README](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/cosmos3_nano_sim_transfer/README.md#original-integration-measurements)
preserves the original author's parity and performance measurements, with their
hardware, dependency versions, and limitations. They are historical results,
not benchmarks newly reproduced for the current checkout. The package provides
CPU contract tests and a manual GPU probe against native cosmos-framework
outputs so a deployment can validate its own checkpoint and environment.

The documented native parity profile disables `emphasize_control_in_prompt`;
enabling that upstream option is not implemented here. Long free rollouts can
drift numerically, and the original 601-frame comparison did not meet the
cross-GPU reference spread against both references. Keep the four-step eager
profile as the baseline when assessing acceleration, an approximate decoder,
shorter history, or a fine-tuned checkpoint. Real-time throughput depends on
the hardware and profile; the original four-step measurements were below
30 generated frames per second.

## Developer Details

- [Integration source, streaming options, and parity probe](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/cosmos3_nano_sim_transfer)
- [Shared Cosmos3 runtime](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/cosmos_interactive)
- [FlashDreams profiling](../documentation/demo_api/guides/profiling.md)

The package depends on the shared Cosmos3 runtime. `config.py` owns exported
model settings, `impl/` contains model implementation, and
`apps/control2v/adapter.py` supplies the v2 application. Model and adapter
tests live in the integration's `tests/` directory. Host-specific simulation,
rendering, transport, and fine-tuned checkpoint presets belong in host adapters.
