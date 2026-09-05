<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LongSANA

LongSANA 2B is a causal, text-to-video SANA-Video integration for FlashDreams
Runtime V2. It uses the public self-forcing 480p checkpoint and keeps a
constant-size recurrent attention state instead of retaining every prior token.

The implementation reuses the existing SANA-WM Gemma/CHI prompt encoder,
normalization, timestep/text projection, and cross-attention components, plus
the shared Wan 2.1 VAE decoder and Runtime V2 diffusion/session machinery.

## Model

| application slug | resolution | rate | default rollout |
| --- | ---: | ---: | ---: |
| `t2v-longsana-2b-480p` | 832 x 480 | 16 FPS | 1,041 frames (~65 s) |

The first autoregressive block emits 41 pixel frames. Every later block emits
40. The released sampler uses four self-forcing steps at raw timesteps
`1000, 960, 889, 727`, flow shift 7, CFG 1, and motion score 10.

## Install and run

```bash
uv sync --package flashdreams-longsana --extra dev --group test --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-longsana-2b-480p \
  --timeout 30 \
  --output-path artifacts/longsana.mp4 -- \
  --prompt "A red panda walks through a misty bamboo forest at sunrise." \
  --total-blocks 2 --seed 0 --no-compile
```

The generator, Gemma text encoder, and Wan VAE download from Hugging Face on
first use. The application accepts only the checkpoint's native 832 x 480
resolution.

`--timeout` bounds the shared interactive T2V presentation session. Use the
benchmark command below for exact generated-frame clips; a timeout-bounded
Runtime V2 MP4 may repeat its latest frame while the UI remains active.

## Constant-memory cache

Each of the 20 transformer blocks owns three recurrent tensors:

- a cumulative rotated `V @ K^T` matrix;
- a cumulative positive-key sum used by the linear-attention denominator;
- the final frame needed by the causal temporal convolution.

The state is updated in-place after a clean-timestep forward pass. Its size is
independent of generated duration (152.53 MiB for batch size one at the native
configuration), and absolute temporal RoPE positions advance across blocks.

## Benchmark and validation

See [VALIDATION.md](VALIDATION.md) for measured results and qualitative review.

Run the checked-in diverse-prompt suite:

```bash
uv run --no-sync python integrations_v2/longsana/scripts/benchmark.py \
  --output-dir artifacts/longsana_benchmark --blocks 2
```

Use `--long-blocks 6` to extend the temporal-continuity case while keeping the
other cases short. The command writes MP4 clips, contact sheets, per-block
Runtime V2 stage timings, GPU allocation/peak measurements, cache-size history,
and a machine-readable `summary.json`.

Capture a PyTorch operator trace for one warmed steady-state block with:

```bash
uv run --no-sync python integrations_v2/longsana/scripts/operator_profile.py \
  --output-dir artifacts/longsana_profile
```

The first block includes lazy generator loading and is reported separately.
Steady-state throughput excludes prompt encoding but includes diffusion, Wan
decode, and the cache-finalization pass. Compare identical resolution, frame
rate, block count, seed, and prompt set when benchmarking another DiT.

## Tests

```bash
uv run --no-sync pytest integrations_v2/longsana/tests -m ci_cpu
uv run --no-sync ruff check integrations_v2/longsana \
  flashdreams/flashdreams/recipes/wan/autoencoder/vae.py
```
