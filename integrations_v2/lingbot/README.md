<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lingbot World

Lingbot World is a streaming camera-controlled image-to-video model integration.
Its public model variants are `StreamInferencePipelineConfig` literals in
`lingbot.config`; the reusable Cam2V application owns interactive I/O.

## Pipeline configurations

- `PIPELINE_LINGBOT_WORLD_FAST`
- `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3`
- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST`
- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3`
- `PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF`
- `PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF_TAEHV`

## Install

```bash
uv sync --package flashdreams-lingbot --inexact
```

Checkpoints download from Hugging Face on first use. Export `HF_TOKEN` when the
selected repository requires authentication.

## Cam2V application

```bash
uv sync --package flashdreams-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

Shared [`apps/cam2v`](../../apps/cam2v/README.md) documentation lists controls,
application arguments, and development commands.

## Programmatic pipeline access

```python
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST

pipeline = PIPELINE_LINGBOT_WORLD_FAST.setup().to("cuda").eval()
```

## Tests

```bash
uv run --no-sync pytest integrations_v2/lingbot -m ci_cpu
```

### Notes on performance options

These options are fields on `LingbotWorldDiTNetworkConfig`:

| Option | Values | Purpose |
| --- | --- | --- |
| `linear_backend` | `torch`, `rowwise_fp8` | Use standard Torch linears for the accuracy baseline or row-wise FP8 for faster large projections. |
| `self_attention_backend` | `wan`, `fp8_tma`, `scaled_fp8`, `sage` | Select Wan attention, FP8 TMA, scaled FP8, or in-tree INT8-QK/FP8-PV Sage attention. |
| `self_attention_use_tma` | `true`, `false` | Prefer the TMA attention kernel on supported GPUs; otherwise use the pointer-based kernel. |

`sage` does not require the external SageAttention package. Causal Sage caches
reuse an FP32 sum of finalized history keys within each chunk, reducing only
the live tail during denoising. Windows shorter than a chunk keep the full-cache
mean path. The FP16-PV TMA kernel folds the fixed probability scale into `exp2`
and carries the same scale in the softmax denominator, removing a per-tile
multiply.

These low-precision paths are not bit-exact to the previous implementation.
New reduction shapes require compilation warmup, which should be excluded from
steady-state benchmarks. The changes were validated on Linux with an RTX 5090;
Windows execution has not been validated.
