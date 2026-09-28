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
- `PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_MAX_PERF`
- `PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_MAX_PERF_TAEHV`

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
| `self_attention_backend` | `wan`, `fp8_tma`, `scaled_fp8` | Use WAN attention for the accuracy baseline, direct FP8 for maximum speed, or dynamically scaled FP8 to preserve more range. |
| `self_attention_use_tma` | `true`, `false` | Prefer the TMA attention kernel on supported GPUs; otherwise use the pointer-based kernel. |

The max-performance presets select `rowwise_fp8`, `fp8_tma`, and TMA. Direct
FP8 is faster than scaled FP8 but gives up the latter's per-tensor scale
compensation.
