<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SANA-WM

SANA-WM provides bidirectional and streaming camera-controlled image-to-video
models. Its public model variants are `StreamInferencePipelineConfig` literals
in `sana_wm.config`; the reusable Cam2V application owns interactive I/O.

## Pipeline configurations

- `PIPELINE_SANA_WM_BIDIRECTIONAL`
- `PIPELINE_SANA_WM_STREAMING`

The streaming configuration uses the distilled Stage-1 schedule and chunk-causal
LTX-2 refiner/VAE path. Both configurations load the public SANA-WM checkpoints
directly and support their model-specific BF16, FP8, and FP4 paths.

## Install

```bash
uv sync --package flashdreams-sana-wm --inexact
```

Checkpoints download from Hugging Face on first use.

## Cam2V application

```bash
uv sync --package flashdreams-sana-wm --inexact
uv run --no-sync flashdreams-run-v2 cam2v-sana-wm-streaming \
  --mode webrtc --host 0.0.0.0 --port 8089 -- \
  --example-data
```

The adapter passes live controls through the SANA-WM control remapper and
conditioning function. It appends each generated 24-frame block to camera
history while keeping the encoded prompt and first frame cached.

`--example-data` downloads the official `demo_0.png` and paired prompt into the
FlashDreams example-data cache. Explicit image and prompt arguments override it.

Shared [`apps/cam2v`](../../apps/cam2v/README.md) documentation lists controls,
application arguments, output modes, and development commands.

## Programmatic pipeline access

```python
from sana_wm.config import PIPELINE_SANA_WM_STREAMING

pipeline = PIPELINE_SANA_WM_STREAMING.setup().to("cuda").eval()
```

## Tests

```bash
uv run --no-sync pytest integrations_v2/sana_wm -m ci_cpu
```
