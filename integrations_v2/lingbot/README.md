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

## Install

```bash
uv sync --package flashdreams-lingbot --inexact
```

Checkpoints download from Hugging Face on first use. Export `HF_TOKEN` when the
selected repository requires authentication.

## Cam2V application

See [`apps/cam2v/README.md`](apps/cam2v/README.md) for the launch command. The
shared [`apps/cam2v`](../../apps/cam2v/README.md) documentation lists controls,
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
