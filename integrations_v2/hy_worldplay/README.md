<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay

HY-WorldPlay is a streaming camera-controlled image-to-video model with action,
camera-trajectory, and reconstituted-context memory conditioning. Its public
model configuration is the `HyWorldPlayPipelineConfig` literal exported from
`hy_worldplay.config`.

## Pipeline configuration

- `PIPELINE_HY_WORLDPLAY_WAN_I2V_5B`

The pipeline uses the HY-WorldPlay distilled checkpoint by default and keeps
memory-selection settings on the pipeline config.

## Install

```bash
uv sync --package flashdreams-hy-worldplay --inexact
```

Export `HF_TOKEN` when checkpoint access requires authentication.

## Cam2V application

See [`apps/cam2v/README.md`](apps/cam2v/README.md) for the launch command. The
shared [`apps/cam2v`](../../apps/cam2v/README.md) documentation lists controls,
application arguments, and development commands.

## Programmatic pipeline access

```python
from hy_worldplay.config import PIPELINE_HY_WORLDPLAY_WAN_I2V_5B

pipeline = PIPELINE_HY_WORLDPLAY_WAN_I2V_5B.setup().to("cuda").eval()
```

## Tests

```bash
uv run --no-sync pytest integrations_v2/hy_worldplay -m ci_cpu
```
