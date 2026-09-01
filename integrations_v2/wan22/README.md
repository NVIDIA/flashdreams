<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `wan22`

Wan 2.2 TI2V-5B inference recipe and standalone application. The package
provides the pre-rolled `WanInferencePipelineConfig` literal, the diffusers
`state_dict` remap for the Wan-AI `Wan2.2-TI2V-5B-Diffusers` checkpoint, and
the `t2v-wan22-ti2v-5b` application entry point.

Wan 2.2 TI2V-5B is **bidirectional**: the application accepts a prompt and
first-frame image, then generates the complete 81-frame, 1280x640 clip in one
rollout instead of advancing through multiple causal blocks. It therefore
requires exactly one block (`--total-blocks 1`, which is the default);
multi-block generation is not supported. The application runs through the
shared null, MP4, local-window, or WebRTC host. HY-WorldPlay also reuses the
pipeline config as its base recipe.

## Application integrations

| application slug | pipeline config | description |
| --- | --- | --- |
| `t2v-wan22-ti2v-5b` | `wan22-ti2v-5b` | Bidirectional, single-block Wan 2.2 TI2V-5B at 1280x640, conditioned on a prompt and first frame. |

## Install

The package is registered as a `uv` workspace member, so a repo-root sync
installs it:

```bash
uv sync
```

For a targeted workspace environment:

```bash
uv sync --package flashdreams-wan22
```

Standalone editable installation also works:

```bash
uv pip install -e integrations_v2/wan22
```

## Hugging Face setup

Checkpoints are downloaded from Hugging Face on first use. Set an authentication
token if required by your environment:

```bash
export HF_TOKEN=<your-hf-token>
```

## Run

Generate the single-block MP4 with the v2 application. Because this is TI2V,
`--image-path` is required and must point to an existing image:

```bash
uv run --package flashdreams-wan22 flashdreams-run-v2 \
  t2v-wan22-ti2v-5b --output-path artifacts/t2v-wan22-ti2v-5b.mp4 -- \
  --prompt "A cinematic ocean wave at sunset." \
  --image-path /absolute/path/to/first-frame.png \
  --no-compile
```

See the [application README](apps/t2v/README.md) for the launch command and the
[shared T2V guide](../../apps/t2v/README.md) for common arguments.

## Public surface

- `PIPELINE_WAN22_TI2V_5B` — the assembled pipeline literal.
- `WAN22_TI2V_5B_DIT_DIFFUSERS_PATH` — diffusers sharded-safetensors index URL.
- `wan22_ti2v_5b_dit_state_dict_transform` — diffusers → flashdreams
  DiT key remap.
- `WAN_CONFIGS` — `{name: pipeline_config}` registry dict.

## Programmatic access

The pipeline config remains available directly for downstream integrations:

```python
from wan22.config import PIPELINE_WAN22_TI2V_5B
```

## Tests

```bash
uv run --group test --package flashdreams-wan22 \
  pytest integrations_v2/wan22/tests integrations_v2/wan22/tests
```
