<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `wan22`

Wan 2.2 TI2V-5B inference recipe and standalone application. The package
provides the pre-rolled `WanInferencePipelineConfig` literal, the diffusers
`state_dict` remap for the Wan-AI `Wan2.2-TI2V-5B-Diffusers` checkpoint, and
the `ti2v-wan22` application entry point.

The application accepts a prompt and first-frame image, then generates the
model's standard single-block 81-frame, 640x1280 rollout through the shared
null, MP4, local-window, or WebRTC application host. HY-WorldPlay also reuses
the pipeline config as its base recipe.

## Application

MP4 output:

```bash
uv run --package flashdreams-wan22 flashdreams-run ti2v-wan22 \
  --output mp4 --output-path artifacts/ti2v-wan22.mp4 \
  --prompt "A cinematic ocean wave at sunset." \
  --image-path /path/to/first-frame.png
```

Null output:

```bash
uv run --package flashdreams-wan22 flashdreams-run ti2v-wan22 \
  --output null \
  --prompt "A cinematic ocean wave at sunset." \
  --image-path /path/to/first-frame.png
```

WebRTC output:

```bash
uv run --package flashdreams-wan22 flashdreams-run ti2v-wan22 \
  --output webrtc --host 0.0.0.0 --port 8080 \
  --prompt "A cinematic ocean wave at sunset." \
  --image-path /path/to/first-frame.png
```

Then open `http://localhost:8080/request_session`.

## Public surface

- `PIPELINE_WAN22_TI2V_5B` — the assembled pipeline literal.
- `WAN22_TI2V_5B_DIT_DIFFUSERS_PATH` — diffusers sharded-safetensors index URL.
- `wan22_ti2v_5b_dit_state_dict_transform` — diffusers → flashdreams
  DiT key remap.
- `WAN_CONFIGS` — `{name: pipeline_config}` registry dict.

## Install

Workspace member; pulled in by repo-root `uv sync`.

```python
from wan22.config import PIPELINE_WAN22_TI2V_5B
```
