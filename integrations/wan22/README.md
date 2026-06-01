<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `wan22`

Wan 2.2 TI2V-5B inference recipe config — the pre-rolled
`WanInferencePipelineConfig` literal and the diffusers `state_dict`
remap for the Wan-AI `Wan2.2-TI2V-5B-Diffusers` checkpoint.

Config-only package; downstream runners
(`integrations/hy_worldplay`, `flashdreams-run` slugs) layer the I/O
wrapper on top.

## Public surface

- `PIPELINE_WAN22_TI2V_5B` — the assembled pipeline literal.
- `WAN22_TI2V_5B_DIT_DIFFUSERS_PATH` — diffusers safetensors URL.
- `wan22_ti2v_5b_dit_state_dict_transform` — diffusers → flashdreams
  DiT key remap.
- `WAN_CONFIGS` — `{name: pipeline_config}` registry dict.

## Install

Workspace member; pulled in by repo-root `uv sync`.

```python
from wan22.config import PIPELINE_WAN22_TI2V_5B
```
