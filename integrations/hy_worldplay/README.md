<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `hy_worldplay`

Integration of [HY-World 1.5 / WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
into `flashdreams`. WorldPlay is Tencent Hunyuan's real-time
interactive world model — a streaming video diffusion model with
action + camera-trajectory conditioning and reconstituted-context
memory.

This is the **standalone "mini-repo" plugin**, packaged as a `uv`
workspace member, following the
[`integrations/self_forcing`](../self_forcing/README.md) pattern.

## What ships in this PR (phase 1)

| slug | description |
| --- | --- |
| `hy-worldplay-wan-i2v-5b` | HY-WorldPlay WAN-5B I2V (Wan 2.2 TI2V backbone, action + camera trajectory conditioning, reconstituted-context memory). Distilled checkpoint, 4 inference steps. |

This first PR is intentionally a **vendor-wrapper**: the runner
delegates pipeline construction and inference to upstream's
`wan/generate.py` `WanRunner` directly, so output is bit-for-bit
identical to a vanilla `torchrun wan/generate.py ...` invocation. The
parity check at `tests/parity_check/` verifies that baseline.

The upstream WAN pipeline does not slice cleanly into flashdreams'
`StreamInferencePipeline` 3-stage encode/diffuse/decode interface
(action + memory + chunked AR + distributed VAE), so the slug is
exposed via a standalone CLI (`python -m hy_worldplay.cli`) rather
than as a `flashdreams-run` subcommand for now. Promotion to
`flashdreams-run` is part of phase 2 (see "Staging plan" below).

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/hy_worldplay
```

The upstream HY-WorldPlay tree is **not** a Python dependency; you
provision it once and point the runner at it. The easiest way is to
let the parity-check script clone it for you:

```bash
bash integrations/hy_worldplay/tests/parity_check/run.sh
# clones to integrations/hy_worldplay/tests/parity_check/HY-WorldPlay
```

…and then pass that path via `--hy-worldplay-repo-root`. Or clone
manually:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
```

## HuggingFace setup

Both the base Wan 2.2 backbone and HY-WorldPlay's WAN-5B distilled
weights are auto-downloadable from HuggingFace; set an auth token
first.

```bash
export HF_TOKEN=<your-hf-token>
export HF_HOME=~/.cache/huggingface  # default
```

The HY-WorldPlay WAN models are bundled in the
[`tencent/HY-WorldPlay`](https://huggingface.co/tencent/HY-WorldPlay)
repo:

```bash
huggingface-cli download tencent/HY-WorldPlay \
    wan_transformer wan_distilled_model \
    --local-dir /path/to/models
```

That gives you:

```
/path/to/models/
├── wan_transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── wan_distilled_model/
    └── model.pt
```

## Run

Single-GPU (matches upstream's
[`wan/README.md`](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/wan/README.md)
1-GPU example):

```bash
uv run python -m hy_worldplay.cli \
    --image-path ./assets/img/test.png \
    --ar-model-path /path/to/models/wan_transformer \
    --ckpt-path /path/to/models/wan_distilled_model/model.pt \
    --hy-worldplay-repo-root /path/to/HY-WorldPlay \
    --num-chunk 1 \
    --pose "w-4" \
    --output-dir outputs
```

Multi-GPU (4 GPUs, matches upstream's 4-GPU example):

```bash
uv run torchrun --nproc_per_node=4 --no-python --module hy_worldplay.cli \
    --image-path ./assets/img/test.png \
    --ar-model-path /path/to/models/wan_transformer \
    --ckpt-path /path/to/models/wan_distilled_model/model.pt \
    --hy-worldplay-repo-root /path/to/HY-WorldPlay \
    --num-chunk 4 \
    --pose "w-16" \
    --output-dir outputs
```

Per-runner `--help` lists every overridable field:

```bash
uv run python -m hy_worldplay.cli --help
```

### Camera control

Same pose-string grammar as upstream:

| token | action | example |
| --- | --- | --- |
| `w-N` / `s-N` | forward / backward, N latents | `w-16` |
| `a-N` / `d-N` | strafe left / right, N latents | `d-4` |
| `up-N` / `down-N` | pitch up / down, N latents | `up-2` |
| `left-N` / `right-N` | yaw left / right, N latents | `right-1` |

Multiple actions are comma-separated. The total latent count must
equal `--num-chunk * 4`. Or pass a JSON file produced by upstream's
`hyvideo/generate_custom_trajectory.py` to `--pose`.

## Programmatic access

```python
from pathlib import Path

from hy_worldplay.config import RUNNER_HY_WORLDPLAY_WAN_I2V_5B
from dataclasses import replace

cfg = replace(
    RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
    image_path=Path("./assets/img/test.png"),
    ar_model_path=Path("/path/to/models/wan_transformer"),
    ckpt_path=Path("/path/to/models/wan_distilled_model/model.pt"),
    hy_worldplay_repo_root=Path("/path/to/HY-WorldPlay"),
    num_chunk=1,
    pose="w-4",
)
runner = cfg.setup()
runner.run()
```

## Tests

CPU-only smoke tests (no GPU, no upstream tree required):

```bash
uv run --extra dev pytest integrations/hy_worldplay/tests/test_smoke.py
```

End-to-end parity benchmark against upstream (requires GPU, downloads
checkpoints on first run):

```bash
bash integrations/hy_worldplay/tests/parity_check/run.sh
```

See [`tests/parity_check/README.md`](tests/parity_check/README.md)
for what the parity script does and where it writes outputs.

## Staging plan

This integration is planned to roll out across two PRs:

1. **Phase 1 (this PR).** Vendor wrapper + parity check.
   - Plugin packaging (`pyproject.toml`, `uv` workspace member).
   - Thin `HyWorldPlayWanI2VRunner` shim that calls upstream's
     `WanRunner.predict()` so we get bit-identical output to
     `torchrun wan/generate.py` with the same flags.
   - Standalone `python -m hy_worldplay.cli` entry point.
   - Parity-check infra under `tests/parity_check/` that clones
     upstream at a pinned commit, downloads checkpoints, and runs the
     reference benchmark with `EventProfiler`-style per-chunk timings.
2. **Phase 2 (follow-up).** Recipe-level integration.
   - Wire HY-WorldPlay's WAN backbone onto
     `flashdreams.recipes.wan.WanInferencePipeline` so it shares the
     KV cache, CP, and profiler with `self_forcing` /
     `causal_forcing` / `wan21`.
   - Extend the recipe with action + camera-trajectory inputs and the
     reconstituted-context memory module.
   - Promote the slug to a `flashdreams-run hy-worldplay-wan-i2v-5b`
     subcommand (entry-point group `flashdreams.runner_configs`),
     dropping the standalone CLI.
3. **Phase 3 (future).** HunyuanVideo-1.5 8B variant
   (`hyvideo/generate.py` upstream). Heavier integration: multiple
   text encoders (Qwen2.5-VL-7B, ByT5, Glyph-SDXL-v2), gated vision
   encoder (FLUX.1-Redux-dev), 8-way SP, distilled / RL-tuned model
   variants.
