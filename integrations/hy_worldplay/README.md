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

The slug is exposed as a `flashdreams-run hy-worldplay-wan-i2v-5b`
subcommand via the standard `flashdreams.runner_configs` entry-point
group, just like `self_forcing` / `wan21`. Because the upstream WAN
pipeline does not slice cleanly into flashdreams'
`StreamInferencePipeline` 3-stage encode/diffuse/decode interface
(action + memory + chunked AR + distributed VAE), the runner sets
`pipeline=None` on its `RunnerConfig` and owns its own `__init__`
(the base `Runner` skips pipeline construction in that case).
Promotion onto a real `WanInferencePipeline` is phase 2b (see
"Staging plan" below).

## Install

The plugin ships in **two layers** so HY-WorldPlay's heavy upstream
deps (sageattention, cloudpickle, accelerate, ...) don't leak into the
repo-root `uv.lock`:

1. **Lightweight workspace member** — registered in the repo-root
   `pyproject.toml`, picked up by a normal `uv sync`. Gives you the
   `hy_worldplay` import path, the runner config surface, and the
   CPU-only smoke tests. No upstream deps; no GPU; works in the main
   flashdreams venv.
2. **Isolated run / parity sub-venv** under
   [`tests/parity_check/`](tests/parity_check/) — pins
   `sageattention`, `accelerate`, `cloudpickle`, `torch==2.11.*`,
   etc. Used both for the upstream parity baseline *and* for actually
   invoking `flashdreams-run hy-worldplay-wan-i2v-5b` end-to-end on a
   GPU. This
   mirrors the [`self_forcing/tests/parity_check`](../self_forcing/tests/parity_check)
   layout and keeps HY-WorldPlay's heavy stack scoped to the
   integration directory.

Day-to-day setup:

```bash
# layer 1: lightweight workspace install (from repo root)
uv sync

# layer 2: heavy run/parity sub-venv (from the parity-check dir)
( cd integrations/hy_worldplay/tests/parity_check && uv sync )
```

Once both have run, `flashdreams-run hy-worldplay-wan-i2v-5b` works
from the parity-check sub-venv via `uv run --project ...` (see below).

The upstream HY-WorldPlay tree is **not** a Python dependency; you
provision it once and point the runner at it. The easiest way is to
let the parity-check script clone it for you:

```bash
bash integrations/hy_worldplay/tests/parity_check/run.sh
# clones to integrations/hy_worldplay/tests/parity_check/HY-WorldPlay
# and syncs the sub-venv as a side effect
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
# NOTE: positional args after the repo id are treated as *exact filenames*,
# not directory prefixes, so use ``--include`` glob patterns for whole
# subdirectories (otherwise huggingface-cli silently fetches zero files).
huggingface-cli download tencent/HY-WorldPlay \
    --include "wan_transformer/*" "wan_distilled_model/*" \
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

All GPU invocations go through the parity-check sub-venv (see "Install"
above). Use `uv run --project <path>` so uv picks that venv instead of
the main flashdreams one — the heavy deps (sageattention, accelerate,
cloudpickle, ...) only live there.

Single-GPU (matches upstream's
[`wan/README.md`](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/wan/README.md)
1-GPU example):

```bash
PARITY=integrations/hy_worldplay/tests/parity_check

uv run --project "${PARITY}" flashdreams-run hy-worldplay-wan-i2v-5b \
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
uv run --project "${PARITY}" torchrun \
    --nproc_per_node=4 --no-python flashdreams-run hy-worldplay-wan-i2v-5b \
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
uv run --project "${PARITY}" flashdreams-run hy-worldplay-wan-i2v-5b --help
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

The integration is staged across multiple PRs. Phase 1 ships in this
PR; phase 2 has a hard prerequisite in core flashdreams that has to
land first.

1. **Phase 1 — this PR.** Vendor wrapper + parity check.
   - Plugin packaging (`pyproject.toml`, `uv` workspace member);
     heavy upstream deps scoped to the parity sub-venv so they don't
     leak into the repo-root `uv.lock`.
   - Thin `HyWorldPlayWanI2VRunner` shim that calls upstream's
     `WanRunner.predict()` so we get bit-identical output to
     `torchrun wan/generate.py` with the same flags.
   - Registered with `flashdreams-run` via the
     `flashdreams.runner_configs` entry-point group; invoked from the
     parity sub-venv (which has the heavy deps + an editable
     `flashdreams` install so the console script resolves).
   - Parity-check infra under `tests/parity_check/` that clones
     upstream at a pinned commit, downloads checkpoints, and runs the
     reference benchmark. Numeric per-frame parity bar enforced
     (see [`tests/parity_check/README.md`](tests/parity_check/README.md)).

2. **Phase 2a (prerequisite, lives in `flashdreams/recipes/wan/`).**
   Add a **Wan 2.2 5B** recipe to core flashdreams. Today the recipe
   family only covers Wan 2.1 (1.3B / 14B variants used by
   `self_forcing` / `causal_forcing` / `wan21`) and Wan 2.2 14B; the
   5B variant — which HY-WorldPlay's WAN backbone is built on — is
   not implemented yet. Without it, phase 2b has nothing to layer
   onto. This work does not depend on HY-WorldPlay and is a useful
   addition in its own right.

3. **Phase 2b (this directory, follow-up to 2a).** Recipe-level
   integration on top of the new flashdreams Wan 2.2 5B recipe.
   - Replace the vendor-wrapper runner with the more optimized
     flashdreams Wan 2.2 5B `WanInferencePipeline` so HY-WorldPlay
     shares the KV cache, context-parallelism, profiler, and attention
     dispatch with the rest of the `wan*` family.
   - Extend that pipeline with HY-WorldPlay's deltas: action inputs,
     camera-trajectory conditioning, and the reconstituted-context
     memory module.
   - Drop the parity sub-venv: the run path collapses back into the
     main flashdreams venv once the heavy upstream deps (sageattention,
     cloudpickle, accelerate, ...) are no longer needed. The
     `flashdreams-run hy-worldplay-wan-i2v-5b` slug already shipped in
     phase 1 stays as the stable user-facing interface.
   - Re-run the parity check against the phase-1 baseline so the
     refactor is gated on a numeric diff, not eyeballs.

4. **Phase 3 — future.** HunyuanVideo-1.5 8B variant
   (`hyvideo/generate.py` upstream). Heavier integration: multiple
   text encoders (Qwen2.5-VL-7B, ByT5, Glyph-SDXL-v2), gated vision
   encoder (FLUX.1-Redux-dev), 8-way SP, distilled / RL-tuned model
   variants.
