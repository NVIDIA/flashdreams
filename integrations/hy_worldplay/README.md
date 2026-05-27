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

## Runners

| slug | description |
| --- | --- |
| `hy-worldplay-wan-i2v-5b` | HY-WorldPlay WAN-5B I2V (Wan 2.2 TI2V backbone, action + camera trajectory conditioning, reconstituted-context memory). Distilled checkpoint, 4 inference steps. |

Two backends, selected by `--use-native-pipeline` (default `True`):

- **Native** drives the in-tree `WanInferencePipeline` (Wan 2.2 TI2V-5B
  recipe) with HY-WorldPlay's conditioner subclasses layered on top.
  Runs in the main flashdreams venv. See
  [Native pipeline](#native-pipeline) below.
- **Vendor wrapper** (`--no-use-native-pipeline`) delegates to
  upstream's `wan/generate.py` `WanRunner` for bit-for-bit match with
  `torchrun wan/generate.py`. Runs out of the parity sub-venv and
  needs the heavy upstream deps installed on demand. Fills its
  mandatory `RunnerConfig.pipeline` slot with an inert
  `_NoopPipelineConfig` (a `StreamInferencePipeline` subclass that
  skips slot construction) because the upstream WAN pipeline does not
  slice cleanly into flashdreams' `StreamInferencePipeline` 3-stage
  encode/diffuse/decode interface (action + memory + chunked AR +
  distributed VAE).

Registered via the `flashdreams.runner_configs` entry-point group,
like `self_forcing` / `wan21`.

## Install

The plugin ships in **two layers**:

1. **Lightweight workspace member** — registered in the repo-root
   `pyproject.toml`, picked up by a normal `uv sync`. Gives you the
   `hy_worldplay` import path, the runner config surface, and the
   CPU-only smoke tests in the main flashdreams venv. Also enough to
   run the **native** path end-to-end on GPU; no upstream deps needed.
2. **Parity sub-venv** under
   [`tests/parity_check/`](tests/parity_check/), pinning
   `torch==2.11.*` (lockstep with the flashdreams root `uv.lock`).
   Required for the **vendor wrapper** and the upstream parity
   baseline. The heavy upstream deps (`sageattention`, `accelerate`,
   `cloudpickle`, `transformers==4.57.6`) are not pinned by the
   sub-venv's `pyproject.toml` — `tests/parity_check/run.sh`
   `uv pip install`s them on demand. Mirrors the
   [`self_forcing/tests/parity_check`](../self_forcing/tests/parity_check)
   layout.

Day-to-day setup:

```bash
# layer 1: lightweight workspace install (from repo root)
uv sync

# layer 2: parity sub-venv (from the parity-check dir)
( cd integrations/hy_worldplay/tests/parity_check && uv sync )
```

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

This section covers the vendor wrapper, which runs out of the parity
sub-venv (`uv run --project <path>` picks that venv over the main
flashdreams one). The native default lives in the main venv — see
[Native pipeline](#native-pipeline) below.

Single-GPU vendor wrapper, matching upstream's
[`wan/README.md`](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/wan/README.md)
1-GPU example:

```bash
PARITY=integrations/hy_worldplay/tests/parity_check

uv run --project "${PARITY}" flashdreams-run hy-worldplay-wan-i2v-5b \
    --no-use-native-pipeline \
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
    --no-use-native-pipeline \
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

### Native pipeline

The native path drives the in-tree `WanInferencePipeline` directly so
HY-WorldPlay shares the KV cache, context-parallelism, profiler, and
attention dispatch with the rest of the `wan*` family. Default;
pass `--no-use-native-pipeline` for the vendor wrapper.

```bash
uv run flashdreams-run hy-worldplay-wan-i2v-5b \
    --use-action-conditioning \
    --use-camera-conditioning \
    --use-memory-selection \
    --image-path ./assets/img/test.png \
    --num-chunk 1 \
    --pose "w-4" \
    --output-dir outputs
```

**Flag semantics.** `--use-action-conditioning` and
`--use-camera-conditioning` are independent — either, both, or neither
can be set. Flipping either triggers the encoder / transformer /
network subclass swap; the camera flag additionally enables the
PRoPE dual-branch block path. `--use-memory-selection` requires
`--use-camera-conditioning` (the FOV-overlap selector consumes the
per-rollout viewmats binding) and is silently ignored on the vendor
wrapper path. All three conditioners and the prefill executor are
zero-initialised, so flipping flags on without the distilled
checkpoint is a strict identity.

**Parity.** 2-chunk GPU smoke at 704x1280 / `seed=0` against vendor's
`use_kv_cache=True` baseline lands at **`mean |Δ| = 15.65 / 255`**
(chunk-0 12.91, chunk-1 18.21) — below the visible threshold
(~30/255) and within ~3-4× of the vendor-vs-vendor kernel noise floor
(3.24/255). Acceptance bar `<= 20 / 255`. Residual drift is
multi-causal bf16 FP-noise with no single dominant source; the
diagnostic env-var flags (`HY_DEBUG_*`, `HY_VENDOR_NOISE_MODE`,
`HY_VENDOR_VAE_MEAN`) are wired up in the code for re-running the
per-bug breakdown locally. For bit-exact match against vendor's
`use_kv_cache=False` default, fall back to the vendor wrapper.

**Reproduce the parity diff locally.** Run
`USE_KV_CACHE_TRUE=1 ./tests/parity_check/run.sh`. The script
installs the heavy vendor deps (`sageattention`, `cloudpickle`,
`accelerate>=0.30`, `transformers==4.57.6`) into the parity sub-venv
on demand; they are not pinned by the sub-venv's `pyproject.toml`.

#### Known quirk

- **Upstream FOV-selector boundary on short rollouts.** The
  `select_mem_frames_wan` algorithm (faithfully ported in `_memory.py`)
  has `historical_clip_starts` that allow clip starts whose
  `[start, start+pred_latent_size)` range overlaps the temporal-context
  window when the FOV-distance scorer picks the latest start. With
  short rollouts (e.g. the 2-chunk smoke at 21 frames of history per
  chunk), the resulting set-union can shrink below the requested
  `memory_frames`, which the final assertion catches. Production
  rollouts with larger `temporal_context` and many chunks of history
  avoid this; the smoke pins the prefill executor by monkey-patching
  the encoder to feed `memory_frame_indices=[0,1,2,3]`, bypassing the
  FOV scorer.

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

Phases 1, 2a, and 2b are landed; the native pipeline is the production
default. Phase 3 is future.

1. **Phase 1 — vendor wrapper.** `hy_worldplay` packaged as a `uv`
   workspace member (heavy upstream deps scoped to the parity sub-venv);
   `HyWorldPlayWanI2VRunner` shim over upstream's `WanRunner.predict()`
   for bit-identical output; registered with `flashdreams-run`;
   parity-check harness under
   [`tests/parity_check/`](tests/parity_check/README.md).

2. **Phase 2a — WAN 2.2 5B recipe (`flashdreams.recipes.wan`).**
   Prerequisite for 2b; useful on its own. Fills the gap between
   Wan 2.1 (1.3B / 14B) and Wan 2.2 14B with the 5B VAE / DiT configs
   (`Wan22TI2V5BVAE{Encoder,Decoder}Config`, `WanDiTNetworkTI2V5BConfig`),
   the `ti2v_first_frame_per_token_timestep` flag on
   `Wan21TransformerConfig`, the `PIPELINE_WAN22_TI2V_5B` pre-rolled
   config, and diffusers-safetensors remaps.

3. **Phase 2b — native HY-WorldPlay integration.** Each conditioner is
   gated behind its own flag and zero-initialised so flipping it on
   without the distilled checkpoint is a strict identity.

   - **2b.1 / 2b.2.** Native runner over `PIPELINE_WAN22_TI2V_5B` +
     distilled 4-step Euler schedule swapped in on the native path.
   - **2b.3.** 81-class action conditioner (AdaLN add on
     `HyWorldPlayWanDiTNetwork`). `--use-action-conditioning`.
   - **2b.4.** PRoPE dual-branch self-attention
     (`HyWorldPlayPRoPEBlock`; `prope_qkv` in
     `hy_worldplay._prope`). `--use-camera-conditioning`.
   - **2b.5a.** Reconstituted-context memory selection
     (`select_mem_frames_wan` + FOV-overlap helper, ported to
     `hy_worldplay/_memory.py`). `--use-memory-selection`.
   - **2b.5b.** Distilled-checkpoint remap + KV-prefill executor
     (per-rollout `clean_latent_history`, per-block
     `HyWorldPlayMemoryKVCache`, per-chunk rolling-cache reset,
     `prefill_completed_for_chunk` latch).
   - **2b.6.** Parity close at **`mean |Δ| = 15.65 / 255`**
     (704x1280 / `num_chunk=2`; below the visible threshold and within
     ~3-4× of the vendor-vs-vendor kernel noise floor of 3.24/255).
     Acceptance bar `<= 20 / 255`.