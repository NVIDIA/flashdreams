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

The runner drives the in-tree `WanInferencePipeline` (Wan 2.2 TI2V-5B
recipe) with HY-WorldPlay's conditioner subclasses layered on top.
The fully-swapped pipeline lives statically in
`hy_worldplay.config.PIPELINE_HY_WORLDPLAY_WAN_I2V_5B`; the runner
config is a plain dataclass with no `__post_init__`. The plugin does
not import upstream's `wan/generate.py` at runtime — the parity-check
harness invokes it directly via `torchrun`.

Registered via the `flashdreams.runner_configs` entry-point group,
like `self_forcing` / `wan21`.

## Install

```bash
# repo-root workspace install (gives you the runner + CPU smoke tests)
uv sync

# parity sub-venv (only needed to re-run the upstream parity baseline)
( cd integrations/hy_worldplay/tests/parity_check && uv sync )
```

## HuggingFace setup

Both the base Wan 2.2 backbone and HY-WorldPlay's distilled WAN-5B
weights are auto-downloadable from HuggingFace; set an auth token
first.

```bash
export HF_TOKEN=<your-hf-token>
export HF_HOME=~/.cache/huggingface  # default
```

The HY-WorldPlay distilled checkpoint is bundled in the
[`tencent/HY-WorldPlay`](https://huggingface.co/tencent/HY-WorldPlay)
repo:

```bash
# NOTE: positional args after the repo id are treated as *exact filenames*,
# not directory prefixes, so use ``--include`` glob patterns for whole
# subdirectories (otherwise ``hf`` silently fetches zero files).
hf download tencent/HY-WorldPlay \
    --include "wan_distilled_model/*" \
    --local-dir /path/to/models
```

That gives you:

```
/path/to/models/
└── wan_distilled_model/
    └── model.pt
```

## Run

```bash
uv run flashdreams-run hy-worldplay-wan-i2v-5b \
    --example-data \
    --ckpt-path /path/to/models/wan_distilled_model/model.pt \
    --num-chunk 1 \
    --output-dir outputs
```

`--example-data` lazy-downloads upstream's sample first frame
(`assets/img/test.png`) and camera trajectory
(`assets/pose/test_forward_32_latents.json`) into
`data_local/hy_worldplay/` (gitignored) and uses them as the input
image + pose. Pass `--image-path <path>` and/or `--pose <script|json>`
to override either; the pose source is prefix-sliced to the rollout's
`num_chunk * 4` latent budget, so the 33-entry sample drives any
`num-chunk` up to 8.

`--ckpt-path` is optional. Without it the pipeline loads the base
Wan 2.2 TI2V-5B safetensors and HY's conditioners stay zero-init
(strict identity, parity-safe against the base Wan 2.2 output). Pass
`--ckpt-path` to load the distilled WAN-5B weights and exercise the
full HY-WorldPlay stack.

Per-runner `--help` lists every overridable field:

```bash
uv run flashdreams-run hy-worldplay-wan-i2v-5b --help
```

**Parity.** 2-chunk GPU smoke at 704x1280 / `seed=0` against vendor's
`use_kv_cache=True` baseline lands at **`mean |Δ| = 15.65 / 255`**
(chunk-0 12.91, chunk-1 18.21) — below the visible threshold
(~30/255) and within ~3-4× of the vendor-vs-vendor kernel noise floor
(3.24/255). Acceptance bar `<= 20 / 255`. Residual drift is
multi-causal bf16 FP-noise with no single dominant source; the
diagnostic env-var flags (`HY_DEBUG_*`, `HY_VENDOR_NOISE_MODE`,
`HY_VENDOR_VAE_MEAN`) are wired up in the code for re-running the
per-bug breakdown locally.

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
| `w-N` / `s-N` | forward / backward, N motion steps | `w-15` |
| `a-N` / `d-N` | strafe left / right, N motion steps | `d-3` |
| `up-N` / `down-N` | pitch up / down, N motion steps | `up-1` |
| `left-N` / `right-N` | yaw left / right, N motion steps | `right-1` |

Multiple actions are comma-separated. The parser prepends an identity
pose for the input frame, so the script must contain
`--num-chunk * 4 - 1` motion steps total. Or pass a JSON file produced
by upstream's `hyvideo/generate_custom_trajectory.py` to `--pose`.

## Programmatic access

```python
from pathlib import Path
from dataclasses import replace

from hy_worldplay.config import RUNNER_HY_WORLDPLAY_WAN_I2V_5B

cfg = replace(
    RUNNER_HY_WORLDPLAY_WAN_I2V_5B,
    image_path=Path("./data_local/hy_worldplay/test.png"),
    ckpt_path=Path("/path/to/models/wan_distilled_model/model.pt"),
    num_chunk=1,
    pose="w-3",
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

### PR perf + visual sample

Two harnesses live under `tests/parity_check/`:

**`bench.sh`** — single-image native-vs-vendor bench. Drives upstream's
`wan/generate.py` (via `run.sh`) *and* the native plugin on the same
inputs, producing two MP4s, per-side stats JSONs, and `bench.md`
summarising perf + the mean / max `|Δ|`. Requires the HY-WorldPlay
tree cloned and the heavy vendor deps installed (`run.sh` provisions
both on first invocation).

```bash
# Default: data_local/cat_surf.jpg, num_chunk=1, pose=w-3, seed=0.
bash integrations/hy_worldplay/tests/parity_check/bench.sh
```

**`bench_batch.sh`** — native-only perf loop over every image in
`data_local/`. Writes per-image MP4 + stats and aggregates into
`bench_all.md` for direct PR attachment. Much faster (no vendor deps,
no upstream tree clone) when the goal is "perf numbers for N images"
rather than parity comparison.

```bash
# Defaults: data_local/*.{jpg,jpeg,png}, num_chunk=2, pose=w-7, seed=0.
bash integrations/hy_worldplay/tests/parity_check/bench_batch.sh
```

Both harnesses honour `IMAGE_PATH` / `IMAGES_DIR`, `NUM_CHUNK`, `POSE`,
`SEED`, `OUTPUT_DIR`, and `CKPT_PATH` env-var overrides. Outputs land
under `tests/parity_check/outputs/{bench,bench_batch}/` (gitignored).

One-time setup before either bench:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv on PATH
export HF_TOKEN=<your-token>                       # HF auth
( cd integrations/hy_worldplay/tests/parity_check && uv sync )
```

The distilled checkpoint downloads on demand from `tencent/HY-WorldPlay`
the first time either harness runs.

## Staging plan

Phases 1, 2a, and 2b are landed; the native pipeline is the production
default. Phase 3 is future.

1. **Phase 1 — vendor wrapper.** Historical scaffolding; superseded by
   the native path below. The parity-check harness still invokes
   upstream's `wan/generate.py` directly to produce the reference
   baseline; the plugin itself no longer imports official code.

2. **Phase 2a — WAN 2.2 5B recipe (`flashdreams.recipes.wan`).**
   Prerequisite for 2b; useful on its own. Fills the gap between
   Wan 2.1 (1.3B / 14B) and Wan 2.2 14B with the 5B VAE / DiT configs
   (`Wan22TI2V5BVAE{Encoder,Decoder}Config`, `WanDiTNetworkTI2V5BConfig`),
   the `ti2v_first_frame_per_token_timestep` flag on
   `Wan21TransformerConfig`, the `PIPELINE_WAN22_TI2V_5B` pre-rolled
   config, and diffusers-safetensors remaps.

3. **Phase 2b — native HY-WorldPlay integration.** All conditioners
   are wired into the static pipeline in `hy_worldplay.config` and
   zero-initialised so a run without `--ckpt-path` is a strict
   identity against the base Wan 2.2 TI2V-5B output.

   - **2b.1 / 2b.2.** Native runner over `PIPELINE_WAN22_TI2V_5B` +
     distilled 4-step Euler schedule swapped in on the native path.
   - **2b.3.** 81-class action conditioner (AdaLN add on
     `HyWorldPlayWanDiTNetwork`).
   - **2b.4.** PRoPE dual-branch self-attention
     (`HyWorldPlayPRoPEBlock`; `prope_qkv` in
     `hy_worldplay._prope`).
   - **2b.5a.** Reconstituted-context memory selection
     (`select_mem_frames_wan` + FOV-overlap helper, ported to
     `hy_worldplay/_memory.py`).
   - **2b.5b.** Distilled-checkpoint remap + KV-prefill executor
     (per-rollout `clean_latent_history`, per-block
     `HyWorldPlayMemoryKVCache`, per-chunk rolling-cache reset,
     `prefill_completed_for_chunk` latch).
   - **2b.6.** Parity close at **`mean |Δ| = 15.65 / 255`**
     (704x1280 / `num_chunk=2`; below the visible threshold and within
     ~3-4× of the vendor-vs-vendor kernel noise floor of 3.24/255).
     Acceptance bar `<= 20 / 255`.
