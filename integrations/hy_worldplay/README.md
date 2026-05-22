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
(action + memory + chunked AR + distributed VAE), the runner fills
its mandatory `RunnerConfig.pipeline` slot with an inert
`_NoopPipelineConfig` (a `StreamInferencePipeline` subclass that
overrides `__init__` to skip slot construction) and owns its own
`__init__`. Promotion onto a real `WanInferencePipeline` is phase 2b
(see "Staging plan" below).

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

### Native pipeline (preview)

Phase 2b is migrating the runner off upstream's `WanRunner` onto the
in-tree `WanInferencePipeline` so HY-WorldPlay shares the KV cache,
context-parallelism, profiler, and attention dispatch with the rest of
the `wan*` family. The migration is staged behind a `--use-native-pipeline`
feature flag and lands incrementally:

- **2b.1.** Native runner driving `PIPELINE_WAN22_TI2V_5B` for the
  I2V base case. No action / camera-trajectory / memory conditioning.
- **2b.2.** Scheduler swap to Euler with the upstream-distilled fixed
  4-step schedule. Native mode now uses the same denoising loop as
  the vendor wrapper, so any remaining drift comes from the missing
  conditioners + non-distilled checkpoint.
- **2b.3.** Action conditioner: 81-class discrete labels
  (`trans * 9 + rotate`) embedded by a dedicated MLP and summed into
  the time embedding before the AdaLN modulation projection. Activated
  by `--use-action-conditioning` alongside `--use-native-pipeline`. The
  action MLP's residual head is zero-initialised, so flipping the flag
  on without HY-WorldPlay's distilled weights is a strict identity.
- **2b.4.** Camera-trajectory conditioner: PRoPE
  dual-branch self-attention. Each block runs the stock RoPE attention
  branch *plus* a parallel branch that applies per-frame camera-
  projective transforms (`P = lift(K) @ viewmats`) to Q / K / V via
  `flashdreams.core.attention.prope.prope_qkv`, attends against a
  dedicated second KV cache, and projects through a separate `o_prope`
  linear before summing back into the block output. Activated by
  `--use-camera-conditioning` alongside `--use-native-pipeline`. The
  `o_prope` projection is zero-initialised so the branch contributes
  exactly zero residual until HY-WorldPlay's distilled checkpoint
  loads non-zero weights for it (strict no-op until then).
- **2b.5a.** Reconstituted-context memory **selection**.
  Ports upstream's
  `wan/models/utils.py::select_mem_frames_wan` policy +
  `hyvideo/utils/retrieval_context.py::calculate_fov_overlap_similarity`
  helper to `hy_worldplay/_memory.py`, and threads the per-AR-step
  selected frame indices through the encoder onto
  `HyWorldPlayCtrl.memory_frame_indices`. Activated by
  `--use-memory-selection` alongside `--use-camera-conditioning` (the
  selector needs the bound per-rollout `viewmats` history). The
  Monte-Carlo FOV-overlap sphere is built once per rollout on the
  pipeline device (50_000 points by default, matching upstream's
  `WanInferencePipeline.__init__`). At AR steps where
  `current_frame_idx < context_window_length` the encoder emits
  `memory_frame_indices=None` to mirror upstream's "elif use_memory"
  branch -- no selection runs.
- **2b.5b-part1 (this release).** Distilled-checkpoint weight
  remap. Adds `hy_worldplay/_checkpoint.py::hy_worldplay_distilled_state_dict_transform`,
  which unwraps upstream's `.pt` envelope (`generator` /
  `generator_ema` subkey + `model.` / `_fsdp_wrapped_module.` prefix
  stripping), layers on the base 5B diffusers
  `WanTransformer3DModel` -> `WanDiTNetwork` remap, and adds three
  HY-specific rewrite rules so
  `condition_embedder.action_embedder.linear_{1,2}` lands on
  `action_embedding.{0,2}` and `blocks.{i}.attn1.to_out_prope.0`
  lands on `blocks.{i}.self_attn.o_prope`. Auto-routed through the
  runner's `__post_init__` whenever `--ckpt-path` is supplied
  alongside `--use-action-conditioning` or
  `--use-camera-conditioning`: the transformer's
  `checkpoint_path` is reset to the distilled `.pt` and its
  `state_dict_transform` swapped to the HY remap. Verified end-to-end
  via `strict=True` load against a freshly built
  :class:`HyWorldPlayWanDiTNetwork` (889 keys, 0 missing / 0
  unexpected), which lights up the action MLP's `linear_2` and every
  block's `o_prope` from zero-init to non-zero norms. The
  conditioners now produce real, non-zero residuals on top of the
  base 5B trunk.
- **2b.5b-part2 (this release).** KV-prefill executor structural
  skeleton. Three coupled pieces all wired up end-to-end on the
  HY native path: (a) per-rollout clean-latent history buffer on
  :class:`HyWorldPlayWan21TransformerCache.clean_latent_history`,
  appended via the new `finalize_kv_cache` override that supersedes
  the parent's rolling-window stamp pass (HY mode uses memory cache
  instead); (b) per-block flat
  :class:`HyWorldPlayMemoryKVCache` slot on
  :class:`HyWorldPlayPRoPEBlockCache` that stores K / V at
  upstream's RoPE-collapsed positions `[0, K)` for both the
  standard and PRoPE branches; (c) prefill pass at AR step 0 of
  every chunk past the first, dispatched from
  :meth:`HyWorldPlayWan21Transformer.predict_flow` via the new
  :meth:`prefill_memory_kv_cache` driver, which slices the history
  at the encoder-supplied `memory_frame_indices`, builds RoPE
  freqs at the collapsed `[0, K)` positions via the rope adapter's
  `_freq_components` primitive, and runs
  :meth:`HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache` -- a
  patchify + AdaLN modulation re-pass that calls each block's new
  `prefill_memory_kv` (cross-attn / FFN / head all skipped). The
  dual-branch attention now consumes the memory cache via a
  `cat([memory_K, current_K], dim=seq)` prepend that's a strict
  no-op when the memory cache is empty (chunk 0 baseline). The
  per-chunk rolling-cache reset is owned by the HY cache subclass's
  `start` override, which pokes `_prev_chunk_idx` so the inherited
  `before_update(autoregressive_index)` accepts the synthetic
  "next chunk" transition. Per-rollout viewmats / Ks / action
  buffers are still per-AR-step on the ctrl as of this release;
  `_slice_per_frame` falls back to a `[:K]` truncation that is
  parity-incorrect (flagged in code with a TODO and pinned by
  CPU tests). The followup work (full parity + per-rollout
  metadata threading + sub-venv removal + default flag flip) is
  tracked under **2b.5b-part2-followup** below.

- **2b.5b-part2-followup (this release).** Two pieces landed
  together because the second was discovered while validating
  the first:
  - *Per-rollout metadata threading.* `HyWorldPlayCtrl` gains
    `rollout_viewmats` / `rollout_Ks` / `rollout_action` slots,
    populated by `HyWorldPlayWanCtrlEncoder.forward` from the
    full-trajectory tensors that the encoder already maintains
    (`_viewmats` / `_intrinsics` / `_action_labels`) -- previously
    only sliced into the per-AR-step `viewmats` / `Ks` / `action`
    fields. The prefill driver's parity-incorrect
    `_slice_per_frame` stub is replaced by
    `_index_rollout_buffer`, which uses
    `tensor.index_select(axis, memory_frame_indices)` against the
    rollout buffer when bound and falls back to the per-step
    slice only when the conditioner is disabled (the
    conditioner's own gate makes the slice content unobservable
    in that case). The patchify rebuild passes the new fields
    through unchanged. CPU tests (4 new in `test_prefill.py`)
    pin defaults / patchify survival / encoder attach / unbound-
    conditioner fallback.
  - *GPU smoke validates structural skeleton.* The 2-chunk
    rollout boots end-to-end on a real RTX 6000 Pro with the
    distilled checkpoint, ~28 GB peak GPU memory at 256x448
    pixel resolution. The prefill executor was instrumented with
    a synthetic `memory_frame_indices=[0, 1, 2, 3]` (the
    upstream FOV selector has known boundary issues with short
    rollouts -- see Known Quirks below) and confirmed to (a)
    fire on chunk 1, (b) read the per-rollout buffers via
    `_index_rollout_buffer`, (c) populate the
    `HyWorldPlayMemoryKVCache` per block, and (d) feed
    `forward_dual_branch` so the noise prediction completes
    without NaN. Three drive-by fixes shipped to make the smoke
    work:
    - `wan22_ti2v_5b_vae_state_dict_transform` (in
      `flashdreams/recipes/wan/autoencoder/vae.py`) now remaps
      the `mid_block.resnets.{0,1}.{norm1,conv1,norm2,conv2,
      conv_shortcut}` keys to the `middle.{0,2}.residual.{0,2,3,
      6}` Sequential layout. Without this, 12 VAE params per
      side stayed on `meta` device and the pipeline crashed with
      `Cannot copy out of meta tensor` at `.to(device)`. This is
      a base recipe fix that benefits *all* Wan22 5B native
      callers, not just HY-WorldPlay.
    - `_native_runner._bind_camera_data` now `unsqueeze(0)`s the
      viewmats / Ks tensors so they reach
      `flashdreams.core.attention.prope.prope_qkv` in the
      required `[batch=1, cameras, 4, 4]` rank.
    - `HyWorldPlayWanCtrlEncoder._compute_memory_indices` casts
      the bound viewmats to fp32 before the
      `.cpu().numpy()` round-trip in
      `select_memory_frame_indices` -- numpy has no bf16 ABI so
      the bf16 cast applied for PRoPE attention can't survive
      the round-trip.
    - `_native_runner.run` casts the preprocessed first-frame
      tensor to the pipeline's parameter dtype so the residual
      VAE's first `CausalConv3d` doesn't see a fp32-vs-bf16 dtype
      mismatch.

- **2b.5b-part2-followup (still pending).** Three remaining
  items: (1) end-to-end parity diff vs the phase-1 vendor-wrapper
  baseline at production resolution (704x1280) with the full
  upstream FOV memory selector; (2) drop the parity sub-venv
  (`sageattention`, `cloudpickle`, `accelerate`,
  `transformers==4.57.6`) once parity passes; (3) flip
  `--use-native-pipeline` to the default for the
  `hy-worldplay-wan-i2v-5b` recipe.

Try the native path (single GPU, runs in the main `flashdreams`
venv -- no parity sub-venv needed):

```bash
uv run flashdreams-run hy-worldplay-wan-i2v-5b \
    --use-native-pipeline \
    --use-action-conditioning \
    --use-camera-conditioning \
    --use-memory-selection \
    --image-path ./assets/img/test.png \
    --num-chunk 1 \
    --pose "w-4" \
    --output-dir outputs
```

`--use-action-conditioning` and `--use-camera-conditioning` are
independent toggles -- either, both, or neither can be set, depending
on which conditioner you want to ablate. Both share the same encoder /
transformer / network subclass tree; flipping either flag triggers the
swap, and the camera flag additionally enables the PRoPE dual-branch
block path on the DiT. `--use-memory-selection` requires
`--use-camera-conditioning` (the FOV-overlap selector consumes the
per-rollout viewmats binding) and is a no-op without it; setting it
without `--use-native-pipeline` is silently ignored on the vendor
wrapper path.

The native path's parity status as of this release: the prefill
executor structural skeleton (history buffer, per-block memory KV
cache, collapsed-position RoPE prefill, dual-branch concat, per-chunk
rolling-cache reset), the per-rollout viewmats / Ks / action
threading, and the GPU boot path have all landed. 99 CPU tests pin
the structural invariants and a 2-chunk GPU smoke at 256x448 confirms
the prefill executor runs on real bf16 weights without crashing. What
remains for full numerical parity is the end-to-end parity diff at
production resolution against the phase-1 vendor-wrapper baseline,
which feeds the parity sub-venv removal and the
`--use-native-pipeline` default flip (the still-pending three
**2b.5b-part2-followup** items). For bit-for-bit parity with upstream
`wan/generate.py` today, use the default (vendor-wrapper) invocation
above. See
[`docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md`](../../docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md)
for the full design.

#### Known quirks observed during GPU smoke validation

These do not block parity but are worth tracking for the eventual
parity diff:

- **Prefill executor fires once per denoising step rather than
  once per chunk.** `_is_first_step_of_chunk` returns `True` at
  every scheduler step of chunk N (N > 0), causing
  `prefill_memory_kv_cache` to fire `num_inference_steps` times
  per chunk instead of once. The prefill is idempotent (same
  inputs -> same K/V), so this is correct-but-wasteful: it costs
  `num_inference_steps - 1 = 3` extra prefill passes per chunk on
  the distilled 4-step schedule. Worth fixing as a perf
  optimization in a separate phase.
- **Upstream FOV-selector boundary on short rollouts.** The
  upstream `select_mem_frames_wan` algorithm (faithfully ported
  in `_memory.py`) has `historical_clip_starts` that allow clip
  starts whose `[start, start+pred_latent_size)` range overlaps
  the temporal-context window when the FOV-distance scorer picks
  the latest start. With short rollouts (the 2-chunk smoke at
  21 frames of history per chunk), the resulting set-union can
  shrink below the requested `memory_frames`, which the final
  assertion catches. Production rollouts with larger
  `temporal_context` and many chunks of history avoid this. The
  GPU smoke pins the prefill executor itself by monkey-patching
  the encoder to feed a fixed `memory_frame_indices=[0,1,2,3]`
  list bypassing the FOV scorer; full FOV-selected runs need a
  longer rollout or a relaxed `pred_latent_size` constraint.

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

   Landed deliverables (importable from `flashdreams.recipes.wan`):
   - **VAE.** `Wan22TI2V5BVAEEncoderConfig` /
     `Wan22TI2V5BVAEDecoderConfig` flip on the 5B-specific
     16x-spatial, 48-channel, residual + outer-patchify (patch_size
     = 2) knobs on the generalised `WanVAE`. The diffusers
     ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae`` safetensors loads
     directly via `wan22_ti2v_5b_vae_state_dict_transform` (no
     repacking).
   - **DiT.** `WanDiTNetworkTI2V5BConfig` configures the 3072d / 30-
     layer / 48-channel-latent / no-CLIP-cross-attention 5B variant
     on the existing `WanDiTNetwork`. The shared `Block` / `Head`
     AdaLN modulation path squeezes the modulation axis so both
     scalar and per-token timesteps broadcast correctly; the network
     `forward` dispatches between them based on the timestep tensor
     rank.
   - **Transformer.** A new
     `Wan21TransformerConfig.ti2v_first_frame_per_token_timestep`
     flag composes with the existing `stamp_image_latent` mask-
     inject path: at AR step 0 the scheduler's scalar timestep is
     rewritten to a per-token tensor with ``t=0`` at the first-frame
     conditioning tokens (driven by the I2V mask) and the
     scheduler ``t`` elsewhere. AR >= 1 keeps the scalar shape so
     CUDA-graph capture stays stable.
   - **Pipeline.** No new pipeline module: the existing
     `WanInferencePipeline` covers TI2V 5B by configuring the I2V
     control encoder around the 5B VAE plus the transformer flags
     above. `PIPELINE_WAN22_TI2V_5B` is the pre-rolled config.
   - **Checkpoint remap.** `wan22_ti2v_5b_dit_state_dict_transform`
     remaps the diffusers `WanTransformer3DModel` layout to the
     bare `WanDiTNetwork` keys (same structural mapping the
     `fastvideo_causal_wan22` integration uses for the 14B MoE).

3. **Phase 2b (this directory, follow-up to 2a).** Recipe-level
   integration on top of the new flashdreams Wan 2.2 5B recipe.
   Staged behind a `--use-native-pipeline` feature flag and shipped
   across five sub-PRs; the design is captured in
   [`docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md`](../../docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md).

   - **2b.1 (landed).** Native `HyWorldPlayWanI2VNativeRunner` drives
     `PIPELINE_WAN22_TI2V_5B` for the I2V base case (no
     action / camera / memory). Selected by
     `--use-native-pipeline`; vendor wrapper stays the default so the
     phase-1 parity bar is preserved. See "Native pipeline (preview)"
     under [Run](#run).
   - **2b.2 (landed).** New `FlowMatchEulerDiscreteSchedulerConfig`
     in `flashdreams.infra.diffusion.scheduler` carries upstream's
     distilled 4-step fixed schedule
     `(1000, 960, 888.89, 727.27, 0)`. `HyWorldPlayWanI2VRunnerConfig.__post_init__`
     swaps it in only when `use_native_pipeline=True`; the base
     `PIPELINE_WAN22_TI2V_5B` recipe keeps its
     `FlowMatchUniPCSchedulerConfig` for non-HY callers.
   - **2b.3 (landed).** Action conditioner (81-class discrete →
     time-embed AdaLN add) on a `HyWorldPlayWanDiTNetwork` subclass
     plus an action-aware encoder / transformer pair, gated behind
     `--use-action-conditioning`. The action MLP's residual head
     ships zero-initialised so the conditioner is a strict identity
     until HY-WorldPlay's distilled checkpoint is layered on top.
   - **2b.4 (landed).** Camera-trajectory conditioner (PRoPE dual-
     branch self-attention) on a `HyWorldPlayPRoPEBlock` subclass,
     plus the `prope_qkv` math port to
     `flashdreams.core.attention.prope` and per-AR-step
     `viewmats` / `Ks` slicing through the existing encoder. Gated
     behind `--use-camera-conditioning`. The block's `o_prope`
     projection ships zero-initialised so the PRoPE branch
     contributes exactly zero residual until the distilled
     checkpoint loads on top (strict identity until then).
   - **2b.5a (landed).** Reconstituted-context memory **selection**.
     Ports upstream's `select_mem_frames_wan` policy +
     `calculate_fov_overlap_similarity` helper to
     `hy_worldplay/_memory.py`, adds `memory_frame_indices` to
     `HyWorldPlayCtrl`, and arms the encoder via
     `HyWorldPlayWanCtrlEncoder.set_memory_config` so each AR step
     with enough history (`current_frame_idx >=
     context_window_length`) emits a sorted, deduplicated frame-
     index list onto the per-AR-step ctrl. Gated behind
     `--use-memory-selection` (requires `--use-camera-conditioning`
     so the viewmats history is bound). The list is produced but
     not yet consumed -- the KV-prefill executor lands in 2b.5b.
   - **2b.5b.** KV-prefill executor (transformer pre-pass with
     `is_cache=True` on the selected memory frames) + the
     `flashdreams.core.attention.kvcache.BlockKVCache` arbitrary-
     position write extension it needs; HY-WorldPlay distilled-
     weight remap (`action_embedding`, `o_prope`, memory layers);
     re-run the parity check against the phase-1 baseline; drop the
     parity sub-venv (`sageattention`, `cloudpickle`, `accelerate`,
     `transformers==4.57.6` are no longer needed once the upstream
     tree imports are gone); flip `--use-native-pipeline` to
     default.

4. **Phase 3 — future.** HunyuanVideo-1.5 8B variant
   (`hyvideo/generate.py` upstream). Heavier integration: multiple
   text encoders (Qwen2.5-VL-7B, ByT5, Glyph-SDXL-v2), gated vision
   encoder (FLUX.1-Redux-dev), 8-way SP, distilled / RL-tuned model
   variants.
