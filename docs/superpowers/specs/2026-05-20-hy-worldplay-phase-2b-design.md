<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay Phase 2b — Native Pipeline Migration Design Spec

**Status:** approved (2026-05-20)
**Owner:** wenqing
**Tracking:** `wenqing/hy-worldplay-integration` branch
**Phase 2a prerequisite:** landed
([`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B`](../../../flashdreams/flashdreams/recipes/wan/config.py)).

## Goal

Replace the phase-1 vendor-wrapper runner (`HyWorldPlayWanI2VRunner` →
upstream `wan.generate.WanRunner`) with a native flashdreams
[`WanInferencePipeline`](../../../flashdreams/flashdreams/recipes/wan/pipeline.py)
driven by `PIPELINE_WAN22_TI2V_5B`, and add HY-WorldPlay's three
conditioner extensions on top:

1. Action conditioning (81-class discrete labels, AdaLN modulation).
2. Camera-trajectory conditioning (PRoPE in dual-branch self-attention).
3. Reconstituted-context memory (frame-index selection policy + KV
   prefill).

Once the native path is feature-complete and parity-validated, retire
the parity sub-venv: HY-WorldPlay's heavy upstream deps
(`sageattention`, `cloudpickle`, `accelerate`, `transformers==4.57.6`)
become unnecessary and the run path collapses back into the main
flashdreams venv.

The phase-1 user-facing CLI slug (`flashdreams-run
hy-worldplay-wan-i2v-5b`) stays stable across the transition.

## Why split into multiple PRs

A complete phase 2b is ~2,500-4,000 LoC plus tests, parity validation,
and incremental verification against a numeric parity bar. Shipping in
a single PR risks (a) merging buggy conditioners and breaking the
phase-1 parity invariant, (b) review fatigue, and (c) bisect pain when
a downstream regression surfaces. We decompose into 5 sub-PRs, each
under ~1,000 LoC and each ending in a verifiable state.

## Sub-PR decomposition

| sub-PR | scope | LoC | depends on |
|---|---|---|---|
| **2b.1 (landed)** | Native runner driving `PIPELINE_WAN22_TI2V_5B`, I2V base case only (no conditioners). Behind a `--use-native-pipeline` feature flag; vendor wrapper stays default. Single-GPU only. | ~300-500 | none |
| **2b.2 (landed)** | Add `FlowMatchEulerDiscreteSchedulerConfig` (with the upstream-specific distilled 4-step hardcoded schedule) to `flashdreams.infra.diffusion.scheduler`. Swap the scheduler in HY-WorldPlay's `__post_init__` only; `PIPELINE_WAN22_TI2V_5B` stays neutral with UniPC for non-HY callers. | ~200-300 | 2b.1 |
| **2b.3 (landed)** | Action conditioner: `HyWorldPlayWanDiTNetworkConfig` + `HyWorldPlayWanDiTNetwork` subclass with a zero-init `action_embedding` MLP summed into `temb` before AdaLN modulation. Companion `HyWorldPlayWanCtrlEncoder` slices per-AR-step labels; `HyWorldPlayWan21Transformer` threads them via `network_extra_kwargs`. Gated by `--use-action-conditioning`. | ~300-500 | 2b.2 |
| **2b.4 (landed)** | Camera-trajectory conditioner: port `prope_qkv` to `flashdreams.core.attention.prope` and ship the dual-branch RoPE+PRoPE attention as `HyWorldPlayPRoPESelfAttention` + `HyWorldPlayPRoPEBlock`. Plumb viewmats + intrinsics through `HyWorldPlayCtrl` / encoder. Gated by `--use-camera-conditioning`. `o_prope` zero-init keeps the branch a strict identity at random init. | ~900 | 2b.3 |
| **2b.5** | Memory module + KV-prefill hook: port `select_mem_frames_wan` selection policy, extend transformer cache with "prefill from these frame indices" semantics, hook into `Wan21Transformer.predict_flow` for per-chunk prefill at step 0. Drop parity sub-venv. Re-run parity. Flip `--use-native-pipeline` to default. | ~400-700 + cleanup | 2b.4 |

Total: ~1,800-3,000 LoC of new code + ~500-1,000 LoC of tests + docs.

## Architecture overview

### Phase-1 (current state — vendor wrapper)

```
HyWorldPlayWanI2VRunner.run()
  └─> upstream WanRunner.predict()
        ├─> upstream WanPipeline (action + camera + memory + KV cache)
        ├─> upstream FlowMatchEulerDiscreteScheduler (hardcoded 4-step)
        └─> upstream par_vae decoder

flashdreams `pipeline` slot: _NoopPipelineConfig (inert, satisfies the
RunnerConfig contract; never executed).
```

### Phase-2b (target — native pipeline)

```
HyWorldPlayWanI2VRunner.run()           # native mode
  ├─> pipeline.initialize_cache(text, image)         # WanInferencePipeline
  └─> for ar_idx in range(num_chunk):
        ├─> pipeline.generate(ar_idx, cache,
        │       input=HyWorldPlayCtrl(            # encoder payload
        │           i2v_first_frame=...,          # phase-1 conditioner
        │           action=...,                   # 2b.3
        │           viewmats=..., Ks=...,         # 2b.4
        │           memory_frame_indices=...,     # 2b.5
        │       ))
        └─> pipeline.finalize(ar_idx, cache)

Internally:
  encoder (WanI2VCtrlEncoder + HY conditioners)
    -> diffusion_model (Wan21Transformer w/ KV cache, CP, CUDA graph,
                        per-token timestep at AR-0, action AdaLN add,
                        PRoPE dual-branch attention, memory KV prefill)
       -> FlowMatchEulerDiscreteScheduler
    -> decoder (Wan22TI2V5BVAEDecoder, streaming)

flashdreams `pipeline` slot: WanInferencePipelineConfig (full native).
```

## Why these decomposition choices

### Why feature flag at 2b.1 instead of full cutover

The phase-1 vendor wrapper is the *only* mechanism currently shipping
bit-exact-against-upstream output, and the parity check enforces it.
Cutting over to the native path before all conditioners + scheduler are
in place would break that invariant immediately. The feature flag lets
us:

- Land the native runner skeleton and verify the pipeline boots, the AR
  loop runs, the VAE decodes correctly — all without touching the parity
  bar.
- Iterate on conditioners independently. Each follow-up PR can
  numerically diff `--use-native-pipeline` against the vendor wrapper
  baseline and surface regressions one conditioner at a time.
- Keep the user-facing slug stable. Users get the wrapper by default;
  early adopters opt in.

### Why scheduler before conditioners (2b.2 before 2b.3)

Upstream's distilled checkpoint expects the fixed 4-step
`[1000, 960, 888.89, 727.27, 0]` schedule. Until we match that, every
conditioner PR would show parity drift attributable to the scheduler,
not the conditioner under test — making code review and parity
debugging much harder. Landing the scheduler first means subsequent
conditioner PRs can run a single-conditioner parity diff.

### Why action before camera (2b.3 before 2b.4)

Action is the simplest conditioner — a discrete label → MLP → add to
`temb` → AdaLN. The transformer subclass and `network_extra_kwargs`
plumbing built here become the template for camera and memory.
Landing it first gives us a clean, isolated PR demonstrating the
extension seam works end-to-end. Camera's PRoPE machinery is far
heavier and will reuse the same plumbing pattern.

### Why memory last (2b.5)

Memory selection is a no-op until we have at least one cross-chunk AR
rollout to select frames from. The memory module also requires a
KV-prefill hook in `Wan21Transformer` that's most cleanly added once
the action + camera conditioners' tensors are flowing through the
network (so the prefilled cache entries carry the right
action/camera-conditioned K/V values, not action-zero K/V values).

### Why parity sub-venv removal at 2b.5

The parity sub-venv exists because the vendor wrapper imports
`sageattention`, `cloudpickle`, `accelerate`, and a 4.x-line
`transformers`. Once the native path is the default and the wrapper
ships behind a back-compat flag (or is removed outright), none of
those deps are needed. The parity check itself stays — it just runs
against the native path instead of via the wrapper.

## Sub-PR 2b.1 design (this session)

### Files touched

| file | nature |
|---|---|
| `integrations/hy_worldplay/hy_worldplay/runner.py` | extend `HyWorldPlayWanI2VRunnerConfig` with `use_native_pipeline` flag + branch logic; extend `HyWorldPlayWanI2VRunner` with native-mode AR loop |
| `integrations/hy_worldplay/hy_worldplay/_vendor_pipeline.py` | (unchanged for backward compat) |
| `integrations/hy_worldplay/hy_worldplay/_native_runner.py` | NEW — native-mode helpers (image preprocessing, AR loop, mp4 writer) factored out so `runner.py` stays readable |
| `integrations/hy_worldplay/tests/test_smoke.py` | extend with native-mode config assertions |
| `integrations/hy_worldplay/tests/test_native_smoke.py` | NEW — `ci_gpu`-marked end-to-end test |
| `integrations/hy_worldplay/README.md` | "Native pipeline (preview)" section under "Run" |

### Detailed behaviour

**Config switch.** `HyWorldPlayWanI2VRunnerConfig` gains:

```python
use_native_pipeline: bool = False
"""When ``True``, route inference through the in-tree
``PIPELINE_WAN22_TI2V_5B`` instead of upstream's ``WanRunner``. Native
path is feature-flagged for phase 2b incremental rollout."""
```

The `pipeline` field default factory becomes a function that returns
`PIPELINE_WAN22_TI2V_5B` if `use_native_pipeline=True` and
`_NoopPipelineConfig()` otherwise.

**Two-mode runner.** `HyWorldPlayWanI2VRunner.__init__` branches:

- `use_native_pipeline=False` (default): existing behaviour
  (vendor wrapper, `_NoopPipeline`, delegates to upstream).
- `use_native_pipeline=True`: drops all upstream-tree imports
  (`wan_transformer`, `hy_worldplay_repo_root`, `ckpt_path` checks
  become advisory), constructs `self.pipeline = config.pipeline.setup()`
  via the standard `Runner` machinery, and uses the native `run()` path.

**Native `run()` path.**

```python
def _run_native(self):
    image = _preprocess_first_frame(config.image_path)  # PIL -> tensor
    prompt = self._resolve_prompt()
    cache = self.pipeline.initialize_cache(
        text=prompt,
        negative_text=config.negative_prompt,
        image=image,
        height=config.pixel_height,
        width=config.pixel_width,
    )
    chunks = []
    for ar_idx in range(config.num_chunk):
        decoded_chunk = self.pipeline.generate(ar_idx, cache)
        chunks.append(decoded_chunk)
        self.pipeline.finalize(ar_idx, cache)
    video = torch.cat(chunks, dim=-3)  # along T axis (post-shape audit)
    if self.is_rank_zero:
        _write_mp4(video, config.output_dir / f"{config.runner_name}.mp4",
                   fps=config.fps)
```

**Image preprocessing.** `_preprocess_first_frame(path)`:

1. Load PIL image, convert to RGB.
2. Resize to `(pixel_height, pixel_width)` with aspect-ratio centre crop
   (mirror upstream's `(704, 1280)` policy).
3. Convert to `[1, 3, H, W]` float32 tensor in `[-1, 1]` range to match
   `WanI2VCtrlEncoder`'s expected input.

**Limitations of 2b.1 (intentional, documented in README):**

- No action / camera / memory conditioning (model gets default-zero
  conditioning for those — output will *not* match the vendor wrapper).
- Single GPU only. CP / `par_vae` parity comes later.
- Uses the existing `FlowMatchUniPCSchedulerConfig` from
  `PIPELINE_WAN22_TI2V_5B` (scheduler swap is 2b.2).
- No numeric parity check yet; CI smoke just asserts "mp4 of expected
  shape exists".

### Tests

**`tests/test_smoke.py` additions (CPU-only):**

```python
def test_use_native_pipeline_routes_to_wan_pipeline():
    """Native flag must swap the pipeline slot from noop to the real
    PIPELINE_WAN22_TI2V_5B."""
    cfg_native = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
    )
    assert isinstance(cfg_native.pipeline, WanInferencePipelineConfig)
    assert cfg_native.pipeline.recipe_name == "wan22-ti2v-5b"
    # Default (wrapper) path is unchanged:
    cfg_wrapper = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
    )
    assert isinstance(cfg_wrapper.pipeline, _NoopPipelineConfig)
```

**`tests/test_native_smoke.py` (NEW, `ci_gpu`):**

```python
@pytest.mark.ci_gpu
def test_native_pipeline_end_to_end(tmp_path):
    """Full --use-native-pipeline rollout on a single GPU produces a valid
    mp4 of the expected shape. No parity assertion (that comes at 2b.5)."""
    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        image_path=Path("integrations/hy_worldplay/tests/assets/cat_surf.jpg"),
        prompt="A cat surfing on ocean waves",
        num_chunk=1,
        pose="w-4",
        output_dir=tmp_path,
    )
    cfg.setup().run()
    mp4 = tmp_path / "hy-worldplay-wan-i2v-5b.mp4"
    assert mp4.exists() and mp4.stat().st_size > 1024
```

(A small fixture image `tests/assets/cat_surf.jpg` is committed as part
of this sub-PR so the test is hermetic. ~10 KB.)

### README change

Add a new section under "Run" titled **"Native pipeline (preview)"**
showing:

```bash
uv run flashdreams-run hy-worldplay-wan-i2v-5b \
    --use-native-pipeline \
    --image-path ./assets/cat_surf.jpg \
    --num-chunk 1 \
    --pose "w-4" \
    --output-dir outputs
```

with a note: "Native pipeline ships incrementally across phases
2b.1-2b.5. Phase 2b.1 supports the I2V base case only — action,
camera-trajectory, and reconstituted-context-memory conditioning land
in 2b.3 / 2b.4 / 2b.5 respectively. Use the default (vendor-wrapper)
path for parity with upstream's `wan/generate.py`."

## Sub-PR 2b.2 design (landed)

Add `FlowMatchEulerDiscreteSchedulerConfig` +
`FlowMatchEulerDiscreteScheduler` to
`flashdreams.infra.diffusion.scheduler`. The config carries an
optional `fixed_timesteps: tuple[float, ...] | None` field; when set,
the scheduler skips the linspace+warp derivation and uses the
caller-supplied schedule directly. Default (`None`) reproduces the
standard diffusers `FlowMatchEulerDiscreteScheduler` behaviour
(`linspace(sigma_max, sigma_min, N+1)[:-1]` warped by `shift`, with a
trailing `0.0`).

The scheduler swap is **HY-WorldPlay-scoped**, not a global change to
`PIPELINE_WAN22_TI2V_5B`. Rationale: the base Wan 2.2 5B recipe is a
neutral building block other integrations may want; binding it to the
distilled 4-step Euler schedule would break non-distilled callers
(40-step UniPC is the right default for the non-distilled
checkpoint). HY-WorldPlay's `__post_init__` already deep-copies the
pipeline config; the same hook now swaps the scheduler on the copy:

```python
self.pipeline.diffusion_model.scheduler = (
    FlowMatchEulerDiscreteSchedulerConfig(
        num_inference_steps=4,
        fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
    )
)
```

Parity bar: with the native runner from 2b.1 + the scheduler swap
from 2b.2, the per-frame uint8 RGB delta vs the vendor wrapper
baseline should narrow by the scheduler component, leaving only the
conditioner-driven residual (action / camera / memory still missing).
Sub-PR 2b.3 closes the next slice.

## Sub-PR 2b.3 design (landed)

**Action conditioner.** Upstream represents action as a discrete
81-class label per latent (`trans_class * 9 + rotate_class`), embedded
via a sinusoidal timestep projection + `TimestepEmbedding` MLP, and
added to the time embedding `temb` before AdaLN modulation.

Native implementation (shipped in `integrations/hy_worldplay/hy_worldplay/_action.py`):

- `HyWorldPlayWanDiTNetworkConfig` / `HyWorldPlayWanDiTNetwork` subclass
  the Wan 2.2 TI2V 5B DiT, adding an `action_embedding` MLP (same
  shape as `time_embedding`: ``Linear(freq_dim, dim) → SiLU → Linear(dim, dim)``)
  with the residual head zero-initialised. The overridden `forward()`
  sinusoidally encodes the per-latent labels, runs them through the
  MLP, and adds the result to the time embedding before the
  modulation projection. Per-frame embeddings are
  ``repeat_interleave``d to per-token granularity to match the
  post-patchify token axis.
- `HyWorldPlayWanCtrlEncoderConfig` / `HyWorldPlayWanCtrlEncoder`
  subclass `I2VCtrlEncoder`. The runner binds the per-rollout
  81-class label tensor via `set_action_labels()`; each AR step
  slices ``[ar_idx * len_t : (ar_idx + 1) * len_t]`` and attaches it
  to a `HyWorldPlayCtrl` payload (subclass of `I2VCtrl`).
- `HyWorldPlayWan21TransformerConfig` / `HyWorldPlayWan21Transformer`
  subclass `Wan21Transformer` with two thin overrides:
  `predict_flow` reads `input.action` and forwards it as
  ``network_extra_kwargs["action"]`` (the base already plumbs that
  through to the network forward); `patchify_and_maybe_split_cp`
  rebuilds the payload as a `HyWorldPlayCtrl` so the action slice
  survives the patchify (the base rebuilds as a plain `I2VCtrl` and
  would drop it).

Runner plumbing: `HyWorldPlayWanI2VRunnerConfig` gained a
`use_action_conditioning: bool = False` flag; when set together with
`use_native_pipeline`, `__post_init__` swaps the deep-copied
pipeline's encoder + transformer + network slots in-place (only for
stock instances — user overrides are respected). The native runner's
`_bind_action_labels()` parses the existing `pose` field via the new
`hy_worldplay._pose` module (a port of upstream's
`pose_string_to_json` + `pose_to_input`) and binds the labels on the
encoder before the AR loop.

CP / multi-rank: the action embedding's per-frame → per-token
expansion currently asserts `cp_size == 1`; multi-rank support is
introduced together with the PRoPE camera path in 2b.4 (both
features need the same `split_inputs_cp` wiring).

Parity stance: the residual head's zero-init means the action
contribution is exactly zero at random / zero init, so the native
path stays parity-aligned with the 2b.2 baseline even with
`--use-action-conditioning` on. Real action conditioning becomes
active once HY-WorldPlay's distilled checkpoint loads on top
(2b.5 weight remap).

## Sub-PR 2b.4 design (landed)

**Camera-trajectory conditioner (PRoPE).** Heaviest conditioner sub-PR.
Upstream's camera conditioning fuses positional information from
per-latent `viewmats` (4×4 W2C matrices) and `intrinsics` (3×3) into
the self-attention Q/K/V via PRoPE (projective positional encoding):

```
query_prope, key_prope, value_prope = prope_qkv(
    query, key, value, viewmats=viewmats, Ks=Ks
)
hidden_states_prope = sdpa(query_prope, key_prope, value_prope, ...)
hidden_states = self.o(rope_branch) + self.o_prope(prope_branch)
```

Implementation as shipped:

- `flashdreams.core.attention.prope` (new core module) ports
  `prope_qkv` + the per-camera 4×4 block-diagonal projection helpers.
  Cross-checked against a numpy reference in
  `flashdreams/tests/test_prope.py` (6 tests covering the intrinsic
  and intrinsic-free branches, identity round-trip, and the head-dim
  / camera-divisibility shape contracts). One precision fix vs
  upstream: the lift-K helper now allocates with the input dtype so
  float64 callers don't get a silent fp32 downcast on the
  `out[..., :3, :3] = Ks` write (no observable diff at fp32 / bf16).
- `hy_worldplay._camera` ships three classes:
  - `HyWorldPlayPRoPESelfAttention`: subclass of `SelfAttention` with
    a parallel `o_prope` linear (zero-init) and an independent
    `attn_op_prope`. Its `forward_dual_branch` computes Q/K/V once,
    writes raw K/V to the standard cache and PRoPE-transformed K/V to
    a second cache, runs two SDPA calls, applies `apply_fn_o` to the
    PRoPE branch output, then sums `self.o(rope) + self.o_prope(prope)`.
  - `HyWorldPlayPRoPEBlockCache`: extends `BlockCache` with a
    `prope_self_attn: BlockKVCache` slot and routes
    `before_update` / `after_update` to both caches.
  - `HyWorldPlayPRoPEBlock`: subclass of `Block` that swaps in the
    dual-branch self-attn and overrides `initialize_cache` /
    `forward` to thread `viewmats` + `Ks` through. The block raises
    a clear `ValueError` if `viewmats` is missing so misconfigured
    runs surface at the first block invocation, not as a confusing
    parity drift later.
- `HyWorldPlayWanDiTNetworkConfig` gains a `use_prope_blocks: bool`
  knob; `HyWorldPlayWanDiTNetwork._build_block` returns
  `HyWorldPlayPRoPEBlock` instances when it's set (and the `forward`
  routes `viewmats` / `Ks` through `block_extra_kwargs`). The action-
  only path keeps the stock `Block` so 2b.3 callers are not affected.
- `HyWorldPlayCtrl` gains optional `viewmats` / `Ks` fields;
  `HyWorldPlayWanCtrlEncoder.set_camera_data` binds them per-rollout
  and each `forward` slices the per-AR-step window.
  `HyWorldPlayWan21Transformer.predict_flow` threads them via
  `network_extra_kwargs` and `patchify_and_maybe_split_cp` preserves
  them through the I2V payload rebuild.
- `HyWorldPlayWanI2VRunnerConfig.use_camera_conditioning` flag.
  `__post_init__` reuses the same encoder / transformer / network
  subclass swap as `use_action_conditioning` (the two share a single
  subclass tree) and additionally flips `use_prope_blocks=True` on
  the network config. The native runner parses the pose string into
  `(viewmats, Ks)` via the shared `_pose.parse_pose_data` helper and
  binds them on the encoder before the rollout.

Parity caveats:
- CP > 1 is intentionally gated off here (both the action branch and
  the new PRoPE branch); multi-rank lands in a follow-up alongside
  memory.
- `sageattention` is replaced by flashdreams' native SDPA-backed
  attention (single `attn_op_prope: RingAttention`). Expect ~1-2 ULP
  delta vs upstream attributable to the kernel switch.

Parity target: with action + camera + scheduler all matched, the
only remaining residual should come from the memory module (still
not implemented) and from the `sageattention` → native-attention
substitution.

## Sub-PR 2b.5 design (later)

**Memory module + KV prefill + cleanup.**

Upstream's memory is a frame-index *selection policy* feeding a
one-shot per-chunk KV cache prefill:

- For `chunk_i >= 1` and `chunk_i * 4 >= context_window_length`, call
  `select_mem_frames_wan` to pick (a) `temporal_context_size` recent
  frames + (b) `memory_frames - temporal_context_size` older frames
  scored by FOV overlap with the current camera pose.
- At denoising step 0 of each chunk, prefill the per-layer KV cache
  by running the DiT with `is_cache=True` over the selected frames'
  latents + actions + camera (not the current chunk).
- For the remaining denoising steps, the regular forward pass
  concatenates cached KV with the current chunk's KV before attention.

Native implementation:

- Port `select_mem_frames_wan` and the FOV overlap utility from
  `wan/models/utils.py` to `hy_worldplay/memory.py`.
- Extend `Wan21TransformerCache` (or add a `HyWorldPlayWanTransformerCache`
  subclass) with explicit "prefill these frame indices" semantics.
  flashdreams already has `BlockKVCache` with sink+window; we add a
  "manual prefill from frame index list" entry point.
- `HyWorldPlayWanTransformer.predict_flow` adds a hook at AR step 0 of
  each chunk to run the prefill pass before the regular denoising
  loop.
- Encoder produces `memory_frame_indices` per chunk; plumb via
  `network_extra_kwargs`.

**Sub-venv removal:** with the native runner default-on and no upstream
imports needed:

- Delete `integrations/hy_worldplay/tests/parity_check/pyproject.toml`
  and `uv.lock` (or simplify to a no-deps stub — the parity *check*
  itself stays, but it now runs against the native pipeline using the
  main flashdreams venv, not a separate sub-venv).
- Remove the `tool.uv.override-dependencies = ["transformers==4.57.6"]`
  line that the phase-2a follow-up added (no longer needed once
  `sageattention` / `cloudpickle` / `accelerate` and the 4.x
  `transformers` line are gone).
- Update `integrations/hy_worldplay/run-docker.sh` to drop the
  `uv run --project ${PARITY}` layer; container invocation collapses
  to a direct `uv run flashdreams-run ...`.
- README "Install" section collapses from two layers to one.

**Final parity:** re-run `tests/parity_check/run.sh` (now driving the
native pipeline). Numeric per-frame uint8 RGB delta should be ≤ ε
(where ε is the documented mean-|Δ| bar from phase-1's parity check —
~5/255 was the original threshold, now ideally ≤ 1/255 since we no
longer have the torch-version drift).

**Flip the default:** `use_native_pipeline: bool = True` becomes the
new default. The vendor wrapper stays available for one release as a
back-compat escape hatch, then is removed in a follow-up.

## Out of scope for phase 2b

- Multi-GPU `par_vae` (distributed VAE decode). 2b.1 targets single-GPU
  only. Multi-GPU VAE decode parity can be a post-phase-2b follow-up;
  in the meantime, multi-GPU support reduces to context-parallel
  attention on the transformer side, which flashdreams already wires.
- HunyuanVideo-1.5 8B variant (`hyvideo/generate.py` upstream). Phase 3
  per the integration README staging plan.
- Refactoring the lingbot / wan21 / self_forcing integrations to share
  the action / camera / memory conditioner modules. If those
  integrations ever need similar conditioners, the modules built here
  can be promoted from `hy_worldplay/` to a shared
  `flashdreams.recipes.wan.conditioners` module — but only on demand.

## Risks & mitigations

| risk | mitigation |
|---|---|
| 2b.1 native runner boots but produces visually broken video (wrong shape, wrong colour space, etc.) | The smoke test asserts mp4 shape + size only. Manual verification on `cat_surf.jpg` before commit. If output is broken, narrow to either preprocessing or pipeline output-shape audit before merging. |
| Pipeline output tensor shape doesn't match `[B, C, T, H, W]` expected by `export_to_video` | Run a one-liner shape audit during 2b.1 implementation (`print(pipeline.generate(0, cache).shape)`); add a `.permute(...)` if needed and document in the code. |
| Each conditioner's parity drift compounds and 2b.5's final parity check fails by a wide margin | The feature flag means each sub-PR can be parity-diffed against the wrapper independently. If 2b.3 lands action and parity drifts unexpectedly, we fix before merging 2b.4. |
| Subclass tree (`HyWorldPlayWanDiTNetwork` → `HyWorldPlayWanBlock` → custom action / PRoPE / memory hooks) becomes unwieldy | Keep each subclass single-purpose; prefer composition (a single `HyWorldPlayConditioners` payload struct flowed via `network_extra_kwargs`) over deep inheritance chains. Audit subclass diameter at 2b.4. |
| `sageattention` removal causes ULP-level drift that propagates and breaks parity bar at 2b.5 | Document the expected `sageattn` → native-attention delta in 2b.4. If it exceeds the parity bar, options are (a) widen the bar with documentation, (b) keep `sageattention` as an optional dep in 2b.4 only and drop it later, (c) port a flashdreams-native sage-like attention impl. |

## Success criteria

| sub-PR | success criterion |
|---|---|
| 2b.1 | `flashdreams-run hy-worldplay-wan-i2v-5b --use-native-pipeline` on `cat_surf.jpg` produces a valid `(81, H, W, 3)` mp4 on single GPU. CPU smoke test asserts feature-flag routing. CI passes. |
| 2b.2 | Scheduler swap in. Numeric parity diff vs vendor wrapper baseline narrows by the scheduler component (which the conditioner-less path lets us isolate). |
| 2b.3 | Action conditioning matches upstream's AdaLN modulation. Conditioner-only parity diff vs baseline narrows by the action contribution (target: visible motion semantics match — exact ε set when that PR opens). |
| 2b.4 | Camera-trajectory conditioning matches upstream's PRoPE attention. Conditioner-only parity diff narrows by the camera contribution (target: camera moves match the pose string — exact ε set when that PR opens). |
| 2b.5 | Memory module matches upstream's KV prefill. Full pipeline parity diff matches the documented phase-1 parity bar (mean per-frame uint8 RGB delta target: ≤ the phase-1 threshold from `tests/parity_check/README.md`). Parity sub-venv removed. Default flipped to native. |

## Open questions

(none blocking sub-PR 2b.1; tracked for follow-up sub-PRs)

- **Conditioner module location.** Do action / camera / memory modules
  live under `integrations/hy_worldplay/hy_worldplay/` (plugin-local)
  or `flashdreams/recipes/wan/conditioners/` (shared)? Default: plugin-
  local; promote on demand. Revisit at 2b.5.
- **Distilled vs non-distilled checkpoint.** Phase 1 ships the
  distilled 4-step checkpoint. Should we also expose the non-distilled
  50-step path through the same slug, or as a separate config?
  Default: defer to phase 3.
- **Whether to vendor `sageattention` permanently.** Phase 2b.4 plans
  to drop it. If parity drift is unacceptable, we revisit.
