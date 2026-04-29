# Alpadreams KV-Drop Experiment Report

This document describes the design and implementation of the
**KV-drop** inference-time experiment for the alpadreams autoregressive
rollout. The goal is to test whether the same trained checkpoint can be
inferenced with a *sliding-window overlap* between consecutive AR steps,
where the trailing `kv_drop_t` latent frames per step are discarded
from the persistent KV cache and re-rolled / re-denoised at the head of
the next step's query window. Implementation is gated by a new
`kv_drop_t` field on `CosmosTransformerConfig`; setting it to `0`
exactly reproduces the existing rollout.

## Motivation

In the existing alpadreams rollout each AR step:

- denoises `len_t` latent frames per step (`len_t = 4` in the chunk4
  recipes),
- commits all `len_t` K/V tokens to the persistent KV cache,
- emits `len_t * temporal_compression` decoded frames per step,
- advances the AR pointer by `len_t * temporal_compression` decoded
  frames.

The KV-drop scheme tests an inference-only modification:

- still denoise `len_t` latent frames per step,
- but commit only `len_t - kv_drop_t` of them to the persistent KV
  cache and to the streaming decoder,
- emit `(len_t - kv_drop_t) * temporal_compression` decoded frames per
  step,
- advance the AR pointer by `(len_t - kv_drop_t) * temporal_compression`
  decoded frames; the next step's query window therefore *overlaps* the
  current step's tail by `kv_drop_t` latent frames.

The empirical question is whether the dropped tail being re-denoised
with one extra step of future context — at the head of the next AR
step's window — improves visual quality, costs quality, or stays
roughly neutral, given that the trained checkpoint never saw this
inference pattern.

## Scheme

For chunk4 (`len_t = 4`) with `kv_drop_t = 1`:

```text
latent t →   0  1  2  3  4  5  6  7  8  9 10 11
step 0      [▓  ▓  ▓  ░]
                      └─↻ re-denoised in step 1
step 1               [▓  ▓  ▓  ░]
                              └─↻ re-denoised in step 2
step 2                        [▓  ▓  ▓  ░]
                                       └─↻ re-denoised in step 3
step 3                                 [▓  ▓  ▓  ░]
```

`▓` = committed (KV cache + decoder + output), `░` = denoised then
discarded, `↻` = re-rolled with fresh noise as the head of the next
step's window.

Per step:

- DiT denoises `len_t = 4` latent positions (no compute change).
- KV cache writes `(len_t - kv_drop_t) * pH * pW = 3 * pH * pW` tokens.
- Streaming decoder consumes 3 latent frames, producing `3 * 4 = 12`
  RGB frames (or `1 + (3 - 1) * 4 = 9` at AR step 0 due to the
  first-decode trim).
- AR pointer advances by 9 (step 0) / 12 (steady-state) decoded frames.
- Consecutive HDMap input windows overlap by `kv_drop_t * 4 = 4`
  decoded frames.

## Key invariants

1. **Persistent cache window stays at `window_size_t` latent frames**
   (chunk4 = 8). The new path does NOT change the persistent locality
   budget; the trained checkpoint still sees at most 8 frames of
   committed history.
2. **Attention-visible window stays at `window_size_t + sink_size_t`
   latent positions per self-attention call**. Within a step the
   committed prefix attends to history + the transient X tail, but the
   *visible* sequence is capped by temporarily hiding the oldest
   `excess` cached tokens FROM THE ATTENTION VIEW ONLY. The persistent
   buffer is never modified by this hiding.
3. **`kv_drop_t = 0` is the legacy path bit-for-bit.** All new control
   flow degenerates: the commit prefix equals the full denoising
   window, no transient tail, no eviction-from-view, decoder receives
   the full clean latent.
4. **Within-step self-attention is preserved.** The X tail K/V is
   computed locally for the current `predict_flow` call and concatenated
   into the attention K/V (after capping). All `len_t` queries in the
   current step see history + full `len_t` current K/V, just as during
   training.
5. **Multiple `predict_flow` calls within one AR step (denoising loop +
   finalize-pass at `context_noise = 128`) overwrite the same commit
   slice.** The persistent cache state at the end of an AR step is the
   finalize-pass committed prefix; the transient tail from any pass is
   never persisted.
6. **RoPE positions remain consistent across overlap.** Step `t`'s
   queries land at absolute RoPE positions
   `[t * _pT_commit, t * _pT_commit + _pT)`, where
   `_pT_commit = (len_t - kv_drop_t) // patch_temporal`. Overlap
   positions get the same absolute RoPE index in both AR steps that
   touch them.

## Files modified

### Core attention and KV cache

- [`flashdreams/core/attention/kvcache.py`](flashdreams/flashdreams/core/attention/kvcache.py)
  - Refactored `BlockKVCache` from split filling/steady-state
    chunk-aligned writes into a unified roll-and-append path for the
    `sink_size == 0` case. The path supports non-divisible windows
    (`window_size` no longer needs to be a multiple of `chunk_size`):
    each new chunk evicts exactly
    `max(0, _n_cached + chunk_size - window_size)` of the oldest
    rolling tokens, then appends the chunk at the right edge. Repeated
    same-`chunk_idx` updates overwrite a tracked
    `_last_write_start / _last_write_end` slice instead of appending
    again.
  - The `sink_size > 0` path is preserved unchanged (still asserts
    `(window_size + sink_size) % chunk_size == 0`); a `FIXME` notes that
    extending the non-divisible roll-and-append path to non-zero sink is
    out of scope for this experiment.
  - `cached_k()` / `cached_v()` now use `min(_n_cached + chunk_size,
    total_size)` to compute the visible end, which collapses to the old
    formula when divisible.

### Self-attention

- [`flashdreams/recipes/alpadreams/transformer/impl/modules.py`](flashdreams/flashdreams/recipes/alpadreams/transformer/impl/modules.py)
  - Rewrote `SelfAttention.forward`: project the full `len_t` K/V
    locally (with RoPE), write only the leading `kv_cache.chunk_size`
    tokens to the persistent cache, then build the attention-visible
    K/V as `cat(cached_k_visible, k_trans)` where
    `cached_k_visible = cached_k()[..., excess:, :, :]` and
    `excess = max(0, cached_len + trans_len - (window_size + sink_size))`.
    The base `MultiHeadAttention.forward` and the `CrossAttention`
    path are untouched, so cross-attention (which projects K/V from a
    different feature dimension) keeps its original signature and
    behavior.

### Transformer + recipes

- [`flashdreams/recipes/alpadreams/transformer/__init__.py`](flashdreams/flashdreams/recipes/alpadreams/transformer/__init__.py)
  - Added `kv_drop_t: int = 1` to `CosmosTransformerConfig` with
    assertions (`0 <= kv_drop_t < len_t` and
    `(len_t - kv_drop_t) % patch_temporal == 0`) and derived helpers
    `_len_t_commit = len_t - kv_drop_t` (pre-patchify) and
    `_pT_commit = _len_t_commit // patch_temporal` (post-patchify).
  - `CosmosTransformer.__init__` now raises `NotImplementedError` when
    `kv_drop_t > 0` is combined with T-axis context parallelism
    (`cp_groups.T_size > 1`) or `compile_network=True`, because both
    cases need design work that is out of scope for the first cut.
  - `CosmosTransformer.initialize_autoregressive_cache` now passes
    `chunk_size = num_tokens_per_view_per_step * cfg._pT_commit`
    (instead of `* cfg._pT`) to `network.initialize_cache`. The
    persistent cache therefore advances by the committed temporal
    stride.
  - `CosmosTransformer.predict_flow` shifts RoPE by
    `ar_idx * cfg._pT_commit` (instead of `ar_idx * cfg._pT`).

- [`flashdreams/recipes/alpadreams/config.py`](flashdreams/flashdreams/recipes/alpadreams/config.py)
  - `_transformer_config(...)` accepts a `kv_drop_t: int = 1` kwarg and
    forwards it to `CosmosTransformerConfig`.
  - chunk4 builders
    (`build_sv_2steps_chunk4_loc8_pshuffle_lighttae`,
    `build_mv_2steps_chunk4_loc8_pshuffle_lighttae`) default
    `kv_drop_t=1` (the experiment).
  - chunk2 / chunk3 builders default `kv_drop_t=0` because they use a
    stateful Wan VAE HDMap encoder that is not yet supported with
    overlap (see "First-cut limitations" below). Existing tests against
    these recipes therefore continue to exercise the legacy path.

### Pipeline

- [`flashdreams/infra/pipeline/base.py`](flashdreams/flashdreams/infra/pipeline/base.py)
  - Added `_pre_decode_hook(clean_latent, autoregressive_index)` to
    `StreamInferencePipeline.generate`, called between the diffusion
    model and the decoder. Default returns `clean_latent` unchanged.
    The `FinalState` stashed on the pipeline cache holds the unsliced
    clean latent, so subclasses can return a sliced view without
    affecting the AR cache or the finalize path.

- [`flashdreams/recipes/alpadreams/pipeline.py`](flashdreams/flashdreams/recipes/alpadreams/pipeline.py)
  - Caches `_len_t_latent`, `_len_t_commit`, and `_kv_drop_t` from the
    transformer config.
  - Raises `NotImplementedError` in `__init__` when `kv_drop_t > 0` is
    combined with a stateful `WanVAEEncoder` HDMap encoder.
  - Splits the per-step frame-count API into:
    - `get_num_input_frames(i)` (HDMap input window, sized by `len_t`):
      - step 0: `1 + (len_t - 1) * temporal_compression`
      - steady: `len_t * temporal_compression`
    - `get_num_output_frames(i)` (decoded frames + AR pointer advance,
      sized by `_len_t_commit`):
      - step 0: `1 + (_len_t_commit - 1) * temporal_compression`
      - steady: `_len_t_commit * temporal_compression`
    - `get_num_frames(i)` is kept as a back-compat alias for
      `get_num_output_frames(i)`.
  - Overrides `_pre_decode_hook` to slice
    `clean_latent[:, :, : self._len_t_commit]` before passing to the
    streaming decoder, so the decoder cache (TAEHV / Wan VAE) advances
    by the committed prefix only. Skipped when
    `_len_t_commit == _len_t_latent`.

### Example runner

- [`flashdreams/examples/run_alpadreams.py`](flashdreams/examples/run_alpadreams.py)
  - Added `--kv_drop_t` CLI flag. When set, it is passed straight to
    the recipe builder (rather than mutating the config after build),
    so other knobs that depend on `kv_drop_t` are picked up
    consistently.
  - The AR loop now uses `get_num_input_frames(i)` to slice the HDMap
    input window and `get_num_output_frames(i)` to advance the AR
    pointer, so consecutive HDMap windows overlap by
    `kv_drop_t * temporal_compression` decoded frames when
    `kv_drop_t > 0`.

### gRPC server (production)

- [`integrations/alpadreams/.../video_model_flashdreams_pipeline.py`](integrations/alpadreams/alpadreams/conditioning/video_model_flashdreams_pipeline.py)
  - Builds `CosmosTransformerConfig` directly. Pinned `kv_drop_t=0`
    explicitly so the production server keeps the legacy
    non-overlapping rollout regardless of the new field's default; this
    avoids quietly turning on the experiment for production traffic.

## First-cut limitations

When `kv_drop_t > 0`, the implementation explicitly raises
`NotImplementedError` in any of the following situations:

1. **Wan VAE HDMap encoder.** The chunk2 / chunk3 recipes use the
   `WanVAEEncoder`, whose `WanVAECache` advances by however many input
   pixel frames it sees. Overlapping HDMap input windows would push the
   encoder state past where the AR pointer actually is. PixelShuffle
   (chunk4) is stateless w.r.t. temporal context (`last_frame` mode
   re-indexes from each local input window) and is safe.
2. **T-axis context parallelism.** With `cp_groups.T_size > 1`, each
   rank's local prefix slice does not equal the global temporal
   prefix; ranks owning the dropped tail would mis-commit their tokens.
   HW-only CP is also gated for now until tested.
3. **`torch.compile`.** Non-divisible rolling windows, transient tails,
   and dynamic `excess` values introduce dynamic shapes that have not
   been validated under `torch.compile`. Run with `--no_compile`.

The plan calls these out as `FIXME`s in the relevant code paths.
Lifting them is future work and is independent of the
`kv_drop_t = 1` experiment itself.

## How to run

Two runs on the same prompt / HDMap / seed make the comparison clean:

Baseline (legacy non-overlapping rollout):

```bash
torchrun --nproc_per_node=1 flashdreams/examples/run_alpadreams.py \
    --n_cameras 1 --total_blocks 60 \
    --overwrite_config_name sv_2steps_chunk4_loc8_pshuffle_lighttae \
    --no_compile --kv_drop_t 0
```

Experiment (`kv_drop_t = 1` overlap):

```bash
torchrun --nproc_per_node=1 flashdreams/examples/run_alpadreams.py \
    --n_cameras 1 --total_blocks 60 \
    --overwrite_config_name sv_2steps_chunk4_loc8_pshuffle_lighttae \
    --no_compile --kv_drop_t 1
```

The 4-view config is also supported:
`mv_2steps_chunk4_loc8_pshuffle_lighttae`.

## Expected behavior to verify

- **Bit-for-bit baseline at `--kv_drop_t 0`.** With everything else
  held constant, the experiment knob `0` should produce the same
  rollout as `HEAD` (chunk size, RoPE offsets, encoder window, decoder
  window, output frame count all unchanged).
- **Frame counts at `--kv_drop_t 1` (chunk4).** Step 0 emits 9 RGB
  frames (vs 13 in the baseline), steady-state steps emit 12 RGB
  frames (vs 16). Total emitted frames per `total_blocks` is therefore
  smaller under `kv_drop_t = 1`.
- **Persistent KV cache cap.** After steady state, each view's
  persistent KV cache holds at most `window_size_t = 8` committed
  latent frames worth of K/V (same as baseline).
- **Attention-visible K/V cap.** Each self-attention call sees at most
  `window_size_t + sink_size_t = 8` latent positions, by temporarily
  hiding the oldest cached tokens when adding the transient tail. The
  persistent cache itself is not modified by the hiding.
- **Transient tail does not persist.** After an AR step finishes, the
  cache reflects only the finalize-pass's committed prefix; the X
  tokens at the tail are absent from `cache.cached_k()`.
- **Quality (open empirical question).** The trained checkpoint never
  saw this inference pattern. Three plausible outcomes:
  1. Mostly fine, with a small quality dip;
  2. A noticeable seam at the step-0 / step-1 boundary because step 1
     re-denoises latent index 3 (= source frame 12) from fresh noise
     against a length-3 cache rather than the length-4 cache that
     training expected;
  3. Degradation that snowballs as the AR step count grows.
- **Run-time cost.** DiT compute is unchanged (still denoises `len_t`
  positions per step), but extra encoder / decoder work is wasted on
  the dropped tail, so wall-time is roughly the same. The benefit is
  purely on the conditioning pattern at commit time, not on throughput.

## Pointers

- Plan / design notes: `.cursor/plans/visualize-ar-overlap-scheme_*.plan.md`
  (kept for the design discussion; not required to use the
  implementation).
- Concrete numbers used in this report come from the chunk4 recipe
  (`len_t = 4`, `pH * pW = 45 * 80`, `window_size_t = 8`,
  `temporal_compression = 4`, `patch_temporal = 1`).

## Command to run the experiment

Experiment command:
```bash
torchrun --nproc_per_node=1 examples/run_alpadreams.py \
    --n_cameras 1 --total_blocks 60 \
    --overwrite_config_name sv_2steps_chunk4_loc8_pshuffle_lighttae \
    --no_compile --kv_drop_t 1
```

Baseline command:
```bash
torchrun --nproc_per_node=1 examples/run_alpadreams.py \
    --n_cameras 1 --total_blocks 60 \
    --overwrite_config_name sv_2steps_chunk4_loc8_pshuffle_lighttae \
    --no_compile --kv_drop_t 0
```
