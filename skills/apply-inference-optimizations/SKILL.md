---
name: apply-inference-optimizations
description: >
  Apply FlashDreams-style inference speedups to model integrations after a
  baseline exists: bounded windows and fixed K/V caches, cache/decode overlap,
  `torch.compile`, CUDA graph capture, attention backend checks, decoder layout
  or replacement, transfer/materialization changes, and ordered presentation
  tuning. Use when porting known optimizations into a runner, demo, serving
  adapter, or downstream integration while preserving quality and reset
  behavior.
---

# Apply inference optimizations

Use this skill after `profile-model-performance` has identified the dominant
stage. Keep every optimization opt-in until benchmark and quality evidence show
that it is safe for the target workflow.

## Ground rules

- Preserve the existing quality path as the baseline. Add faster paths as flags
  or config variants first.
- Change one optimization family at a time unless a combined profile is the
  explicit validation target.
- Add or extend the benchmark harness in the same change as the runtime flag.
- After applying an optimization, continue with `validate-performance-quality`
  before promoting it as a default or documenting a speedup claim.
- Print active settings and stage timing summaries so runs are auditable.
- Read `flashdreams-integrations` before moving code across `core`, `infra`,
  recipes, or integrations. Avoid model-specific branches in shared layers; add
  config slots or override hooks instead.

## Model and denoise path

Use these when the measured hot stage is the transformer, scheduler loop, or
model-side cache interaction.

- Prefer fixed shapes before compiler work: bounded history windows, static
  chunk sizes, preallocated buffers, and stable cache storage.
- Restrict `torch.compile` to the smallest fixed-shape callable that contains
  real compute. Keep cache mutation, dynamic setup, reset, and I/O outside the
  compiled region unless a probe proves the broader scope is worth it.
- Separate pre-saturation and post-saturation behavior. Dynamic cache fill can
  remain eager while the saturated steady-state call is compiled.
- Test compile modes and persistent compiler cache behavior in fresh processes.
  Record first-visible chunk cost, hidden prewarm cost, and steady-state gain.
- Add CUDA graph capture only around a region with stable shapes, stable device
  pointers, deterministic stream ordering, and explicit warmup. Start with the
  smallest useful graph before attempting whole-step capture.
- Profile actual SDPA kernels before forcing attention backends. If the default
  already dispatches to the desired backend, backend forcing is unlikely to
  help and may change numerics.

## Cache path

Use these when timing shows append/slice churn, K/V refresh, history-window
growth, or cache synchronization cost.

- Replace repeated concatenate/slice pruning with fixed-size block storage when
  the attention semantics allow it. Include sink tokens or pinned context
  regions only if the original model relied on them.
- Prove parity before and after the rolling boundary. Short pre-roll equality
  is not enough if the bug only appears once the window evicts old frames.
- Keep finalized history and current in-flight tokens equivalent to the baseline
  path; off-by-one cache windows create plausible but divergent rollouts.
- Overlap cache maintenance on a side stream only after the synchronous path is
  correct. Report both submit time and next-step wait time.
- Reset, scene switches, prompt changes, and shape changes must cancel or flush
  pending async cache work and rebuild state deliberately.

## Decode path

Use these when VAE or decoder time dominates the chunk budget.

- Keep the full-quality decoder as the reference path unless the user explicitly
  chooses a preview-quality mode.
- Build a decoder-only probe that decodes identical latents through reference
  and candidate paths. Do not use separate autoregressive rollouts as strict
  decoder-quality evidence.
- Try layout and memory-format changes before broad compiler changes when
  profiling shows copy/layout overhead.
- Compile only stateless or carefully isolated decoder submodules first. Whole
  streaming decoder compilation can silently corrupt cache state or alter
  numerics; reject it unless every output frame matches the reference within
  the accepted tolerance.
- Treat CUDA graph without compile as a correctness probe first. If it only
  saves a few milliseconds, record that and avoid live-path complexity.
- Lightweight decoders are preview candidates until same-latent metrics and
  visual artifacts show they are close enough to the quality decoder for the
  intended use.

## Transfer and presentation path

Use these when generated frames are ready faster than users see them, or when
CPU work dominates after decode.

- Delay GPU-to-CPU materialization until the consumer needs frames. Use lazy
  CUDA frame objects, pinned host prefetch, or batched copies when the local
  code already has those patterns.
- Measure CPU image/video encode before moving it to a thread or GPU codec.
- Preserve generated frame order for quality demos. Frame dropping can diagnose
  queue backlog, but it changes motion continuity and should not be the
  recommended quality path.
- Tune ordered pacing and bounded queues after throughput changes. Faster
  generation can feel slower if old frames accumulate in the presenter.
- Report best, average, and worst estimated input-to-visible latency when the
  workflow is interactive.

## Promotion criteria

Promote an optimization to the `Recommended` validation status only when all
apply:

- steady-state total chunk time or target latency improves materially;
- stage timing shows the expected bottleneck moved or shrank;
- quality validation passes against the right reference;
- startup, prewarm, reset, scene-switch, and shape-change behavior are
  acceptable;
- the fallback path remains available;
- docs state the command, caveats, whether the setting is enabled by default,
  and its `validate-performance-quality` status: `Recommended`,
  `Useful opt-in`, `Rejected`, or `Deferred`.

## Rejection notes

Record failed attempts with the same care as successful ones: exact command,
observed speed, quality result, failure mode, and what would have to change to
revisit the idea. This prevents future integrations from repeating unsafe
compiler, decoder, cache, or presentation shortcuts.
