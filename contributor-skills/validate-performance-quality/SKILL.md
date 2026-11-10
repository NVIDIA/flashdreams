---
name: validate-performance-quality
description: Design benchmark, quality, and documentation validation for FlashDreams-style performance changes. Use when adding or updating sweep commands, profiler probes, decoder-quality comparisons, compile/cache probes, manual GPU validation, performance summaries, model cards, or README guidance after optimizing a model integration, demo, or serving path.
---

# Validate performance quality

Use this skill after `apply-inference-optimizations` changes a runtime path.
Performance changes are not complete until they have a reproducible benchmark,
the right quality reference, and documentation that explains defaults versus
validated opt-in paths.

## Benchmark contract

Every benchmark or summary should make these facts recoverable:

- exact command and commit;
- model, checkpoint, precision, input, prompt/control schedule, seed,
  resolution, chunk/window sizes, and all performance flags;
- GPU model, number of GPUs, driver, CUDA, PyTorch, cuDNN, and relevant compiler
  cache state;
- warmup policy, number of measured chunks, first-visible/startup timing, and
  steady-state timing;
- median and p90 total chunk time, stage timings, throughput/FPS, memory when
  available, and any warnings or fallback kernels.

Use fresh processes when measuring compile/autotune, persistent compiler cache,
attention backend selection, or startup behavior. Use a long enough run to
separate cache fill and steady state.

## Quality contract

Choose the reference that isolates the behavior being changed:

- Decoder changes: decode the same latent tensors through reference and
  candidate decoders.
- Cache changes: compare against the original cache path before and after the
  rolling-window boundary, including reset behavior.
- Model compile, CUDA graph, or attention backend changes: use short static or
  controlled schedules first, then motion-heavy smoke tests.
- Integration ports: compare against upstream or an existing FlashDreams
  baseline with matched inputs, weights, scheduler, seed, and decode path.
- Presentation changes: validate ordered frame continuity and queue latency;
  do not treat dropped-frame smoothness as quality equivalence.

Useful artifacts: per-candidate videos, side-by-side videos, amplified diff
videos, contact sheets, metrics JSON, Markdown summaries, logs, profiler traces,
and worst-frame samples. Useful metrics include PSNR, MAE, RMSE, sharpness,
high-frequency energy, temporal MAE, and LPIPS or domain-specific scores when
already available.

Long moving autoregressive rollouts are good smoke tests, but they are weak
strict metrics because speed or numerical drift can change the content being
compared. Prefer short static clips and same-latent comparisons for acceptance.

## Harness design

- Provide a CLI that can run a baseline and one or more candidates in a stable
  order, with labels derived from settings.
- Save raw per-step records as JSON and a compact Markdown summary for humans.
- Include stage timing fields, settings, artifact paths, and quality metrics in
  machine-readable output.
- Support warmup exclusion and optional comparison-video generation.
- Capture failed optional candidates without losing successful rows.
- Add CPU tests for label generation, argument validation, matrix construction,
  and summary parsing. Mark real generation, profiler, and quality-regression
  runs as `manual` or GPU-only according to repo convention.
- Keep benchmark outputs, checkpoints, traces, and generated videos out of git
  unless the repo explicitly tracks small reference artifacts.

## Acceptance table

Summarize decisions in a table or bullets with these statuses:

- **Recommended**: quality passed, speed or latency improved materially, startup
  and reset behavior are acceptable, and the fallback remains documented.
- **Useful opt-in**: good for a specific target, but has a clear tradeoff such
  as startup cost, latency, memory, quality, hardware dependence, or manual
  prewarm.
- **Rejected**: speed was too small, quality regressed, output diverged
  unacceptably, state/reset behavior was unsafe, or complexity outweighed gain.
- **Deferred**: promising but requires a larger architecture, serving, hardware,
  training, or validation effort.

## Documentation update

Update the docs that future agents and users will read:

- README or demo docs: current recommended command, required hardware, caveats,
  expected startup behavior, and known fallbacks.
- Performance summary: what worked, what is opt-in, what failed, headline
  numbers, quality evidence, and remaining bottleneck.
- Model card or benchmark page: methodology, stack-matched comparisons, artifact
  links, and hardware/software environment.
- Plan or learnings note: hypotheses tested, interpretation, and deferred work.

Do not overgeneralize single-hardware results. Write them as evidence from the
measured stack, not universal guarantees.

## If validation cannot be run

When the current host lacks GPU access, checkpoints, credentials, or time:

- add CPU-verifiable tests for command construction and metadata;
- write exact GPU/manual commands with expected artifact paths;
- mark the final answer and docs clearly as "not run here";
- avoid promoting defaults until the missing validation is actually complete.
