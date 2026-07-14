---
name: profile-model-performance
description: >
  Inspect and baseline performance for FlashDreams-style model integrations and
  interactive demos: map the generation path, add trustworthy timing splits,
  build focused probes, and identify whether decode, model/denoise, cache, data
  transfer, or presentation dominates. Use when starting performance work on an
  existing model runner, demo, serving path, or downstream integration before
  implementing speedups. Pair with `apply-inference-optimizations` after the
  bottleneck is known and `validate-performance-quality` for benchmark and
  quality gates.
---

# Profile model performance

Use this skill before changing runtime behavior. The goal is to produce a
defensible bottleneck map and a short list of candidate optimizations, not to
guess from code shape alone.

## Workflow

1. **Scope the executed path.**
   - Find the user-facing entry point: runner, CLI, interactive server, batch
     script, notebook, or downstream adapter.
   - Trace one generation step through input preparation, encode/context setup,
     model or denoise loop, cache update/finalize, decode, transfer, encode, and
     presentation.
   - Read `flashdreams-integrations` before changing framework boundaries or
     config contracts. Keep this skill focused on measurement and diagnosis.
   - Prefer no-GPU inspection first: config resolution, `--help`,
     `--no-instantiate`, static runner wiring, and small CPU tests.

2. **Establish a reproducible baseline.**
   - Use fixed input, seed, prompt/control schedule, resolution, chunk/window
     settings, checkpoint source, and device.
   - Make the run long enough to separate cache fill, compile/autotune, and
     steady-state chunks.
   - Record exact command, commit, GPU, driver, CUDA, PyTorch, cuDNN, dtype,
     compile cache state, and checkpoint identifiers.
   - Track startup/prewarm wall time separately from visible or steady-state
     timing. Do not average cold compile or cache-fill chunks into the headline
     steady-state metric.

3. **Add timing boundaries that respect CUDA asynchrony.**
   - Use CUDA events or explicit synchronization between major stages when
     attributing GPU time.
   - Report median and p90 after warmup for total chunk time and stage timings.
   - Useful stage names: input/encode, context setup, denoise/model, cache
     update submit, cache update wait, decode, GPU-to-CPU transfer,
     frame/materialization, image/video encode, queue wait, present pacing, and
     end-to-end chunk time.
   - Print the active runtime settings in summaries so logs cannot be detached
     from the flags that produced them.

4. **Classify the bottleneck.**
   - Model/denoise: attention, GEMM, normalization, scheduler loop overhead,
     dynamic shapes, SDPA backend selection, `torch.compile`, CUDA graph
     capture, or copy/layout inside the model step.
   - Cache: append/slice churn, rolling-window materialization, K/V refresh
     cost, cache update synchronization, reset/scene-switch rebuild behavior, or
     stale state after async work.
   - Decode: VAE/decoder wall time, streaming decoder cache, layout
     conversions, convolution/elementwise hot blocks, lightweight decoder
     quality tradeoffs, or unsafe whole-decoder compilation.
   - Transfer and presentation: GPU-to-host copies, CPU image/JPEG encoding,
     browser/server queue backlog, rate limiting, frame pacing, or display
     latency.
   - Multi-GPU/serving: context parallel shape boundaries, distributed cache
     state, device-to-device transfers, per-rank persistence, and scheduler or
     presenter behavior outside a single-process demo.

5. **Build the narrowest useful probe.**
   - Sweep one axis at a time when possible: window size, cache mode, compile
     mode, graph mode, decoder choice, decoder layout, presentation queue, or
     attention backend.
   - Use fresh processes for compile/cache studies so startup behavior and
     persistent compiler cache effects are visible.
   - Use decoder-only same-latent probes for decoder changes so stochastic model
     drift cannot explain quality differences.
   - Profile only after a sweep identifies the hot stage. Treat profiler wall
     time as perturbed attribution evidence, not the headline benchmark.

6. **End with a short diagnosis note.**
   - State the current bottleneck, the baseline numbers, the commands used, and
     the next optimization candidates.
   - Separate proven facts from hypotheses. If evidence is missing because GPU
     validation was not run, say so and provide the exact command to run later.

## Common pitfalls

- Do not infer the active attention or decoder backend from Python control flow;
  confirm with profiler kernels or explicit runtime logging.
- Do not compare moving autoregressive rollouts as strict quality metrics when
  different speeds or kernels can shift camera position or content. Use them as
  smoke tests.
- Do not treat a fast lightweight decoder as a quality replacement without a
  same-latent comparison against the quality decoder.
- Do not promote a startup-heavy compile path unless prewarm, persistent cache,
  reset, and scene-switch behavior are acceptable for the target workflow.
- Do not optimize presentation by dropping generated frames for quality demos;
  diagnose backlog separately, then tune ordered pacing and backpressure.

## Deliverable

A good profiling pass leaves behind:

- a reproducible baseline command;
- trustworthy stage timings with warmup excluded;
- quality/reference artifacts when behavior may change;
- a ranked bottleneck list;
- candidate optimizations with the validation each one would require.
