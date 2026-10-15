---
title: 'Latency tuning'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Tune latency from measurements, not from model size or code shape. Interactive
latency includes model work, decode, transfer, queueing, and presentation; a
faster kernel cannot fix a backed-up presenter, and transport changes cannot
make a slow denoising loop faster.

## 1. Establish a reproducible baseline

Fix the model slug, checkpoint, input, seed, resolution, block count, precision,
compile flags, cache policy, and presentation mode. Record the commit, GPU,
driver, CUDA, PyTorch, cuDNN, and compiler-cache state.

Use `--stats-path` before the application separator. It enables synchronized
pipeline profiling and writes model-step metrics:

```bash
uv run --package flashdreams-self-forcing flashdreams-run-v2 \
    t2v-self-forcing-wan2.1-t2v-1.3b \
    --mode mp4 \
    --output-path artifacts/latency/baseline.mp4 \
    --stats-path artifacts/latency/baseline.json \
    --timeout unbound -- \
    --no-ui \
    --prompt "A fixed benchmark prompt." \
    --total-blocks 14 \
    --seed 1
```

Profiling synchronizes CUDA and changes throughput. Use it to attribute stages,
not as the headline uninstrumented result.

## 2. Separate cold and steady state

The first steps may include checkpoint loading, text/context encoding,
`torch.compile`, kernel autotuning, graph capture, and cache fill. Report at
least:

- application initialization or preload time;
- first visible chunk latency;
- warmup steps excluded;
- steady-state median and p90 chunk time;
- frames per chunk and effective generated FPS;
- peak and reserved GPU memory.

Run long enough to reach a full rolling cache. Use fresh processes when
comparing compile, autotune, or persistent-cache behavior.

## 3. Read the pipeline stages

The base pipeline reports `encode_ms`, `diffuse_ms`, `decode_ms`,
`finalize_ms`, `total_ms_wo_finalize`, and `total_ms` when those stages are
present. It also reports allocated, reserved, and peak GPU memory.

| Dominant observation | Investigate first |
| --- | --- |
| `encode_ms` | control preprocessing, VAE encoder, input layout/copies |
| `diffuse_ms` | denoising steps, attention/GEMM backend, compile, CUDA graphs, model cache |
| `decode_ms` | decoder choice, streaming cache, layout conversion, memory format |
| `finalize_ms` | K/V maintenance, synchronization, overlap opportunity |
| Pipeline fast but display late | device-to-host transfer, encoding, queue depth, pacing, network |
| Latency grows over time | unbounded cache/history, allocator growth, presenter backlog |

Measure transfer and presentation separately when they are not represented by
pipeline metrics. For interactive generation, also report estimated
input-to-visible latency rather than only model FPS.

## 4. Change one bottleneck at a time

Prefer the smallest option that addresses the measured hot stage:

- reduce resolution, denoising steps, or context only when the quality tradeoff
  is acceptable;
- bound and preallocate K/V caches before compiler or graph work;
- compile the smallest fixed-shape compute region;
- capture CUDA graphs only with stable shapes, pointers, and stream ordering;
- compare attention backends on the target GPU instead of assuming portability;
- keep the quality decoder as the reference when evaluating a faster decoder;
- tune ordered pacing and bounded queues after generation gets faster.

Keep optimized paths opt-in until their startup, reset, scene-switch, and
shape-change behavior is validated. Hardware-specific results belong on the
model page, not as universal defaults in this generic guide.

## 5. Validate performance and quality together

Use the same latent for decoder comparisons, compare cache changes before and
after the rolling-window boundary, and use controlled short schedules for
attention or compile changes. Preserve the fallback path.

A defensible result includes the exact command and environment, warmup policy,
median and p90 totals, stage timings, memory, output artifacts, quality method,
and whether the candidate is recommended, useful only as an opt-in, rejected,
or still unvalidated.

For repeatable command execution and reports, use the
[local benchmark guide](../../demo_api/guides/local_benchmarks.md). For low-level
attention and quantization options, continue with
[Accelerated building blocks](accelerated.md) only after profiling identifies
that stage as the bottleneck.
