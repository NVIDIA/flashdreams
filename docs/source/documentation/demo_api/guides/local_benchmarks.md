---
title: 'Local benchmarks'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams-benchmark` runs registered commands, captures their environment,
logs, metrics, and media, and writes an HTML report. It is a manual local
workflow; real-model scenarios require their normal GPU, checkpoint, asset,
and FFmpeg dependencies.

## Run the maintained v2 suite

```bash
uv sync --package flashdreams --group cuda13 --extra runners --inexact
uv run --no-sync flashdreams-benchmark --list-scenarios \
    --scenario-file configs/v2_model_benchmarks.json
uv run --no-sync flashdreams-benchmark \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-self-forcing-quality-10s \
    --output-dir artifacts/benchmarks/self-forcing
```

Use `--dry-run` first when reviewing a new or expensive scenario. Each run
directory contains the rendered command, manifest, environment metadata,
normalized metrics, logs, media, and `report.html`. Keep the complete directory
when comparing a candidate with a baseline.

## Define a scenario

Commands are argument arrays, not shell strings. Use `{output_dir}` for files
that belong to the benchmark artifact:

```json
{
  "schema_version": 1,
  "scenarios": [
    {
      "id": "my-app-smoke",
      "name": "My application smoke",
      "command": [
        "flashdreams-run-v2",
        "t2v-self-forcing-wan2.1-t2v-1.3b",
        "--mode", "mp4",
        "--output-path", "{output_dir}/clip.mp4",
        "--stats-path", "{output_dir}/stats.json",
        "--timeout", "unbound",
        "--",
        "--no-ui",
        "--prompt", "A cat surfing.",
        "--total-blocks", "8"
      ],
      "warmup_steps": 1,
      "requires_runtime_stats": true
    }
  ]
}
```

Load it with `--scenario-file` and select it with `--scenario`. Do not put
credentials in scenario files; pass them through the environment expected by
the model.

## Make comparisons defensible

- Keep input, seed, resolution, block count, checkpoint, precision, and runtime
  flags fixed.
- Record the commit, GPU, driver, CUDA, PyTorch, and compiler-cache state.
- Separate startup/compile time from steady-state measurements.
- Exclude declared warmup steps and report a distribution, not one best sample.
- Compare output quality when a cache, decoder, attention backend, precision,
  or compile change can affect numerics.

Use `--quality-baseline-dir` for the harness's non-gating MP4 comparison. Treat
long moving autoregressive clips as smoke tests rather than strict parity: small
numerical changes can alter later content. The
[latency tuning guide](../../inferencing_api/guides/latency_tuning.md) explains
how to interpret pipeline-stage timings.
