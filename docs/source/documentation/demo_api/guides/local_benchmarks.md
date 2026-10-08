---
title: 'Local benchmarks'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams-benchmark` runs registered commands, collects logs and metrics,
and writes an HTML report. It is a manual, local GPU workflow; it does not gate
CI.

## Use the maintained v2 scenario suite

The checked-in suite exercises the current `flashdreams-run-v2` application
slugs. Install the harness, list the scenarios, and select the ones to run:

```bash

uv sync --package flashdreams --group cuda13 --extra runners --inexact
uv run --no-sync flashdreams-benchmark --list-scenarios \
    --scenario-file configs/v2_model_benchmarks.json
uv run --no-sync flashdreams-benchmark \
    --scenario-file configs/v2_model_benchmarks.json \
    --scenario t2v-self-forcing-quality-10s \
    --output-dir artifacts/benchmarks/self-forcing

```

Scenario commands still own their runtime requirements. Model runs require an
NVIDIA GPU, host FFmpeg, access to their checkpoints, and any input assets the
application normally resolves.

Each run directory contains a manifest, environment metadata, normalized
metrics, command logs, generated media, and `report.html`. Keep the entire
directory when comparing a later candidate with a baseline.

## Define a local scenario

Use a JSON scenario file for a command that is not in a checked-in suite.
Commands are argument arrays, not shell strings:

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
        "--output-path",
        "{output_dir}/clip.mp4",
        "--timeout",
        "unbound",
        "--",
        "--prompt",
        "A cat surfing.",
        "--total-blocks",
        "8"
      ],
      "warmup_steps": 1
    }
  ]
}

```

Run it with `--scenario-file` and `--scenario` exactly as for a checked-in
suite.
