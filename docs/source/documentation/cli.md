---
title: 'CLI Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

## Command Shape To Run A Demo/Model

This section will only cover `<System Arguments>`. Refer to the [Running a model](../models/index.md#running-a-model) section for more information on `<Demo Preset>` and `<Demo Arguments>`:
```bash
uv run flashdreams-run-v2 <Demo Preset> <System Arguments> -- <Demo Arguments>
```

## System Arguments

### General

| Argument | Description |
| --- | --- |
| `--timeout {SECONDS,unbound}` | Bound the whole application run, including initialization and replacement sessions. |
| `--total-model-steps {N,unbound}` | Bound model steps across the whole application run, including replacement sessions. |
| `--mode {mp4,webrtc,native-window}` | Select file, browser, or local-window presentation. Default: `mp4`. |
| `--stats-path PATH` | Write JSON model-step measurements and enable synchronized per-stage pipeline profiling. |

### MP4 Specific

| Argument | Description |
| --- | --- |
| `--output-path PATH` | MP4 destination. Required in MP4 mode. |

### WebRTC Specific

| Argument | Description |
| --- | --- |
| `--host HOST` | Interface to serve on. Default: `127.0.0.1`; use `0.0.0.0` for remote clients. |
| `--port PORT` | Port to serve on. Default: `0`, which selects an available port. |

### Native Window Specific

| Argument | Description |
| --- | --- |
| `--window-title TITLE` | Native window title. Default: `FlashDreams`. |

### Default Demo Session Overrides

These options override the default demo session values requested by a particular demo implementation. Omit them to use the application's default values.

| Argument | Description |
| --- | --- |
| `--pixel-width N` | Override generated-frame width. |
| `--pixel-height N` | Override generated-frame height. |
| `--fps N` | Override the generated-frame playback rate. |
| `--layout {tchw,btchw,bcthw,bvtchw}` | Override generated tensor layout. |
| `--backpressure-mode {block,drop_oldest}` | Choose how the model thread handles a full presentation queue. |
| `--presentation-mode {on_demand,continuous}` | Choose when the presentation loop renders. |

#### Backpressure Mode

- `block` waits for queue capacity so generated chunks are retained.

- `drop_oldest` discards the oldest queued chunk to favor recent output.

#### Presentation Mode

- `on_demand` presents each selected model frame when it arrives.

- `continuous` keeps the UI responsive between model frames by reusing the newest frame.

### Examples

Stream to a browser:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode webrtc --host 0.0.0.0 --port 8089
```

Write every generated frame in order to an MP4:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode mp4 --output-path output.mp4 \
  --timeout unbound --backpressure-mode block --presentation-mode on_demand
```

MP4 mode requires `--timeout` and/or `--total-model-steps`; pass `unbound` to
name an explicit unlimited policy. WebRTC and native-window modes may omit both
limits. When a session is replaced, it receives the time and model steps left
for the whole run.

Preload and validate a v2 application without opening a window:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --preload-application -- --example-data
```

Preload runs `IApplication.init` and one model block. It defaults to warning;
set `FLASHDREAMS_PREPARATION_POLICY` to `none`, `warn`, or `error`. Findings
are written to `preparation_issues.txt` in the FlashDreams cache, or to
`FLASHDREAMS_PREPARATION_ISSUES_PATH` when set. Use
`--skip-preload-validation` to skip the first-block validation.

Open a local native window:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode native-window --window-title FlashDreams
```

## Legacy runner commands

`flashdreams-run` remains available for external `RunnerConfig` plugins through
`flashdreams.runner_configs` or `FLASHDREAMS_RUNNER_CONFIGS`. Current workspace
packages register applications through `flashdreams.applications_v2`; none
register v1 runners or the retired `flashdreams.applications` API. An empty v1
registry therefore has no model commands to list. Use
`uv run flashdreams-run-v2 --help` for current applications.

The old `flashdreams.demo`, `flashdreams.runtime.demo`, and
`flashdreams.serving.webrtc` packages have been removed. These remaining command
callers need migration:

| Caller | Migration |
| --- | --- |
| LingBot scenarios in `configs/deterministic_quality_benchmarks.json` and `configs/one_minute_demo_benchmarks.json` | Use `cam2v-lingbot-world-fast-taehv-window15-sink3`. Move example inputs, block count, and seed after `--`; keep frame dimensions, FPS, MP4 path, and stats path before it. |
| OmniDreams scenarios in those files, and built-in `omnidreams-sv-example-smoke` | `interactive-drive-omnidreams` is the current application, but it does not expose the old recorded-example UUID/replay arguments. Recreating the same conditioning and quality baseline needs a separate migration. |
| HY-WorldPlay `tests/parity_check/bench.sh` and `bench_batch.sh` | `cam2v-hy-worldplay` accepts `--pose-path` and `--total-blocks`, but does not expose the old `--pose`, `--num-chunk`, or `--ckpt-path` flags. Reproduce the camera trajectory and checkpoint/noise settings before using it for parity comparisons. |
| FlashVSR `scripts/benchmark_postprocess_multigpu.sh` | The script expects v1 runner flags, post-processing, and output filenames. The current Wan T2V application has no post-processing CLI option; changing the executable alone does not preserve this benchmark. |

Some scenarios call `tools.benchmarks.strict_run`, whose default entry point is
v1. A migrated scenario must explicitly pass `--entrypoint flashdreams-run-v2`
before its first `--`. The maintained
`configs/v2_model_benchmarks.json` suite already does this.

V2 takes presentation options before `--` and application options after it:

```bash
uv run --project integrations_v2/lingbot flashdreams-run-v2 \
  cam2v-lingbot-world-fast-taehv-window15-sink3 \
  --mode mp4 --output-path artifacts/lingbot.mp4 \
  --stats-path artifacts/lingbot-stats.json --timeout unbound \
  --pixel-height 464 --pixel-width 832 --fps 16 -- \
  --example-data --example-idx 0 --total-blocks 14 --seed 1 --no-ui
```

The old positional `local-window` mode becomes `--mode native-window`.
`--output.path` becomes `--output-path`, and `--output.stats-path` becomes
`--stats-path`. There is no general v2 replacement for nested runner overrides
or launch manifests. Inspect a model's supported arguments with
`flashdreams-run-v2 SLUG -- --help`. This prints help before loading the model;
`--preload-application` initializes the application and runs a model block.

## See also

- [Quickstart](../quickstart/index.md)
- [Demo configuration](demo_api/guides/configuration.md)
- [Application slugs](demo_api/guides/application_slugs.md)
- [Demo API](demo_api/index.md)
- [Offline Program Packager](tools/offline_program_packager.md)
