<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lingbot camera-to-video

The FlashDreams v2 camera-to-video application for Lingbot World. This package
contains only the application boundary: it combines the shared
`flashdreams-cam2v` lifecycle and controls with the existing
`flashdreams-lingbot` model config. Its CLI input, example-data, intrinsics,
and world-scale resolution live in this package.

The application loads the pipeline once. Each session owns its autoregressive
cache, keyboard state, camera pose, and UI state.

## Run

```bash
uv sync --package flashdreams-cam2v-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

The command prints the browser URL. Use `W`/`S` to move, `A`/`D` or
`J`/`L` to yaw, `Q`/`E` to strafe, and `I`/`K` to pitch the generated
camera. The retained SlangPy overlay lists active controls, and the arrow keys
mirror `W`/`A`/`S`/`D`.

The UI/write path owns presentation pacing, and WebRTC sends each available
frame as soon as aiortc requests it rather than applying another pacer.
`window.write` synchronously converts the UI result and materializes owned
`VideoFrame` objects. SlangPy rendering/composition and the WebRTC sink use
separate high-priority CUDA streams joined by a readiness event. The sender
retains two queued, unsent frames in FIFO order and evicts the oldest queued
frame on overflow. A frame already dequeued for the sender or encoder is
committed and does not count against that capacity. Write MP4 when the output
must be frame-exact.

Model metrics retain warmup-excluded `steady_state_fps` and step wall time.
`model_step_wall_s` includes input preparation, generation, finalization, and
CUDA completion. One concise console line reports that synchronized wall time
and chunk FPS for every warmup and steady AR step. The interactive
specialization disables the pipeline's device-wide synchronous stage profiler
so model and presentation streams can overlap; use an explicit profiling run
when a GPU-stage breakdown is needed. The UI's recent model-rate value is the
wall-time-weighted throughput of AR steps whose completions fall in the trailing
two seconds. It excludes between-step pacing and presentation and reports no
recent output after two seconds without a completion.

For custom inputs, pass `--image-path` and `--intrinsic-path`. Also pass either
`--world-scale` directly or `--pose-path` so the application can infer the
translation normalizer. Input resolution and example-data downloads are owned
by this package and do not use the legacy Lingbot runtime/schema path.

Use `--warmup-blocks N` to change the five-block default warmup exclusion.

## WebRTC UI regression benchmark

The manual ABBA scenario runs no-UI/UI/UI/no-UI in fresh processes, waits
for a loopback aiortc receiver before generation, excludes five warmup blocks,
and measures twenty blocks per run:

```bash
uv run --project integrations_v2/cam2v_lingbot --no-sync \
  python -m tools.benchmarks.v2_webrtc_ab \
    --config configs/v2_webrtc_benchmarks.json \
    --benchmark cam2v-lingbot-hud-ab \
    --repo-root . \
    --output-dir artifacts/benchmarks/cam2v-lingbot-hud-ab
```

The output contains raw records for every run plus webrtc_ab.json and
webrtc_ab.md. Acceptance checks decoded and model FPS ratios, model-step
timing, presentation publish waits, decode cadence, ordered RTP timestamps, and
receiver drain. Raw and summarized artifacts also retain sender queue/drop and
frame-conservation counts; per-frame `window_write_s` captures synchronous
materialization cost without substituting the synthetic flush frame.
This is a server-to-aiortc-decoder benchmark; it does not measure browser DOM,
display-compositor, or network performance.

## Tests

```bash
uv run pytest integrations_v2/cam2v_lingbot -m ci_cpu -v
```
