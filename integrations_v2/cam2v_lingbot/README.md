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

The command prints the browser URL. Use `W`/`S` to move, `A`/`D` to yaw,
`Q`/`E` to strafe, and `I`/`K` to pitch the generated camera.

The application logs warmup-excluded `steady_state_fps` and a per-block timing
breakdown. `model_step_wall_s` includes input preparation, generation,
finalization, and CUDA completion. The pipeline profiling log provides its GPU
stage breakdown.

For custom inputs, pass `--image-path` and `--intrinsic-path`. Also pass either
`--world-scale` directly or `--pose-path` so the application can infer the
translation normalizer. Input resolution and example-data downloads are owned
by this package and do not use the legacy Lingbot runtime/schema path.

Use `--log-every-blocks N` to reduce log frequency and `--warmup-blocks N` to
change the five-block default warmup exclusion.

## Tests

```bash
uv run pytest integrations_v2/cam2v_lingbot -m ci_cpu -v
```
