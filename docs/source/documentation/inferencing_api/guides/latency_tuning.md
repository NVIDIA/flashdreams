---
title: 'OmniDreams interactive latency tuning'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Interactive launch latency has two different components:

- **Model / chunk latency** is the time spent preparing HDMap conditioning,
  running the OmniDreams DiT, decoding the generated chunk, and updating model
  state. This is usually the dominant input-to-visual delay.
- **Video-transport latency** is the time spent delivering already-generated
  frames to a local window or browser. The presentation mode and network path
  affect this cost, but they do not make the model generate a chunk faster.

Tune the model path first when the profiler shows per-chunk work dominating.
Tune transport when generated frames are ready quickly but arrive late or unevenly
in the viewer.

## Model and backend choice

Use the OmniDreams world-model backend for latency work. The raster backend is
useful for scene, control, and presenter debugging, but it does not exercise the
model path and should not be used as a model-latency reference.

The registered Interactive Drive applications are the supported starting points:

- `interactive-drive-omnidreams` is the default single-view configuration at
  `1280 x 704` (width x height), 30 FPS, 8 generated frames per steady-state
  block, LightVAE enabled, and native DiT acceleration disabled.
- `interactive-drive-omnidreams-optimized-gb300` and
  `interactive-drive-omnidreams-optimized-rtx-pro-6000` select
  hardware-specific optimized PyTorch attention policies at `1280 x 704`.
- `interactive-drive-omnidreams-perf` is the perf-tuned configuration. It lowers the
  default resolution to `1168 x 640`, keeps 30 FPS and 8-frame steady-state
  blocks, enables the performance recipe, and requires the native DiT path.
- `interactive-drive-omnidreams-fast-perf` adds the native FP8 LightVAE
  encoders to the perf configuration.

Run the perf application only on hosts that can build and load the native extension:

```bash

uv run --package flashdreams-omnidreams flashdreams-run-v2 \
    interactive-drive-omnidreams-perf --mode native-window

```

The perf config's `native_dit_acceleration="required"` is intentional. If the
native extension is not available, startup fails instead of silently falling
back to the slower PyTorch path.

## Resolution

Resolution is one of the highest-impact latency knobs because it changes the
amount of HDMap, DiT, and VAE work per chunk. Pass application arguments after
the `--` separator:

```bash

uv run --package flashdreams-omnidreams flashdreams-run-v2 \
    interactive-drive-omnidreams --mode native-window -- \
    --width 1168 --height 640

```

`--width` and `--height` must be positive. The OmniDreams pipeline also
requires both dimensions to be divisible by the VAE spatial compression ratio
times the DiT patch size (16 for the registered single-view configurations).
The registered defaults, `1280 x 704` and `1168 x 640`, satisfy this
constraint.

Lowering resolution reduces per-chunk compute and transport payload size, with
the expected image-quality tradeoff. The application uses the selected
resolution for both HDMap rasterization and world-model output.

## Chunk size constraints

Do not treat chunk size as an arbitrary latency knob. The registered
Interactive Drive applications do not expose a chunk-size command-line option,
and the world-model adapter validates the pipeline at startup:

- The initial conditioning chunk is fixed at 5 frames.
- The registered single-view configurations generate 8-frame steady-state
  chunks.

At 30 FPS, an 8-frame steady-state chunk covers about 267 ms of generated video.
Reducing video-transport latency cannot remove this model-side chunk granularity.

## FP8 and native acceleration

The perf application config uses the OmniDreams single-view native CUDA
extension for the DiT path. These values are properties of the registered
pipeline preset, not application manifest fields:

```yaml

native_dit_acceleration: required
native_dit_backend: fp8_kvcache_cudnn
native_dit_attention_backend: cudnn

```

The implementation supports `disabled`, `auto`, and `required` native
acceleration policies and `fp8_kvcache_cudnn` and `bf16` native DiT
backends. Its attention selector accepts `auto`, `cudnn`, `sparge`,
`sage3`, `sage3_fp8`, and `prefer_sage3_fp8`; the perf application pins
`cudnn`.

The native extension requires a source checkout, `git`, network access, a
CUDA toolchain (`nvcc`) matching the PyTorch build, and a Blackwell-class GPU
(SM 12.0). It downloads pinned third-party sources when first used and builds
for `12.0a` by default. The automatic architecture detection recognizes the
SM 12.0 RTX PRO 6000 and RTX 5090 device families.

H100 / Hopper systems should use the standard PyTorch CUDA path with native DiT
disabled unless you are deliberately maintaining a compatible native build. That
path is supported, but it is not the same perf path as the published GB300
numbers.

GB300 systems should use `interactive-drive-omnidreams-optimized-gb300`;
GB300 is not an SM 12.0 target for the native extension.

The `interactive-drive-omnidreams-fast-perf` application enables native FP8
for both the first-frame and per-step LightVAE encoders. It automatically
exports and caches calibration state at
`artifacts/native_vae/lightvae_fp8_state.pt` when no explicit state is
configured. Custom pipeline configurations can instead set
`native_vae_fp8_state_path` or the
`OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` environment variable.

The regular perf application leaves native VAE acceleration disabled.

## Transport choice

Pick transport based on where the viewer runs:

- `--mode native-window`: local GPU-backed presentation when the host has a
  graphics-capable GPU and display stack.
- `--mode webrtc`: browser presentation. Add `--host 0.0.0.0` and a fixed
  `--port` when connecting from another host.
- `--mode mp4 --output-path FILE`: file output for offline inspection; it is
  not an interactive transport.

Presentation choice affects delivery after a frame exists. If the model is
still spending most of the time inside each chunk, use the perf application
and resolution knobs first.

## Profiling and validated reference

Pass `--stats-path FILE.json` before the application-argument separator to
record model-step measurements. This also enables synchronized per-stage
pipeline profiling, so use it for measurement rather than as a throughput
setting:

```bash

uv run --package flashdreams-omnidreams flashdreams-run-v2 \
    interactive-drive-omnidreams-perf --mode native-window \
    --stats-path artifacts/interactive-drive-stats.json

```

The validated published reference is the single-view GB300 table in
[OmniDreams](../../../models/omnidreams.md), measured at `1280 x 704` (width x height). Its
KV-cache update measurement is off the hot path and excluded from the reported
total. This guide consolidates the supported latency controls; it does not add
new end-to-end hardware measurements.
