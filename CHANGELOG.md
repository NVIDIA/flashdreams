<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

Notable changes for application developers and model integrators are recorded here.
This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [v0.2.0] - 2026-10-21

Changes since [v0.1.0].

### Added

- [Cosmos3 Nano Sim Auto](integrations_v2/cosmos3_nano_sim_auto/README.md):
  interactive and batch driving with selectable RGB cameras, optional joint
  LiDAR generation, and automatically cached model weights and prepared
  residential, rain, and snow scenes.
- [Cosmos3 Nano Sim Bimanual](integrations_v2/cosmos3_nano_sim_bimanual/README.md)
  and HandDreams: action-conditioned robotics video using the public GA checkpoint,
  Quest/Isaac Teleop or relayed controller input, and controller-free recorded-action replay.
- A [v2 Demo API](docs/source/documentation/demo_api/index.md) with application/session
  lifecycle management, separate model and UI loops, and WebRTC, native-window,
  and MP4 output. Custom presentation loops, multiview conditioning, and sensor
  displays are reusable; driving adapters can supply cameras, rendering backends,
  and scene loaders.
- [Accelerated inference components](docs/source/documentation/inferencing_api/guides/accelerated.md)
  for quantization, streaming attention, and bounded KV caches. Tensor/context
  parallel runtime support is available to compatible model adapters.
- Waypoint 1.5 1B, SANA-WM, and SwiftVR integrations; LingBot-World v2 support.
- [Profiling](docs/source/documentation/demo_api/guides/profiling.md) with `IProfiler`,
  NVTX ranges, stage timings, frame rates, and input latency; headless model benchmarks.
- [Offline executable packaging](tools/package-as-offline-exe/README.md) for installed
  v2 applications on Windows and Linux, with asset preloading and preparation validation.
- Crazy Robotaxi, a standalone application with configurable controls, persistent
  settings, live prompt editing, free-roam mode, and a 32 GB RTX 5090 preset.

### Changed

- Migrated demos and model adapters use `apps/` and `integrations_v2/`.
  Launch them with `flashdreams-run-v2`; put runtime options before `--` and
  application options after it. Custom model/UI loop steps return `list[StepResult]`.
  See the [application guide](docs/source/documentation/demo_api/guides/create_demo.md).
- V2 MP4 runs require an explicit `--timeout` and/or `--total-model-steps`.
  Use `--timeout unbound` when the application supplies its own stopping condition.
- Added CUDA 12/13 installation groups (`cuda13` by default; `cuda12` on Linux)
  and deferred optional encoder imports until the encoder is used.

### Removed

- HY-WorldPlay integration and legacy OmniDreams evaluator entry points.
  Use the v2 OmniDreams applications for driving and generation.

### Fixed

- WebRTC latest-frame delivery and control-input latency; device-aware
  postprocessing, including multi-device Interactive Drive pipelines.
- Cosmos3 driving prompt conditioning, short control inputs during inference,
  and braking after focus loss. The demo now warns when driving leaves prepared map coverage.
- Wan 2.2 VAE channel ordering, Wan streaming decoder memory layouts, animated
  AVIF video-input selection, and decoding beyond the requested frame window.

### Security

- PyTorch checkpoint loading now uses `weights_only=True` for local files,
  S3 downloads, and merged distributed-checkpoint caches.

[Unreleased]: https://github.com/NVIDIA/flashdreams/compare/v0.2.0...main
[v0.2.0]: https://github.com/NVIDIA/flashdreams/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/NVIDIA/flashdreams/releases/tag/v0.1.0
