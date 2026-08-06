<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams Shared Demo API

This folder contains the experimental OmniDreams demo built on
`flashdreams.runtime.demo`.

Run commands from the FlashDreams workspace root:

```bash
cd /path/to/flashdreams
export HF_TOKEN=<YOUR-HF-TOKEN>
```

## MP4 Replay

Generate an MP4 from the bundled single-view sample data:

```bash
mkdir -p outputs
uv run --package flashdreams-omnidreams omnidreams-demo replay \
  --output outputs/omnidreams-demo.mp4
```

This replay path mirrors the benchmark runner path: it uses a prompt, first
frame, and pre-rendered HDMap video. It does not load a Ludus scene or render
HDMaps at runtime. The demo defaults to the stable non-perf OmniDreams preset
used by the benchmark path.

To provide benchmark-style assets explicitly:

```bash
uv run --package flashdreams-omnidreams omnidreams-demo replay \
  --prompt "Driving scene from a front-facing car camera." \
  --hdmap-video-paths /path/to/camera_front_wide_120fov_hdmap.mp4 \
  --first-frame-paths /path/to/first_frame.png \
  --camera-names camera_front_wide_120fov \
  --output outputs/omnidreams-demo.mp4
```

Pass `--example-data-uuid <uuid>` to select another bundled single-view sample,
or `--no-example-data` to require explicit asset paths.

The `omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf` preset remains an
explicit `--preset-id` opt-in. It should become the default only after the
compile/cache behavior is reliable enough for the demo path.

## WebRTC

WebRTC uses the shared demo launcher around the existing Omnidreams live WebRTC
runtime. It is still scene-driven and uses Ludus to render HDMap conditioning
from a scene:

```bash
uv run --package flashdreams-omnidreams omnidreams-demo webrtc \
  --host 0.0.0.0 \
  --port 8082
```

The scene UUID is optional; when omitted, the runtime uses the default
Hugging Face WebRTC scene. Override the scene with `--scene-uuid`, select a
weather variant with `--scene-variant default|rain|snow`, or use
`--scene-dir /path/to/local/scene` for a local staged scene.
