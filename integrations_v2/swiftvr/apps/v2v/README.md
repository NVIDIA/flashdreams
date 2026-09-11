<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SwiftVR V2V

This adapter binds the shared
[V2V](../../../../apps/v2v/README.md) application to SwiftVR's 2x video
restoration preset. It reads a video, processes every frame through SwiftVR,
and sends the restored frames to a shared V2 client window. Select MP4 output,
WebRTC browser streaming, or a native window without changing the SwiftVR
adapter.

## Install

From the repository root:

```bash
uv sync --package flashdreams-swiftvr --inexact
```

SwiftVR requires a CUDA-capable NVIDIA GPU. The first run downloads the
checkpoint from `H-oliday/SwiftVR` and prewarms the model, so startup takes
longer than steady-state processing.

MP4 output also requires `ffmpeg` on `PATH`. On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
ffmpeg -version
```

## Run and review an input video

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode mp4 \
  --output-path artifacts/swiftvr-2x.mp4 -- \
  --video-path input.mp4
```

The `--` separates runner options from V2V application options. When the run
finishes, open `artifacts/swiftvr-2x.mp4` in a video player or run:

```bash
ffplay artifacts/swiftvr-2x.mp4
```

Omit `--video-path` to download and process the bounded Big Buck Bunny demo.
Use `--max-chunks 1` for a short smoke test:

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode mp4 \
  --output-path artifacts/swiftvr-smoke.mp4 -- \
  --video-path input.mp4 \
  --max-chunks 1
```

## Watch the output live

For WebRTC browser streaming, install the serving extra and launch the same V2V
entry point with the shared WebRTC client window:

```bash
uv sync --package flashdreams --extra serving --inexact
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode webrtc \
  --host 127.0.0.1 \
  --port 8089 -- \
  --video-path input.mp4
```

Open the URL printed by the runner. On a remote machine, bind to `0.0.0.0` and
make both the HTTP signaling route and WebRTC media path reachable from the
browser.

For a local GPU-backed window, install the local-window extra:

```bash
uv sync --package flashdreams --extra local-window --inexact
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode native-window \
  --window-title "SwiftVR V2V" -- \
  --video-path input.mp4
```

Native-window mode requires a graphical environment, Vulkan support, and
SlangPy. V2V is finite and uninteractive in both live modes: the source is
selected on the command line, and the run stops after all selected chunks are
processed.

## Reproduce a 704p-to-1408p run

For a 1280x704 source video:

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode mp4 \
  --output-path artifacts/swiftvr-2560x1408.mp4 \
  --stats-path artifacts/swiftvr-2560x1408.json -- \
  --video-path input-1280x704.mp4
```

The `v2v-swiftvr` entry point always uses the `swiftvr-2x` preset, so the
output is 2560x1408. `--stats-path` is optional and records model-step timing
for performance review.

Run the following to see all V2V input options:

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr -- --help
```
