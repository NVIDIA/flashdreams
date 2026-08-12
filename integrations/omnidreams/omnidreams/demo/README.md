<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams Shared Demo API

This folder contains the experimental OmniDreams demo built on
`flashdreams.runtime.demo`.

Run commands from the FlashDreams workspace root. The following setup was used
for remote GPU validation on GB300:

```bash
cd /path/to/flashdreams
export HF_TOKEN=<YOUR-HF-TOKEN>
export CUDA_HOME=/usr/local/cuda-13.1
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
hash -r
"$CUDA_HOME/bin/nvcc" --version

uv sync --python 3.12 --package flashdreams-omnidreams --no-dev
```

## Null Replay

Run a short replay without writing video output:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams null \
  --manifest configs/launch_manifest/omnidreams_null.yaml
```

## Precomputed MP4 Replay

Generate an MP4 from bundled single-view sample data and pre-rendered HDMaps
with no manifest:

```bash
uv run flashdreams-run omnidreams mp4
```

The default output is `outputs/omnidreams.mp4`. To override the sample,
rollout length, frame rate, or output path, use the versioned manifest:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams mp4 \
  --manifest configs/launch_manifest/omnidreams_mp4.yaml
```

This replay path mirrors the benchmark runner path: it uses a prompt, first
frame, and pre-rendered HDMap video. It does not load a Ludus scene or render
HDMaps at runtime. The demo defaults to the stable non-perf OmniDreams preset
used by the benchmark path.

Set `scenario.example_data_uuid` to select another bundled single-view sample,
or set `scenario.example_data: false` and provide explicit asset paths.

## Ludus MP4 Replay

Generate an MP4 by rendering HDMap conditioning from a recorded keyboard trace:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams mp4 \
  --manifest path/to/omnidreams-ludus-mp4.yaml
```

The `omnidreams-perf` preset remains an
explicit runner-slug opt-in. It should become the default only after the
compile/cache behavior is reliable enough for the demo path.

## WebRTC

WebRTC uses the shared FlashDreams server, session manager, and runtime worker.
The small model adapter in this package loads one scene, renders HDMap
conditioning with Ludus, and runs OmniDreams from browser WASD controls:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams webrtc \
  --manifest configs/launch_manifest/omnidreams_webrtc.yaml
```

The scene UUID is optional; when omitted, the runtime uses the default
Hugging Face WebRTC scene. Override ``scenario.scene_uuid``, select a weather
variant with ``scenario.scene_variant``, or set ``scenario.scene_dir`` for a
local staged scene.
