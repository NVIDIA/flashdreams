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

mkdir -p outputs
uv sync --python 3.12 --package flashdreams-omnidreams --no-dev
```

## Null Replay

Run a short replay without writing video output:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams null \
  --device cuda:0 \
  --scenario.example-data true \
  --scenario.example-data-uuid 239560dc-33d1-11ef-9720-00044bcbccac \
  --scenario.total-blocks 10
```

## Precomputed MP4 Replay

Generate an MP4 from bundled single-view sample data and pre-rendered HDMaps:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams mp4 \
  --device cuda:0 \
  --scenario.example-data true \
  --scenario.example-data-uuid 239560dc-33d1-11ef-9720-00044bcbccac \
  --scenario.total-blocks 10 \
  --output.fps 30 \
  --output.path outputs/omnidreams-precomputed.mp4
```

The default output is `outputs/omnidreams.mp4` when `--output.path` is omitted.
The versioned manifest form remains supported:

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
  --device cuda:0 \
  --scenario.conditioning-mode ludus-scene-driving \
  --scenario.keyboard-trace \
  integrations/omnidreams/omnidreams/demo/traces/ludus_forward_sweep_60s.json \
  --scenario.scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4 \
  --scenario.total-blocks 10 \
  --output.fps 30 \
  --output.path outputs/omnidreams-ludus.mp4
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
  --host 0.0.0.0 \
  --port 8089 \
  --device cuda:0 \
  --scenario.scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4 \
  --output.video-height 704 \
  --output.video-width 1280
```

Then open:

- [http://localhost:8089/request_session](http://localhost:8089/request_session)
- [http://localhost:8089/healthz](http://localhost:8089/healthz)

The manifest-based WebRTC form remains supported:

```bash
uv run --python 3.12 --package flashdreams-omnidreams flashdreams-run \
  omnidreams webrtc \
  --manifest configs/launch_manifest/omnidreams_webrtc.yaml
```

The scene UUID is optional; when omitted, the runtime uses the default
Hugging Face WebRTC scene. Override ``scenario.scene_uuid``, select a weather
variant with ``scenario.scene_variant``, or set ``scenario.scene_dir`` for a
local staged scene.
