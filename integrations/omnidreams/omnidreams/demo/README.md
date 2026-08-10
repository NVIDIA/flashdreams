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
uv run --python 3.12 --package flashdreams-omnidreams omnidreams-demo replay \
  --output-mode null \
  --device cuda:0 \
  --total-blocks 10
```

## Precomputed MP4 Replay

Generate an MP4 from bundled single-view sample data and pre-rendered HDMaps:

```bash
mkdir -p outputs
uv run --python 3.12 --package flashdreams-omnidreams omnidreams-demo replay \
  --device cuda:0 \
  --example-data \
  --example-data-uuid 239560dc-33d1-11ef-9720-00044bcbccac \
  --total-blocks 225 \
  --fps 30 \
  --output outputs/omnidreams-demo-precomputed-1min.mp4
```

This replay path mirrors the benchmark runner path: it uses a prompt, first
frame, and pre-rendered HDMap video. It does not load a Ludus scene or render
HDMaps at runtime. The demo defaults to the stable non-perf OmniDreams preset
used by the benchmark path.

Pass `--example-data-uuid <uuid>` to select another bundled single-view sample,
or `--no-example-data` to require explicit asset paths.

## Ludus MP4 Replay

Generate an MP4 by rendering HDMap conditioning from a recorded keyboard trace:

```bash
uv run --python 3.12 --package flashdreams-omnidreams omnidreams-demo replay \
  --conditioning-mode ludus-scene-driving \
  --keyboard-trace integrations/omnidreams/omnidreams/demo/traces/ludus_forward_sweep_60s.json \
  --device cuda:0 \
  --scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4 \
  --seed 42 \
  --total-blocks 226 \
  --output outputs/omnidreams-demo--ludus-1min.mp4
```

The `omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf` preset remains an
explicit `--preset-id` opt-in. It should become the default only after the
compile/cache behavior is reliable enough for the demo path.

## WebRTC

WebRTC uses the shared FlashDreams server, session manager, and runtime worker.
The small model adapter in this package loads one scene, renders HDMap
conditioning with Ludus, and runs OmniDreams from browser WASD controls:

```bash
uv run --python 3.12 --package flashdreams-omnidreams omnidreams-demo webrtc \
  --host 0.0.0.0 \
  --port 8089 \
  --device cuda:0 \
  --scene-uuid 0d404ff7-2b66-498c-b047-1ed8cded60d4
```

The scene UUID is optional; when omitted, the runtime uses the default
Hugging Face WebRTC scene. Override the scene with `--scene-uuid`, select a
weather variant with `--scene-variant default|rain|snow`, or use
`--scene-dir /path/to/local/scene` for a local staged scene.
