<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# `omnidreams`

Omnidreams integration package for `flashdreams`.

## Hugging Face assets

Omnidreams resolves public Omni Dreams assets from the `nvidia` Hugging Face
org:

- `nvidia/omni-dreams-models` for checkpoints.
- `nvidia/omni-dreams-samples` for bundled example data.
- `nvidia/omni-dreams-scenes` for WebRTC scenes.

Set `HF_TOKEN` to a token with access to these repos before running or
importing FlashDreams:

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>
```

Internal S3-backed runs can still set `FLASHDREAMS_INTERNAL_STORAGE=1`, which
switches checkpoint and example-data URLs back to `s3://flashdreams`.

## Run interactive-drive (desktop demo)

The `omnidreams.interactive_drive` subpackage ships a single-process
driving demo with a Ludus OpenGL raster backend and a PyTorch world-model
backend ([see its README](omnidreams/interactive_drive/README.md) for the
full guide). From the flashdreams workspace root:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams interactive-drive
```

The `interactive-drive` extra adds `slangpy` (the Vulkan-backed local
windowing runtime); server users running only `omnidreams.webrtc` or
`omnidreams.grpc` can skip it. The default scene auto-stages from
`nvidia/omni-dreams-scenes` on first launch when `HF_TOKEN` is set; use
`omnidreams-prepare` for explicit staging of arbitrary scene UUIDs
or to pre-warm the ~14 GB Cosmos-Reason1 text encoder.

## Run WebRTC server

From the workspace root, run:

```bash
uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 -m omnidreams.webrtc.server --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf --scene-uuid 065dcac9-ee67-4434-a835-c6b816c88e48 --port 8089
```

When `--scene_dir` is omitted, the server downloads the selected scene from the
configured Hugging Face org, extracts its `clipgt-<uuid>.usdz` archive, and
stages it under `FLASHDREAMS_CACHE_DIR` (or `~/.cache/flashdreams`). If
`--scene-uuid` is omitted too, the server uses the default WebRTC scene. The
runtime expects `clipgt/first_image.*` and `clipgt/prompt.txt` under the scene
directory. Pass `--scene_dir <path>` to use a pre-staged local scene instead.

## Run gRPC server

From the workspace root, run:

```bash
uv run --package flash-omnidreams torchrun --nproc_per_node 1 -m omnidreams.grpc.server --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf --host 0.0.0.0 --port 50051
```

The server implements `omnidreams.grpc.protos.video_model.WorldModelService`
and listens on `0.0.0.0:50051` by default. Clients provide the static map,
camera specs, initial frames, prompt, rig trajectory, and dynamic actor state
through the gRPC API. Use `--record_dir <dir>` to save replayable session logs,
and add `--enable_profiling --profile_output <path>` when collecting timing
data. For distributed/context-parallel launches, increase `--nproc_per_node`;
the world size must be compatible with the selected pipeline config's camera
count.
