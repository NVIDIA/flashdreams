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

# flashdreams-lingbot

Lingbot-World streaming camera-control I2V integration + a minimal WebRTC
demo server, packaged as a [`flashdreams`](../..) plugin, in a
standalone repo.

This is a worked example of the
[Adding a new model integration](../../docs/source/developer_guides/new_recipes.rst)
developer-guide flow, extended with a per-plugin runtime server.

## Shipped slugs

| slug | description |
| --- | --- |
| `lingbot-world-fast` | Lingbot World Fast streaming camera-control I2V (Wan VAE decoder, 4-step). |
| `lingbot-world-fast-flash` | Lingbot World Fast-Flash (LightTAE decoder, tighter streaming window). |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/lingbot
```

## HuggingFace setup

Checkpoints are auto-downloaded from HuggingFace at first run. Set an
auth token first.

```bash
# huggingface token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default
```

## Run

Once installed, the slugs are discovered automatically by `flashdreams-run`:

```bash
# List every registered runner (this plugin's slugs appear under "lingbot-world-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run lingbot-world-fast --help

# Single-GPU demo with the bundled example assets (lazy-synced from S3).
uv run flashdreams-run lingbot-world-fast --example-data True --total-blocks 21

# Custom inputs (production layout).
uv run flashdreams-run lingbot-world-fast \
    --image-path /path/to/first_frame.jpg \
    --pose-path /path/to/poses.npy \
    --intrinsic-path /path/to/intrinsics.npy \
    --prompt "your text prompt here" --total-blocks 21
```

Multi-GPU via context-parallelism (Wan 2.1 CP assumes `cp_size == world_size`):

```bash
# e.g. 4GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    lingbot-world-fast --example-data True --total-blocks 21
```

## Programmatic access

Access via runner.
```python
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST as runner_config
from flashdreams.infra.config import derive_config

cfg = derive_config(runner_config, prompt="A cinematic flythrough.", example_data=True)
runner = cfg.setup()
runner.run()
```

Access via pipeline.
```python
import torch
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST as pipeline_config
from lingbot.encoder.camctrl import CamCtrlInput

pipeline = pipeline_config.setup().to("cuda").eval()
sp = pipeline.decoder.spatial_compression_ratio

cache = pipeline.initialize_cache(
    text=["A cinematic flythrough."],
    image=first_frames_t,         # [T=1, C, H, W] in [-1, 1] (batch_shape=())
    height=464 // sp,             # latent height for DiT
    width=832 // sp,              # latent width for DiT
)

total_blocks: int = 21
generated_chunks: list[torch.Tensor] = []
for i in range(total_blocks):
    camctrl_input = CamCtrlInput(
        intrinsics=...,           # [T_chunk, 4] (fx, fy, cx, cy)
        poses=...,                # [T_chunk, 4, 4] camera-to-world
        world_scale=...,
    )
    video_chunk = pipeline.generate(autoregressive_index=i, cache=cache, input=camctrl_input)
    pipeline.finalize(autoregressive_index=i, cache=cache)  # update KV cache
    generated_chunks.append(video_chunk.cpu())              # each chunk is [T, C, H, W]
```

## Run (WebRTC interactive demo)

The `lingbot.webrtc` subpackage exposes a minimal WebRTC server that
binds the integration pipeline to keyboard input over a DataChannel and streams the
generated video back to the browser.

- `GET /request_session` serves a standalone viewer page (HTML/CSS/JS files on disk, not inlined in Python).
- `POST /api/webrtc/offer` performs SDP offer/answer signaling.
- Runtime/model/config preloading during server startup (before handling requests).
- A single active WebRTC session per server process.
- Action-bound control flow:
  1. browser sends an action (`keydown`, `keyup`, or `step`) over DataChannel,
  2. server runs one Lingbot AR inference chunk,
  3. server enqueues chunk frames to the WebRTC track and emits `chunk_done`.

From repository root:

```bash
uv run --package flashdreams-lingbot python -m lingbot.webrtc.server \
    --host 0.0.0.0 --port 8089 --config_name lingbot-world-fast-flash

# 4 GPUs
uv run --package flashdreams-lingbot \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m lingbot.webrtc.server \
  --host 0.0.0.0 --port 8089 \
  --config_name lingbot-world-fast-flash
```

Then open:

- [http://localhost:8089/request_session](http://localhost:8089/request_session)
- [http://localhost:8089/healthz](http://localhost:8089/healthz) (`runtime_ready` indicates preload completion)

### Runtime requirements

- CUDA-capable GPU for Lingbot inference.
- `HF_TOKEN` exported for HuggingFace model access.
- Lingbot example assets available under `assets/example_data/lingbot_world`:
  - `image.jpg`
  - `intrinsics.npy`
  - `poses.npy` (optional but recommended for world-scale normalization)
  - `prompt.txt`

### DataChannel message format

Browser -> server:

```json
{
  "type": "action",
  "action": {
    "event": "keydown",
    "key": "w"
  }
}
```

- Supported `event` values:
  - `keydown` / `keyup` (requires `key` in `w,a,s,d,q,e,i,j,k,l`)
  - `step` (no key required; generates a chunk using current key state)
- Key mapping:
  - `w/s`: forward/backward
  - `a/d` (or `j/l`): yaw left/right
  - `q/e`: strafe left/right
  - `i/k`: pitch up/down
- If multiple key events arrive before the next chunk starts, the server
  aggregates them and applies latest-pressed precedence per component
  (forward/backward, turn, strafe, pitch).

Server -> browser:

```json
{
  "type": "chunk_done",
  "chunk_index": 3,
  "num_frames": 12,
  "enqueued_frames": 12
}
```

## Tests

```bash
uv run --extra dev pytest integrations/lingbot/tests
```
