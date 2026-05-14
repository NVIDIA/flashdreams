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

# `lingbot`

Lingbot integration package for `flashdreams` that exposes a minimal WebRTC server.

## What It Provides

- `GET /request_session` serves a standalone viewer page (`HTML/CSS/JS` files on disk, not inlined in Python).
- `POST /api/webrtc/offer` performs SDP offer/answer signaling.
- Runtime/model/config preloading during server startup (before handling requests).
- A single active WebRTC session per server process.
- Action-bound control flow:
  1. browser sends an action (`keydown`, `keyup`, or `step`) over DataChannel,
  2. server runs one Lingbot AR inference chunk,
  3. server enqueues chunk frames to the WebRTC track and emits `chunk_done`.

## Run

From repository root:

```bash
uv run --package flash-lingbot python -m lingbot.webrtc.server --host 0.0.0.0 --port 8089 --config_name lingbot-world-fast-flash

# 4 gpus
uv run --package flash-lingbot \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m lingbot.webrtc.server \
  --host 0.0.0.0 --port 8089 \
  --config_name lingbot-world-fast-flash
```

Then open:

- [http://localhost:8089/request_session](http://localhost:8089/request_session)
- [http://localhost:8089/healthz](http://localhost:8089/healthz) (`runtime_ready` indicates preload completion)

## Test

From repository root:

```bash
uv run --package flash-lingbot --extra dev pytest integrations/lingbot/tests
```

## Runtime Requirements

- CUDA-capable GPU for Lingbot inference.
- `HF_TOKEN` exported for Hugging Face model access.
- Lingbot example assets available under `assets/example_data/lingbot_world`:
  - `image.jpg`
  - `intrinsics.npy`
  - `poses.npy` (optional but recommended for world-scale normalization)
  - `prompt.txt`

## DataChannel Message Format

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
- If multiple key events arrive before the next chunk starts, the server aggregates them and
  applies latest-pressed precedence per component (forward/backward, turn, strafe, pitch).

Server -> browser:

```json
{
  "type": "chunk_done",
  "chunk_index": 3,
  "num_frames": 12,
  "enqueued_frames": 12
}
```
