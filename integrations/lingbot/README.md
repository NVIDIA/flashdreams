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
uv run --package flash-lingbot python -m lingbot.webrtc.server --host 0.0.0.0 --port 8080 --config_name LingBot-World-Fast
```

Then open:

- [http://localhost:8080/request_session](http://localhost:8080/request_session)
- [http://localhost:8080/healthz](http://localhost:8080/healthz) (`runtime_ready` indicates preload completion)

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
