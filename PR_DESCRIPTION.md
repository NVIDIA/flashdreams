## Summary

Adds a new `flashdreams-lingbot2` workspace integration for LingBot-World v2 using the official `robbyant/lingbot-world-v2-14b-causal-fast` checkpoint layout, plus WebRTC support for LingBot2 text-driven events.

## Changes

- Adds `integrations/lingbot2` with runner configs:
  - `lingbot-world-v2-14b-causal-fast`
  - `lingbot-world-v2-14b-causal-fast-taehv-window15-sink3`
- Wires the LingBot-World v2 DiT checkpoint:
  - `https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast/blob/main/transformers/diffusion_pytorch_model.safetensors.index.json`
- Reuses FlashDreams Wan VAE, UMT5 text encoder, scheduler, streaming pipeline, and LingBot-style camera-control components.
- Adds a simple launch helper:
  - `integrations/lingbot2/scripts/launch_webrtc.sh`
- Adds configurable LingBot2 WebRTC output resolution, with the launch helper
  defaulting to `640x352` for smoother remote interaction.
- Adds an isolated upstream LingBot-World v2 parity harness under
  `integrations/lingbot2/tests/parity_check`.

## WebRTC Text Events

Adds a model-neutral WebRTC DataChannel event message path:

```json
{"type":"event","event_id":"portal","state":"trigger"}
```

The base WebRTC manager now dispatches optional event messages to runtimes that implement `trigger_event(...)` and returns `event_ack` responses. LingBot2 uses this to precompute text-event embeddings at startup and swap the rollout's cached cross-attention text context when the user triggers an event.

The LingBot2 WebRTC initial-scene payload now advertises:

- `capabilities.text_events`
- `event_catalog`
- `active_event_id`

The browser UI renders event controls from that catalog and updates active state from `event_ack` and `chunk_done` messages.

## Validation

- `uv run --no-sync pytest -m ci_cpu integrations/lingbot2/tests/test_smoke.py integrations/lingbot2/tests/test_webrtc_runtime.py`
  - `17 passed, 1 skipped`
- `uv run --no-sync pytest integrations/lingbot2/tests/test_server_routes.py`
  - `10 passed`
- `uv run --no-sync flashdreams-run --no-instantiate lingbot-world-v2-14b-causal-fast-taehv-window15-sink3`
  - config resolves
- End-to-end one-block rollout:
  - `uv run --no-sync flashdreams-run lingbot-world-v2-14b-causal-fast --example-data True --total-blocks 1 --pipeline.enable-sync-and-profile False --pipeline.diffusion-model.transformer.compile-network False --pipeline.diffusion-model.transformer.use-cuda-graph False`
  - wrote `outputs/lingbot-world-v2-14b-causal-fast.mp4`
- WebRTC interaction test:
  - started LingBot2 WebRTC server
  - negotiated a real `aiortc` peer connection
  - sent `{"type":"event","event_id":"portal","state":"trigger"}`
  - received `event_ack` with `active_event_id: "portal"`
  - received `chunk_done` with `active_event_id: "portal"`
- Resolution config tests:
  - `uv run --no-sync pytest integrations/lingbot2/tests/test_distributed_server_main.py`
  - `8 passed`
- Parity harness checks:
  - `bash -n integrations/lingbot2/tests/parity_check/run.sh`
  - `uv lock --check`
  - fresh-clone patch apply/reverse-apply against pinned upstream commit

## Notes

Running `uv run --extra dev ...` on this host attempted to build `transformer-engine-torch` and failed because CUDA headers such as `cudnn.h` / `nccl.h` were unavailable. Validation used the already-synced environment with `--no-sync`.
