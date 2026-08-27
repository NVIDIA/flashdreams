# Fix write-driven WebRTC presentation

## Summary

- Make the UI thread own frame selection, composition, and output cadence.
- Materialize each frame in `window.write()` and let WebRTC deliver accepted
  frames immediately, without a second sender pacer.
- Keep at most two queued, unsent WebRTC frames and evict the oldest queued
  frame when congestion fills the queue.
- Run SlangPy rendering and composition on a high-priority CUDA presentation
  stream, then transfer the ready result through the WebRTC sink's dedicated
  CUDA stream.
- Redraw the retained Cam2V SlangPy controls immediately for relevant browser
  input while preserving the latest model frame.
- Generalize timestamped realtime input buffering and source-aware keyboard
  state for reuse by other applications.

## Presentation and transport

- Advance multi-frame model chunks one frame at a time in the UI thread.
- Adapt presentation cadence to recent model completion throughput without
  bursting after a slow model step.
- Synchronize and materialize CUDA output before `window.write()` returns, so
  queued frames own independent host storage.
- Deliver dequeued frames to aiortc as soon as they are requested. WebRTC
  neither sleeps nor repeats frames.
- Preserve FIFO order for the two unsent queue slots and report frames evicted
  by sender lag.
- Treat transient WebRTC `disconnected` state as recoverable; show an error only
  for terminal `failed` or `closed` states.

## Cam2V responsiveness

- Keep the original retained SlangPy/ImGui-style camera-control overlay.
- Use `PresentationMode.ON_DEMAND` so browser input can refresh the overlay
  before another model chunk completes.
- Display model throughput from AR steps completed in the trailing two seconds
  and retain optional synchronized per-step AR timing logs.
- Propagate browser event IDs and timestamps to the first UI or model frame that
  processed them. Browser frame-correlation support remains available for
  diagnostics without adding a visible HTML/CSS latency panel.
- Coalesce pointer traffic on a separate ordered data channel so it cannot build
  a backlog ahead of keyboard controls.

## Input handling

- Hoist realtime timestamp normalization and catch-up into
  `RealtimeInputTimeline`.
- Hoist source-aware key aliasing and state segments into `KeyboardStateTrack`.
- Keep `KeyboardResampler` as a compatibility facade for existing Cam2V users.

## Benchmarking

- Add a process-isolated ABBA WebRTC benchmark with a loopback aiortc receiver.
- Record model, presentation, sender, and receiver metrics with explicit FPS,
  queue, and RTP timestamp checks.

The loopback benchmark covers server composition through aiortc decoding. It
does not include a real network, browser DOM work, or the physical display
compositor.

## Run

Dummy model:

```bash
uv run flashdreams-run-v2 cam2v-dummy --mode webrtc \
  --host 0.0.0.0 --port 8089 -- \
  --step-wait-seconds 0.9 --frames-per-chunk 12
```

Lingbot:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

## Validation

- Focused Cam2V, runtime, input, WebRTC, and benchmark CPU suites.
- CUDA SlangPy composition, output-readiness, and WebRTC materialization tests.
- Full repository pre-commit checks.
