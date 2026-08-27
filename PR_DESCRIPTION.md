# Polish the Cam2V server-rendered UI

## Summary

- Replace the raw Cam2V image with a Lingbot-inspired HUD composited on the
  server, including session status, progress, model throughput, and camera
  controls.
- Highlight browser key state immediately, including WASD/arrow aliases and
  focus-loss cleanup.
- Show model AR throughput over the trailing two seconds instead of a
  warmup-to-date average, while retaining concise per-step Lingbot timing logs.
- Correlate browser input with the presented frame and display browser-local
  end-to-end input latency.
- Add a repeatable Lingbot HUD/no-HUD WebRTC benchmark and focused CPU/CUDA
  coverage.

This branch is stacked on the write-driven WebRTC presentation fix. Transport
delivery, presentation cadence ownership, frame ownership, the two-frame sender
mailbox, and CUDA sink readiness remain in that prerequisite PR.

## UI and input responsiveness

- Render a custom Cam2V HUD with Lingbot-style camera controls and explicit
  model-generation status.
- Highlight held `W`/`A`/`S`/`D`, `Q`/`E`, `I`/`K`, and `J`/`L` controls;
  arrow keys map to the matching WASD controls.
- Redraw on fresh input under `PresentationMode.ON_DEMAND`, even when model
  generation has not produced a new frame. UI loops may filter events that do
  not affect their output.
- Cache static HUD chrome and upload compact RGBA8 updates through a two-slot
  pinned staging pool on a high-priority presentation stream.
- Add Pillow as the Cam2V image/font rasterizer dependency and include its
  license attribution.

## Live model status

- Compute MODEL GENERATION FPS from AR steps completed in the trailing two
  seconds, weighted by generated frame count and wall time.
- Redraw at a low idle rate so the panel falls to `AR 0.0 FPS` after two
  seconds without a completion.
- Keep warmup-excluded cumulative throughput available to benchmark output.
- Log each Lingbot AR step's synchronized wall time and chunk FPS to the
  console without enabling device-wide profiling barriers in the interactive
  path.

## Input-to-display latency

- Timestamp browser input, attach bounded correlation IDs, and propagate the
  acknowledgement to the exact UI/model frame that processed it.
- Correlate frame markers with `requestVideoFrameCallback` metadata and show
  latest/P50/P90 browser-local latency.
- Expire samples when presentation or sender congestion drops their frame.
- Move coalesced pointer traffic to a separate ordered data channel so mouse
  backlog cannot block keyboard controls.

## Benchmarking

- Add `configs/v2_webrtc_benchmarks.json` and
  `flashdreams/tools/benchmarks/v2_webrtc_ab.py`.
- Run Lingbot HUD/no-HUD variants in ABBA order in fresh processes with a
  gated loopback aiortc receiver.
- Record raw model, sender, and receiver data and generate JSON/Markdown
  summaries with FPS, latency, queue, and RTP-timestamp checks.

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
- CUDA HUD composition and frame-readiness tests on NVIDIA GB300.
- Full repository pre-commit checks.
