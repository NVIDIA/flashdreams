# FlashDreams Cam2V application

`flashdreams-cam2v` owns the reusable v2 application, session, model-generation
loop, camera controls, and timing for interactive camera-to-video models.
Concrete integrations supply an existing runner config plus an input resolver
that turns their asset format into `Cam2VConditioning`.

The application owns the loaded pipeline. Each session owns its autoregressive
cache, keyboard state, camera pose, and server-rendered HUD. The io-thread
composites Lingbot-inspired controls and status panels over the current video
frame; the model-generation-thread runs the model loop and is the only thread
that mutates rollout state. Model status crosses to the UI loop through
`invoke_async` messages.

Browser keyboard events are rendered directly into the outgoing video: held
`W`/`A`/`S`/`D`, `Q`/`E`, `I`/`K`, and `J`/`L` controls glow green, and arrow
keys highlight their matching WASD controls. Losing browser focus clears the
highlighted state. Because the HUD is composited server-side, WebRTC clients do
not need Cam2V-specific HTML, CSS, or JavaScript.

The overlay is enabled by default. Pass `-- --no-ui` after the application
arguments to use the default model-output blitter for headless or benchmark
runs.

The MODEL GENERATION panel's `AR ... FPS` value is the wall-time-weighted
throughput of autoregressive steps whose completions fall in the trailing two
seconds. It excludes between-step pacing, publication, UI, WebRTC, network, and
browser display time. Integrations may enable one concise console record per AR
step; the Lingbot specialization logs its warmup/steady phase, frame count,
synchronized step wall time, and chunk FPS. Model metrics retain the
warmup-excluded cumulative `steady_state_fps` metric for benchmark comparisons.

The browser's INPUT → PRESENTED FRAME panel measures control responsiveness
from the DOM event timestamp to the browser compositor timestamp of the first
frame acknowledging that event. Supported camera keys are acknowledged by the
eager held-key HUD redraw, while events without a UI effect are acknowledged by
a later model frame. Both timestamps use the same browser monotonic clock.

HUD chrome and inactive controls are rasterized once. Dynamic status/key
updates stay as compact RGBA8 host data and use a preallocated pinned staging
pool plus a renderer-owned high-priority CUDA composition stream; floating
model frames retain their source dtype through composition. The result carries
only a CUDA readiness event. WebRTC waits for it on its own high-priority
transfer stream, reuses one pinned host frame, and `window.write` does not
return until it has created an independently owned `VideoFrame`.
`ON_DEMAND` suppresses idle redraws while supported keyboard edges, focus loss,
and reset events refresh the HUD. Pointer motion is ignored without redundant
composition; unsupported tracked keys are acknowledged on a later model frame.
The UI/write path owns the output
cadence; the WebRTC sender does not pace the frames again. It keeps two unsent
frames in FIFO order and evicts the oldest queued frame on overflow. A frame
already dequeued for the sender or encoder is committed and is outside that
capacity. CUDA priority can overtake queued lower-priority kernels, but cannot
preempt a model kernel that is already executing.

For UI testing without loading a real model, run the packaged dummy pipeline:

```bash
uv run flashdreams-run-v2 cam2v-dummy --mode webrtc \
  --host 0.0.0.0 --port 8089 -- \
  --step-wait-seconds 0.9 --frames-per-chunk 12
```

WebRTC delivery retains at most two queued, unsent frames to protect input
responsiveness. It preserves their order and evicts the oldest queued frame on
overflow. Use MP4 when the output must be frame-exact.

The model-generation-thread waits on a `threading.Event` for each synthetic
step while the io-thread continues collecting browser input and presenting
generated frames.

See `integrations_v2/cam2v_lingbot/cam2v_lingbot/app.py` for the minimal
specialization pattern.
