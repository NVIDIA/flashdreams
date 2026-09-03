# FlashDreams Cam2V application

`flashdreams-cam2v` owns the reusable v2 application, session, model-generation
loop, camera controls, and timing for interactive camera-to-video models.
Concrete integrations supply an existing runner config plus an input resolver
that turns their asset format into `Cam2VConditioning`.

The application owns the loaded pipeline. Each session owns its autoregressive
cache, keyboard state, camera pose, and SlangPy UI overlay. The UI thread draws
the retained controls and status widgets over the current video frame; the model
thread runs the model loop and is the only thread that mutates rollout state.
Model status crosses to the UI loop through `invoke_async` messages.

Browser keyboard events are reflected in the outgoing video through the
`Active keys` status line. Arrow keys share the corresponding WASD state, and
losing browser focus clears held controls.

The overlay is enabled by default. Pass `-- --no-ui` after the application
arguments to use the default model-output blitter for headless or benchmark
runs.

The recent model-rate status is the wall-time-weighted throughput of
autoregressive steps whose completions fall in the trailing two seconds. It
excludes between-step pacing, publication, UI, WebRTC, network, and browser
display time. Integrations may enable one concise console record per AR step;
the Lingbot specialization logs its warmup/steady phase, frame count,
synchronized step wall time, and chunk FPS. Model metrics retain the
warmup-excluded cumulative `steady_state_fps` metric for benchmark comparisons.
Presentation pacing uses the complete model-loop wall time, including
post-processing when enabled; the model-only status remains separate.

The v2 runtime runs Cam2V's complete UI-thread lifecycle, including SlangPy
rendering, model-frame conversion, composition, and window writes, on its
default high-priority CUDA presentation stream. `CONTINUOUS` runs the UI every tick so
browser input and time-driven status changes are reflected without waiting for
a new model frame. The UI/write path owns output cadence; WebRTC does not pace
frames again. It keeps two unsent frames in FIFO order and evicts the oldest
queued frame on overflow. A frame already dequeued for the sender or encoder is
committed and is outside that capacity. CUDA priority can overtake queued
lower-priority kernels, but cannot preempt a model kernel that is already
executing.

For UI testing without loading a real model, run the packaged dummy pipeline:

```bash
uv run flashdreams-run-v2 cam2v-dummy --mode webrtc \
  --host 0.0.0.0 --port 8089 -- \
  --step-wait-seconds 0.9 --frames-per-chunk 12
```

The model thread waits on a `threading.Event` for each synthetic step while the
UI thread continues collecting browser input and presenting generated frames.

See `integrations_v2/cam2v_lingbot/cam2v_lingbot/app.py` for the minimal
specialization pattern.
