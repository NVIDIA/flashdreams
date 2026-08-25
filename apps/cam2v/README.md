# FlashDreams Cam2V application

`flashdreams-cam2v` owns the reusable v2 application, session, model-generation
loop, camera controls, and timing for interactive camera-to-video models.
Concrete integrations supply an existing runner config plus an input resolver
that turns their asset format into `Cam2VConditioning`.

The application owns the loaded pipeline. Each session owns its autoregressive
cache, keyboard state, camera pose, and SlangPy UI overlay. The
io-thread runs the UI loop over the current video frame; the
model-generation-thread runs the model loop and is the only thread that mutates
rollout state. Model status crosses to the UI loop through `invoke_async`
messages.

The overlay is enabled by default. Pass `-- --no-ui` after the application
arguments to use the default model-output blitter for headless or benchmark
runs.

For UI testing without loading a real model, run the packaged dummy pipeline:

```bash
uv run flashdreams-run-v2 cam2v-dummy --mode webrtc \
  --host 0.0.0.0 --port 8089 -- \
  --step-wait-seconds 0.9 --frames-per-chunk 12
```

The model-generation-thread waits on a `threading.Event` for each synthetic
step while the io-thread continues collecting browser input and presenting
generated frames.

See `integrations/lingbot/lingbot/cam2v/app.py` for the minimal specialization
pattern.
