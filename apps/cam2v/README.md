# FlashDreams Cam2V application

`flashdreams-cam2v` owns the reusable v2 application, session, model-generation
thread, camera controls, and timing for interactive camera-to-video models.
Concrete integrations supply an existing runner config plus an input resolver
that turns their asset format into `Cam2VConditioning`.

The application owns the loaded pipeline. Each session owns its autoregressive
cache, first frame, keyboard state, camera pose, and ImGui overlay. The
io-thread renders live controls and model timing over the current video frame;
the model-generation-thread consumes new keyboard edges and is the only thread
that mutates rollout state. Model status crosses to the UI thread through
`invoke_async` messages.

The overlay is enabled by default. Pass `-- --no-ui` after the application
arguments to use the default model-output blitter for headless or benchmark
runs.

See `integrations/lingbot/lingbot/cam2v/app.py` for the minimal specialization
pattern.
