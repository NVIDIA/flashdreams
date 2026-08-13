# FlashDreams App Host

`flashdreams-app` is the model-neutral application host. It finds a provider
distribution in the active Python environment, asks it for a declarative
pipeline application spec, constructs the runtime, creates a session, runs the
session loop, and writes presentation artifacts itself.

```bash
uv run flashdreams-app t2v-app mp4 --output o.mp4 --prompt "A waterfall"
uv run flashdreams-app t2v-app webrtc --prompt "A waterfall"
```

## Provider contract

A compatible package must expose an importable module with:

- `add_arguments(parser)`, which registers provider-specific flags. Providers
  without custom flags implement this as a no-op.
- `create_app_spec(config: flashdreams_app.AppConfig)`, returning a
  `PipelineAppSpec` with a `StreamInferencePipelineConfig`, initial
  conditioning, presentation metadata, step count, and a `PipelineContract`
  cache initializer.

The module structurally conforms to `flashdreams_app.AppProvider`; the host
validates this contract when it loads the installed provider.

The host owns process/distributed initialization, pipeline setup,
runtime/session lifecycle, `generate`/`finalize` stepping, MP4 writing, and
WebRTC serving. Providers only select/configure a pipeline and describe how
global conditioning initializes its cache. Pipeline-specific execution options,
including compilation and CUDA graphs, remain encapsulated by the pipeline
config and its `setup()` implementation.
