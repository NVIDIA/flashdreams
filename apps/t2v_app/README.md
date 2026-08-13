# T2V Example Application

`t2v-app` is an example implementation of the `flashdreams-runner`
application ABI. Its public module exports only `create_runtime(arguments)`.

The implementation has three layers:

- `application.py` parses T2V arguments, resolves the YAML pipeline preset, and
  returns an uninitialized `T2VRuntime`.
- `runtime.py` owns pipeline construction, model weights, and one-time device
  initialization.
- `session.py` owns the prompt, autoregressive cache, step counter, and the
  pipeline `generate`/`finalize` calls for each main-loop iteration.

The runner owns mode selection, process setup, session lifecycle, iteration,
and presentation.

```bash
uv run flashdreams-runner t2v-app mp4 \
  --preset-id causal-forcing-wan2.1-t2v-1.3b-chunkwise \
  --prompt "A waterfall at sunset" \
  --output outputs/waterfall.mp4

uv run flashdreams-runner t2v-app webrtc \
  --preset-id self-forcing-wan2.1-t2v-1.3b

uv run flashdreams-runner t2v-app none \
  --steps 2 \
  --prompt "A waterfall"
```

## Pipeline presets

The application loads a YAML preset catalog through
`flashdreams.core.pipeline_presets` and asks the selected pipeline provider to
construct a `StreamInferencePipelineConfig`. The packaged catalog is
[`t2v_app/pipeline_presets.yaml`](t2v_app/pipeline_presets.yaml); pass
`--preset-config` to use another catalog.

Every preset specifies a pipeline provider, application defaults, and
provider-owned pipeline options. `total_blocks` is an optional default for
finite runner modes; `--steps` overrides it. The T2V WebRTC page lets each user
edit the prompt and video duration, keeps the connection open for subsequent
generations, plays the completed MP4, and downloads a ZIP containing the video
and prompt metadata.

FlashDreams' `ObjectGraphPipelineProvider` supports these trusted declarative
nodes:

- `_target: module:attribute` imports and calls a config class with the other
  mapping entries as keyword arguments.
- `_ref: module:attribute` imports a value such as a checkpoint transform
  without calling it.
- `_tuple: [...]` preserves tuple-valued config fields.

A custom package can expose a zero-argument pipeline provider class or instance
implementing `flashdreams.core.pipeline_presets.PipelineProvider` and reference
it from the YAML `provider` field.

See the [`flashdreams-runner` application ABI](../../flashdreams_runner/README.md#application-abi)
for the runtime, session, and mode lifecycle.
