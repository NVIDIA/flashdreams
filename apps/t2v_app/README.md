# T2V App Provider

`t2v-app` is a model provider for the generic `flashdreams-app` host. It
returns a declarative `PipelineAppSpec`; it does not implement a runtime or
session and does not own setup, stepping, finalization, cleanup, MP4 writing,
or WebRTC.

The provider loads a YAML preset catalog through
`flashdreams.core.pipeline_presets` and asks the selected pipeline provider to
construct a FlashDreams `StreamInferencePipelineConfig`. It does not use the
`flashdreams.runner_configs` registry. The packaged catalog is
[`t2v_app/pipeline_presets.yaml`](t2v_app/pipeline_presets.yaml); pass
`--preset-config` to use another catalog.

```bash
uv run flashdreams-app t2v-app mp4 \
  --preset-id causal-forcing-wan2.1-t2v-1.3b-chunkwise \
  --prompt "A waterfall at sunset" \
  --output outputs/waterfall.mp4

uv run flashdreams-app t2v-app webrtc \
  --preset-id self-forcing-wan2.1-t2v-1.3b \
  --prompt "A neon-lit city at night"

uv run flashdreams-app t2v-app mp4 \
  --preset-config /path/to/presets.yaml \
  --preset-id my-t2v-preset \
  --prompt "A waterfall" \
  --output outputs/waterfall.mp4
```

Every YAML preset must specify `provider`, all six runtime/presentation fields,
and the provider-owned `pipeline` options. FlashDreams'
`ObjectGraphPipelineProvider` supports these declarative nodes:

- `_target: module:attribute` imports and calls a config class with the other
  mapping entries as keyword arguments.
- `_ref: module:attribute` imports a value such as a checkpoint transform
  without calling it.
- `_tuple: [...]` preserves tuple-valued config fields.

A custom package can expose a zero-argument provider class (or provider
instance) implementing `flashdreams.core.pipeline_presets.PipelineProvider`
and reference it from `provider`. Preset YAML is trusted configuration because
provider and object-graph references import Python objects.

At the provider boundary, T2V contributes only its preset selection,
conditioning values, presentation metadata, and a cache initializer that maps
the prompt and pixel dimensions to the selected pipeline. `flashdreams-app`
constructs and drives the resulting pipeline.
