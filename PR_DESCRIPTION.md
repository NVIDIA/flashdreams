## Summary

Add a model-neutral `flashdreams-app` entrypoint and a declarative `t2v-app`
provider for running FlashDreams text-to-video pipelines as MP4 jobs or WebRTC
services.

The host owns process initialization, pipeline construction, the
runtime/session lifecycle, autoregressive stepping, finalization, cleanup, and
presentation. The T2V package only selects a pipeline preset and supplies the
model-specific conditioning and cache-initialization callback.

## High-level design

```text
uv run flashdreams-app t2v-app {mp4 | webrtc}
                  |
                  v
+----------------------+     request spec      +----------------------+
| flashdreams_app      | --------------------> | t2v_app              |
| generic host         | <-------------------- | declarative provider |
+----------+-----------+        AppSpec        +----------+-----------+
           |                                             |
           | constructs and drives                       | reads
           v                                             v
+----------------------+                      +------------------------+
| FlashDreams runtime  |                      | pipeline_presets.yaml  |
| API (black box)      |                      | + PipelineProvider     |
+----------+-----------+                      +------------------------+
           |
           | video chunks
           v
      +----+----+
      |         |
      v         v
+----------+ +----------+
| MP4 file | | WebRTC   |
| artifact | | stream   |
+----------+ +----------+
```

## Entrypoint examples

Generate an MP4 with the packaged default preset:

```bash
uv run flashdreams-app t2v-app mp4 \
  --prompt "A waterfall" \
  --output o.mp4
```

Serve the same application through WebRTC:

```bash
uv run flashdreams-app t2v-app webrtc \
  --prompt "A waterfall"
```

Select a packaged preset explicitly:

```bash
uv run flashdreams-app t2v-app mp4 \
  --preset-id self-forcing-wan2.1-t2v-1.3b \
  --prompt "A neon-lit city at night" \
  --output outputs/city.mp4
```

Load a custom YAML preset catalog:

```bash
uv run flashdreams-app t2v-app webrtc \
  --preset-config /path/to/pipeline-presets.yaml \
  --preset-id my-t2v-preset \
  --prompt "A waterfall at sunset"
```

When `--preset-id` is omitted, `t2v-app` uses the catalog's
`default_preset_id`. The packaged default is
`causal-forcing-wan2.1-t2v-1.3b-chunkwise`.

## Architecture

- Add the `flashdreams-app` workspace package and console entrypoint.
- Define a minimal provider boundary:
  `create_app_spec(AppRequest) -> AppSpec`.
- Keep the provider surface data-first: a mode-independent pipeline spec plus
  an `AppConfig` for presentation and mode-specific MP4 or WebRTC run data.
- Require provider modules to conform to `AppProvider` with
  `parse_options(parser, argv)` and `create_app_spec(request)`.
- Add the host-owned `PipelineAppRuntime` and `PipelineAppSession`.
- Keep pipeline setup, `generate`/`finalize`, step tracking, cache release, and
  runtime closure in the host.
- Add host-owned MP4 and WebRTC presentation paths without a runtime adapter.
- Type both presentation paths directly against the shared `InferenceRuntime`
  contract.
- Parse only the provider and presentation mode in the host, then let the
  provider extend the mode-specific parser and parse all remaining arguments
  through `parse_options(parser, argv)`.
- Keep pipeline-specific execution behavior encapsulated by the selected
  pipeline config and its `setup()` implementation.

## T2V provider and presets

- Add the `t2v-app` workspace package.
- Describe T2V through an `AppSpec` rather than implementing another runtime
  or session.
- Add packaged YAML presets for causal-forcing and self-forcing WAN pipelines.
- Resolve pipeline providers directly from the YAML catalog without depending
  on the runner registry.
- Support trusted declarative `_target`, `_ref`, and `_tuple` nodes for pipeline
  object graphs.

## Shared FlashDreams changes

- Add reusable pipeline-preset parsing and provider loading under
  `flashdreams.core.pipeline_presets`.
- Share the generator checkpoint prefix-remapping helper from FlashDreams core
  across the causal-forcing and self-forcing configurations.
- Let the shared asynchronous demo and WebRTC manager consume runtime objects
  directly when no model adapter is needed.

## Validation

- `57` affected CPU tests pass.
- Full `pre-commit run -a` passes, including Ruff formatting, lockfile
  validation, and `ty` type checking.
- `flashdreams-app t2v-app mp4 --help` resolves the installed provider and
  composes host and T2V arguments correctly.
- GPU model generation was not run as part of this change.
