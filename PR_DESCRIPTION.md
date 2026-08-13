## Summary

Add a model-neutral `flashdreams-runner` shell and a `t2v-app` example
application for running FlashDreams pipelines through replay/MP4, WebRTC, or
headless I/O modes.

The application owns inference. Its runtime owns model weights and one-time
initialization; each session owns its prompt, cache, step state, and generation
logic. The runner owns mode selection, process setup, lifecycle, the main loop,
and presentation.

## High-level design

```text
uv run flashdreams-runner t2v-app {mp4 | replay | webrtc | none}
                  |
                  v
+--------------------------+       create_runtime()       +--------------------+
| flashdreams_runner       | ---------------------------> | t2v_app            |
|                          | <--------------------------- |                    |
| select I/O mode          |          Runtime             | Runtime            |
| initialize Runtime       |                              |  model + pipeline  |
| create Session           |                              |                    |
|                          |        input / output         | Session            |
| main loop:               | ---------------------------> |  prompt + cache    |
|   read input             | <--------------------------- |  generate/finalize |
|   Session.generate       |         StepResult           +--------------------+
|   present output         |
+------------+-------------+
             |
       +-----+-----+----------------+
       |           |                |
       v           v                v
   MP4/replay    WebRTC           None
```

## Entrypoint examples

Generate an MP4 with the packaged default preset:

```bash
uv run flashdreams-runner t2v-app mp4 \
  --prompt "A waterfall" \
  --output o.mp4
```

Use the explicit replay mode name and override its finite iteration count:

```bash
uv run flashdreams-runner t2v-app replay \
  --steps 4 \
  --prompt "A waterfall" \
  --output o.mp4
```

Serve the same application through WebRTC:

```bash
uv run flashdreams-runner t2v-app webrtc \
  --prompt "A waterfall"
```

Run without presentation or artifacts:

```bash
uv run flashdreams-runner t2v-app none \
  --steps 2 \
  --prompt "A waterfall"
```

Select a packaged preset explicitly:

```bash
uv run flashdreams-runner t2v-app mp4 \
  --preset-id self-forcing-wan2.1-t2v-1.3b \
  --prompt "A neon-lit city at night" \
  --output outputs/city.mp4
```

When `--preset-id` is omitted, `t2v-app` uses the catalog's
`default_preset_id`. The packaged default is
`causal-forcing-wan2.1-t2v-1.3b-chunkwise`.

## Application ABI

- Require an application module to expose only
  `create_runtime(ApplicationArguments) -> Runtime`.
- Let the application factory extend the selected mode parser and resolve all
  application-specific command-line configuration.
- Define `Runtime.initialize()`, `Runtime.create_session()`, and
  `Runtime.destroy()` for one-time model and process state.
- Keep presentation fields in application-owned `AppConfig`, exposed through
  `Runtime.config` for runner modes.
- Define `Session.generate()` and `Session.destroy()` for per-user prompt,
  cache, world state, and main-loop logic.
- Keep compatibility methods on the base runtime/session classes so shared
  FlashDreams WebRTC code consumes application runtimes directly without a
  runner-specific adapter.

## Runner and modes

- Add the root-level `flashdreams-runner` workspace package and console entrypoint.
- Select and construct runner-owned I/O modes independently of applications.
- Initialize the application runtime with the selected device and I/O handler.
- Own session creation, deterministic batch input, output delivery, and cleanup.
- Add finite replay/MP4, live WebRTC, and finite headless `none` modes behind an
  extensible `IOHandler` contract.
- Keep `mp4` as a compatibility name for replay-to-file behavior.

## T2V example application

- Add `t2v-app` as an implementation of the application ABI.
- Resolve pipeline object graphs from packaged YAML presets without depending
  on the legacy runner registry.
- Construct and retain the FlashDreams pipeline in `T2VRuntime`.
- Create the prompt-conditioned cache and run pipeline `generate`/`finalize`
  inside `T2VSession.generate()`.
- Keep prompts, dimensions, caches, and step indexes isolated per session.

## Shared FlashDreams changes

- Add reusable pipeline-preset parsing and provider loading under
  `flashdreams.core.pipeline_presets`.
- Share the generator checkpoint prefix-remapping helper from FlashDreams core
  across causal-forcing and self-forcing configurations.

## Validation

- Affected CPU tests cover the application ABI, runner lifecycle, mode
  separation, WebRTC construction, T2V runtime/session ownership, and preset
  resolution.
- Ruff, `ty`, Basedpyright, lockfile validation, and CLI help checks pass.
- GPU model generation was not run as part of this change.
