# T2V Application Architecture

The T2V application is an adapter between `flashdreams-runner` and a
FlashDreams streaming inference pipeline. The runner owns orchestration and
presentation; the application owns model configuration, model state, and
generation.

## Minimal application contract

The runner discovers the installed `t2v-app` distribution and imports its
top-level `t2v_app` module. That module exposes one public factory:

```python
create_runtime(arguments: ApplicationArguments) -> Runtime
```

The returned runtime implements the contract in
[`flashdreams_runner/contracts.py`](../../flashdreams_runner/contracts.py):

- `config` describes output identity, frame rate, layout, dimensions, and the
  optional default step count.
- `initialize(device, io_handler)` constructs process-wide model state.
- `create_session(initial_input)` creates isolated generation state.
- `destroy()` releases process-wide resources.

Each session implements:

- `step_index`, the next autoregressive iteration.
- `step(inputs)`, which returns a FlashDreams `StepResult`.
- `destroy()`, which releases session state.

`Runtime` and `Session` also adapt this runner-facing API to the shared
FlashDreams `InferenceRuntime` and `InferenceSession` protocols through
`start_session`, `next_step_request`, and `close`.

## Components

### Application factory

[`t2v_app/application.py`](t2v_app/application.py) parses application
arguments, loads a pipeline preset, and creates an uninitialized `T2VRuntime`.
It does not construct model weights.

### Runtime

[`t2v_app/runtime.py`](t2v_app/runtime.py) owns the configured pipeline and
process-wide model weights. Initialization constructs the pipeline and moves it
to the selected device. The runtime creates one `T2VSession` for each isolated
generation.

### Session

[`t2v_app/session.py`](t2v_app/session.py) owns per-generation state:

- prompt and video dimensions;
- autoregressive cache;
- current block index;
- optional WebRTC recording.

Each generation step calls the pipeline's `generate` and `finalize` methods and
wraps the resulting video tensor in a `StepResult`.

### WebRTC customization

[`t2v_app/webrtc.py`](t2v_app/webrtc.py) is an optional adapter installed only
when the selected I/O handler is `WebRTCMode`. It supplies browser assets,
initial session input, prompt and duration updates, playback, and artifact
download routes.

## Control flow

### Startup

```text
flashdreams-runner
  -> import t2v_app
  -> t2v_app.create_runtime(arguments)
  -> T2VRuntime.initialize(device, io_handler)
  -> io_handler.run(runtime, drive_session)
```

### Finite modes (`mp4`, `replay`, and `none`)

```text
IO handler
  -> drive_session(runtime, input_handler, output_handler)
  -> runtime.create_session(initial_input)
  -> session.step(step_input), repeated until input ends
  -> output_handler.write(step_result)
  -> session.destroy()
```

### WebRTC mode

```text
WebRTCMode
  -> T2VWebRTCCustomization.prepare_initial_input()
  -> shared WebRTC session manager
  -> runtime.start_session(initial_input)
  -> session.next_step_request()
  -> session.step(step_input), repeated until complete
  -> session.close()
```

Browser prompt updates call `T2VRuntime.prepare_session_input()` to replace the
initial input used by the next generation.

## Data boundary

The primary values crossing between the application and FlashDreams are:

- `ApplicationArguments`: runner mode and unparsed application arguments.
- `AppConfig`: presentation metadata consumed by runner I/O modes.
- `InferenceInput`: global conditioning at session creation and optional
  per-step input.
- `StepRequest`: shared-serving request for the next iteration.
- `StepResult`: generated video chunk, layout, metadata, and metrics.
- `OutputArtifact`: persistent output returned by an I/O handler.

Pipeline presets, WAN recipe classes, autoregressive cache contents, browser
routes, and MP4 recording are implementation details rather than part of the
minimal application ABI.

## Ownership boundary

- `flashdreams-runner` owns application discovery, CLI modes, device/process
  setup, lifecycle, iteration, and output presentation.
- `flashdreams.runtime` owns shared inference inputs, requests, results,
  artifacts, and serving protocols.
- `flashdreams.infra` owns reusable pipeline, decoder, post-processing, and
  configuration primitives.
- `t2v_app` owns T2V arguments, presets, pipeline setup, session state,
  generation, and optional WebRTC behavior.
