# FlashDreams Demo API

`flashdreams.demo` is the public authoring layer for demos that should run
through the shared demo runtime. It gives model authors a small set of objects
to implement or compose, while the lower-level `flashdreams.runtime.demo`
package keeps ownership of the session drivers, `StepPipeline`, model worker
thread, metrics, output delivery, and cleanup.

The goal is that a demo can support replay, MP4, null/headless, WebRTC, and
eventually native-window output without each integration reimplementing its own
loop, output handling, input plumbing, or server dispatch.

## Main Pieces

`Application`
: Owns app-level lifecycle. It initializes launch state, creates one model
  session, and closes app resources.

`ApplicationSession`
: Owns one model session. It reports `SessionInfo`, returns the next
  `StepRequirements`, runs one `step(InferenceInput) -> StepResult`, and closes
  per-session resources. The session does not write to output sinks, sleep for
  backpressure, or own the loop.

`Runner`
: Drives one session through the shared runtime. It adapts `Application` and
  `IOHandler` to `run_demo_session(...)` or `run_demo_session_async(...)`, so
  every mode keeps the shared worker-thread, metrics, output, cancellation, and
  cleanup behavior.

`IOHandler`
: Bundles the input, output, and stop-signal side of a run. It provides input
  windows, exposes pull-style input state, emits output chunks, and reports
  `should_exit()`.

`IOHandlerServer`
: Server-shaped facade for transports such as WebRTC. WebRTC does not have a
  ready IO handler until a peer connects, so a server accepts connections and
  passes one `IOHandler` per session to the shared runner callback.

`DemoAdapterApplication`
: Adapter for demos that already implement the lower-level `DemoAdapter`
  contract. Most migrated demos should use this rather than implementing
  `Application` from scratch.

`create_demo_application(...)`
: Small command-app helper for demos that have parser/spec/adapter functions and
  do not need a pass-through subclass.

## Execution Model

The runner owns the loop:

```text
Runner
  -> session.next_step_requirements()
  -> IOHandler.next_window(requirements)
  -> ModelInputProvider.prepare_step(...)
  -> ApplicationSession.step(model_input)
  -> IOHandler.emit_chunk(result)
  -> metrics
```

This is the important boundary: the model session produces a `StepResult` and
returns it. Output delivery, backpressure, transport state, metrics, and cleanup
stay outside the model session.

For a direct `Application` that is not backed by a `DemoAdapter`, the default
input provider passes this step payload to the session:

```python
InferenceInput(
    step={
        "step_index": requirements.step_index,
        "user_window": user_window,
    },
    metadata=requirements.metadata,
)
```

For model-specific conditioning, prefer a `DemoAdapter` with
`create_model_input_provider(...)`. That keeps IO transport details out of the
model.

## Output Modes

Replay-style runs use `create_replay_io_handler(...)`.

```python
from pathlib import Path

from flashdreams.demo import (
    DemoAdapterApplication,
    FileOutputSink,
    Runner,
    create_replay_io_handler,
)
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, Mp4OutputSpec

spec = DemoSpec(
    model_id="my-demo",
    input_mode="replay",
    output=Mp4OutputSpec(path=Path("outputs/my-demo.mp4"), fps=30),
    config=InferenceConfig(model_id="my-demo", device="cuda"),
)

io_handler = create_replay_io_handler(
    output_sink=FileOutputSink(output_path=Path("outputs/my-demo.mp4"), fps=30)
)
app = DemoAdapterApplication(adapter=MyDemoAdapter(), spec=spec)
result = Runner(io_handler=io_handler, app=app).run()
```

For a null/headless run, omit `output_sink` or pass a null output spec through
your command app:

```python
io_handler = create_replay_io_handler()
result = Runner(io_handler=io_handler, app=app).run()
```

For metrics artifacts, add a metric tail:

```python
from flashdreams.demo import BenchmarkStatsOutputSink

io_handler = create_replay_io_handler(
    output_sink=FileOutputSink(output_path=Path("outputs/demo.mp4"), fps=30),
    metric_output_sink=BenchmarkStatsOutputSink(Path("outputs/demo-stats.json")),
)
```

For deterministic CI checks, use `ComparisonOutputSink` as a tail and feed it
the expected `StepResult` sequence.

## WebRTC Shape

WebRTC is intentionally server-shaped. A factory returns an `IOHandlerServer`,
not a ready `IOHandler`, because the transport edges only exist after a peer
connects.

During migration, demos that still own production WebRTC serving can adapt that
serve function with `CallbackIOHandlerServer`:

```python
from flashdreams.demo import CallbackIOHandlerServer


def webrtc_io_handler(args, *, context):
    def serve():
        return serve_my_webrtc_demo(
            spec=build_webrtc_spec(args, device=str(context.device)),
            world_rank=context.world_rank,
        )

    return CallbackIOHandlerServer(serve)
```

Once the transport has a real shared IO handler, the server calls the same
runner callback used by replay:

```python
server.serve(lambda handler: Runner(io_handler=handler, app=app).run())
```

Do not add per-demo WebRTC managers, offer handlers, generation workers, or
WebRTC-specific runtime wrappers for new demos. Put model-specific behavior in
the adapter, provider, runtime, session, or WebRTC UI resources.

## Pull-Based Input

`IOHandler.get_user_input_state(modality, name)` exposes named input state as a
view over the current `UserInputWindow`. This keeps replay deterministic: a
state query and the window used for the same step always agree.

Built-in names are:

```python
from flashdreams.demo import InputName

InputName.KEYBOARD
InputName.MOUSE_POSITION
InputName.MOUSE_BUTTON
InputName.HEAD_POSITION
InputName.HAND_POSITION
```

Keyboard state returns `KeyboardInputState`:

```python
state = io_handler.get_user_input_state("keyboard", InputName.KEYBOARD)
if state is not None and state.is_pressed("w"):
    ...
```

Legacy key probes such as `"key_w"` are also supported:

```python
is_forward = io_handler.get_user_input_state("keyboard", "key_w")
```

For model input construction, prefer consuming the `UserInputWindow` in a
`ModelInputProvider`. The pull API is most useful for author-facing IO handlers,
tests, and simple interactive state checks.

## Command App Helper

For an integration CLI, define parser/spec/adapter functions and bind them with
`create_demo_application(...)`.

```python
import argparse
from pathlib import Path

from flashdreams.demo import create_demo_application
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, Mp4OutputSpec, NullOutputSpec


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--device", default="cuda")
    replay.add_argument("--output-mode", choices=("mp4", "null"), default="mp4")
    replay.add_argument("--output", type=Path)

    webrtc = subparsers.add_parser("webrtc")
    webrtc.add_argument("--device", default="cuda:0")
    webrtc.add_argument("--host", default="0.0.0.0")
    webrtc.add_argument("--port", type=int, default=8080)

    return parser.parse_args(argv)


def replay_spec(args):
    output = (
        NullOutputSpec()
        if args.output_mode == "null"
        else Mp4OutputSpec(path=args.output, fps=30)
    )
    return DemoSpec(
        model_id="my-demo",
        input_mode="replay",
        output=output,
        config=InferenceConfig(model_id="my-demo", device=args.device),
    )


APPLICATION = create_demo_application(
    parse_args=parse_args,
    replay_spec=replay_spec,
    replay_adapter=MyDemoAdapter,
    webrtc_io_handler=webrtc_io_handler,
)


def main(argv=None):
    APPLICATION.main(argv)
```

This keeps the only command branch at IO factory selection. Replay and WebRTC
both end up constructing the same public `Runner`.

## Application Discovery

Packages can expose public demo applications through the
`flashdreams.applications` entry-point group:

```toml
[project.entry-points."flashdreams.applications"]
my-demo = "my_demo.app:create_app"
```

The referenced function should return an object satisfying `Application`:

```python
from flashdreams.demo import Application, DemoAdapterApplication


def create_app() -> Application:
    return DemoAdapterApplication(adapter=MyDemoAdapter(), spec=default_spec())
```

Use `flashdreams.plugins.discover_applications()` to load installed demo
applications.

## Adding A Demo

1. Implement or adapt your model runtime.
   - Existing runtime integrations should usually implement `DemoAdapter`.
   - Direct demos can implement `Application` and `ApplicationSession`.
2. Keep one model step as `step(InferenceInput) -> StepResult`.
   - Do not pass sinks into the session.
   - Do not call output sinks from the session.
   - Do not sleep for consumer backpressure inside the session.
3. Build a `DemoSpec` for each command mode.
   - Put model and scenario settings in the spec.
   - Keep output settings in `Mp4OutputSpec`, `NullOutputSpec`, or
     `WebRTCOutputSpec`.
4. Use `create_replay_io_handler(...)` for MP4/null/replay.
5. Return an `IOHandlerServer` for WebRTC.
6. Run everything through `Runner`.
7. Add fake-model CPU tests before GPU tests.

## Current Migration Notes

- `create_native_window_io_handler(...)` is public, but native-window runtime
  wiring is still a later migration step.
- `CallbackIOHandlerServer` exists to keep migrated demos working while
  production WebRTC transports finish moving behind shared IO handlers.
- `DemoAdapterApplication` is the preferred bridge for existing runtime demo
  adapters. It lets new public APIs land without rewriting every model runtime.
- Lower-level modules under `flashdreams.runtime.demo` remain the source of
  truth for drivers, run modes, output sinks, metrics, and session assembly.
