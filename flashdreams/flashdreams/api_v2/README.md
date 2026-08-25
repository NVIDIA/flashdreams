<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Protocols for the FlashDreams v2 API.

- `application.py`: `IApplication` loads shared state and creates sessions.
- `session.py`: `ISession` represents one session and owns a model loop and a UI loop.
- `thread.py`: `ILoop` holds shared loop state and messaging;
  `IModelLoop` and `IUILoop` define model and UI work.
- `input_source.py` / `output_sink.py` / `client_window.py`: `IClientWindow`
  groups one client's input and output.
- `user_input_event_data.py`: base class for input event data.

Running an application
----------------------

`flashdreams-run-v2` finds an application through the
`flashdreams.applications_v2` entry point and runs it with `ApplicationRunner`.
The application chooses its default `SessionDesc`; `--pixel-width`,
`--pixel-height`, `--fps`, `--layout`, `--backpressure-mode`, and
`--presentation-mode` can override it.

The selected client-window mode handles the run's input and output. MP4 mode
writes a file and has no input. WebRTC mode streams to a browser. Mode-specific
arguments are defined in `runtime_v2/client_window_factory.py`.

Model generation stops when the client closes, the model loop reports that it
is finished, or a model-loop step limit is reached. `run_session` presents
any queued frames before returning. An MP4 window never sends a close event, so
its model loop must finish on its own.

`--stats-path` adds a `MetricsOutputSink`. It writes measurements provided in a
`StepResult` returned via `step`. The client window still receives only the UI
loop's output. Metrics collection does not change either presentation setting.
For equality evaluations, use `PresentationMode.ONLY_PRESENT_NEW` and
`BackpressureMode.BLOCK` so every model frame is presented exactly once and in
order.

See [`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
and the [benchmark README](../../tools/benchmarks/README.md) for the benchmark
suite.

The `apps/t2v` package builds the text-to-video application on these protocols. See
[its README](../t2v_v2/README.md).

Loops and threading
-------------------

An application lives for the length of the parent process. A
session (stored in `IApplication`) lives for one run and owns two `ILoop`
objects: an `IModelLoop` and an `IUILoop`. In `ISession.init`, it must call
`register_model_loop` and may call `register_ui_loop`.

| Runs on | Calls | Owns | Frame rate |
| --- | --- | --- | --- |
| The thread that called `run_session` | `IUILoop.step` | UI-loop state, `run_session` state | `frames_per_second_for_ui` |
| A new Python thread | `IModelLoop.step` | Model-loop state and model logic | `frames_per_second_for_step` |

## Using the loop model to build your own application

### Loop-to-loop communication

Each `ILoop` has mutable `state` when the session registers it. The
registration call (`register_model_loop` or `register_ui_loop`) returns the new
loop object. The only way one loop should change another loop's state is via
`invoke_async`:

```python
new_prompt = str(text_from_ui)
invoke_async(
    self.state.model_loop,
    lambda state, new_prompt=new_prompt: state.set_prompt(new_prompt),
)
```

The call returns immediately, sending the operation to the target loop's message
queue (`self.state.model_loop`). The target loop runs the operation in its next
`step`. It takes a snapshot of queued operations first and only processes this
snapshot until the next `step` is complete. Operations from `invoke_async` must
return `None`. Queued operations that have not run are dropped during shutdown
to prevent endless ping-ponging between loops.

### Loop reset

A reset event sent by an application calls `reset` on each loop, clears its
`latest_result`, and restarts the loop's `step_index` at zero.

### Loop output

All output from your model loop's `step` method returns a list of
`StepResult` channels. The metrics inside this `StepResult` are recorded immediately by a metrics output sink; the actual full `StepResult` is passed along to a presentation manager.

The UI loop pulls from the presentation manager and sends results to the client
window. `presented_model_frame` or `presented_model_frames` can be used to draw
the model output.

Import `SlangPyUILoop` from `flashdreams.runtime_v2.slangpy_ui_loop`, subclass
it, and implement `step_ui(ui, step_index, events)`. The `ui` argument exposes:

- `ui.screen`, the root [`slangpy.ui.Screen`](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#slangpy.ui.Screen)
  that receives top-level widgets.
- Every public type from `slangpy.ui`, including widget constructors such as
  `Window`, `Group`, `Text`, `Button`, `ComboBox`, sliders, drag controls, and
  input controls.

The [SlangPy UI API reference](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui)
is the source of truth for every available widget constructor, method,
property, flag, and callback. FlashDreams delegates these names directly to
`slangpy.ui`; it does not maintain a smaller wrapper API. See the
[`slangpy_ui_demo` examples](../../../apps/slangpy_ui_demo/README.md)
for examples that use model-loop output as part of the UI.

If a UI loop is not registered, the runtime uses the default `IUILoop`
implementation
(`blit_model_output_to_screen_loop.py:BlitModelOutputToScreenLoop`).
It blits the model output to the screen, flattening channels into one frame as
if they were image layers.

This is a minimal session using the default UI-loop implementation:

```python

@dataclass
class ModelState:
    desc: SessionDesc

class ModelLoop(IModelLoop[ModelState]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        del events
        frame = torch.zeros(
            (1, 3, self.state.desc.video_height, self.state.desc.video_width)
        )
        return [
            StepResult(
                step_index=step_index,
                output=frame,
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
            )
        ]

    def reset(self) -> None:
        pass

class Session(ISession):
    def __init__(self, desc: SessionDesc) -> None:
        if desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("This session requires tchw output.")
        self._desc = desc

    @property
    def session_desc(self) -> SessionDesc:
        return self._desc

    def init(self) -> None:
        self.register_model_loop(ModelLoop, state=ModelState(self._desc))
```

`ILoop.is_finished` returns `False` by default. Override it when the model
should end the run on its own, as an MP4-producing model may set if not limiting the maximum number of steps.

### Presentation knobs for loops and benchmarking
---------

The runtime buffers completed model steps and presents at most one frame per UI
tick. Two independent `SessionDesc` settings control mismatched model and UI
rates.

`SessionDesc.backpressure_mode` handles a model thread producing frames faster
than the UI thread can consume them:

- `BackpressureMode.BLOCK` waits when the presentation queue is full. This keeps
  every generated frame and can slow the model thread to the UI thread's pace.
- `BackpressureMode.DROP_OLDEST` discards old buffered work so the UI can catch
  up to newer output. This favors low latency over preserving every frame.

`SessionDesc.presentation_mode` handles the UI thread ticking faster than the
model thread produces frames:

- `PresentationMode.ONLY_PRESENT_NEWEST` is eager: the UI runs every tick and
  may present the newest generated model frame more-than-once when a new frame is not ready.
- `PresentationMode.ONLY_PRESENT_NEW` is safe: the UI runs only after the
  presentation manager advances to a new model frame, preventing duplicate
  output frames.

For equality evaluations, enable `PresentationMode.ONLY_PRESENT_NEW`
with `BackpressureMode.BLOCK`. Together they preserve all generated frames and
present each one exactly once and in order.

Output sinks read floating-point frames as `[-1, 1]` and integer frames as
`[0, 255]`. This is not remappable via a `SessionDesc` setting; the UI loop
should implement its own remapping logic.


Not built yet
-------------

- A third fixed-rate thread for game logic.
- A result field for the number of dropped frames.
- Slowing model generation before the presentation buffer fills.
