<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Protocols for the FlashDreams v2 API.

- `application.py`: `IApplication` loads shared state and creates sessions.
- `session.py`: `ISession` represents one session and runs via two threads: model-thread and ui-thread.
- `slangpy_ui_thread.py`: `SlangPyUIThread` is the user-facing SlangPy UI
  implementation.
- `thread.py`: `IThread` and `UIThread` hold per-thread state and work.
- `input_source.py` / `output_sink.py` / `client_window.py`: `IClientWindow`
  groups one client's input and output.
- `user_input_event_data.py`: base class for input event data.

Running an application
----------------------

`flashdreams-run-v2` finds an application through the
`flashdreams.applications_v2` entry point and runs it with `ApplicationRunner`.
The application chooses its default `SessionDesc`; `--pixel-width`,
`--pixel-height`, `--fps`, and `--layout` can override it.

The selected client-window mode handles the run's input and output. MP4 mode
writes a file and has no input. WebRTC mode streams to a browser. Mode-specific
arguments are defined in `runtime_v2/client_window_factory.py`.

Model generation stops when the client closes, the model-thread reports that it
is finished, or a model-thread step limit is reached. `run_session` presents
any queued frames before returning. An MP4 window never sends a close event, so
its model-thread must finish on its own.

`--stats-path` adds a `MetricsOutputSink`. Allows writing measurements as provided in a `StepResult` object returned via `step`. The client window still receives only the ui-thread's output. Metrics collection
does not change `SessionDesc.presentation_mode`, users must opt into `PresentationMode.LOSSLESS` if they want lossless presentation.

See [`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
and the [benchmark README](../../tools/benchmarks/README.md) for the benchmark
suite.

`flashdreams.t2v_v2` builds the text-to-video API on these protocols. See
[its README](../t2v_v2/README.md).

Threading
---------

An application lives for the length of the parent process. A
session (stored in `IApplication`) lives for one run and owns two `IThread` objects, a model-thread and a ui-thread. In `ISession.init`,
it must call `register_model_thread` and may call `register_ui_thread`.

| Runs on | Calls | Owns | Frame rate |
| --- | --- | --- | --- |
| The thread that called `run_session` | `UIThread.step_ui` | UI state, `run_session` state | `frames_per_second_for_ui` |
| A new Python thread | `IThread.step` for the model-thread | Model-thread state and model-thread logic | `frames_per_second_for_step` |

## Using our threading model to build your own application

### Thread to Thread Communication

Each `IThread` has a mutable `state` when the session registers it. The
registration call (`register_model_thread` or `register_ui_thread`) returns the new thread object. The only way threads should be changing state of another thread is via `invoke_async`:

```python
new_prompt = str(text_from_ui)
invoke_async(
    self.state.model_thread,
    lambda state, new_prompt=new_prompt: state.set_prompt(new_prompt),
)
```

The call returns immediately, sending the operation to the target threads message queue (`self.state.model_thread`).
The target thread runs the operation in its next `step`. It takes a snapshot of queued operations first and only processes this snapshot until next `step` is completed. Operations from `invoke_async` must return `None`. Queued operations that have not run are dropped during shutdown since otherwise two threads could hit an endless loop of ping-ponging messages to each other.

### Thread Reset

A reset event send by an application calls `reset` on each thread,
clears its `latest_result`, and restarts a thread's `step_index` to zero.

### Thread Output

All output from your model-thread's `step` method returns a list of
`StepResult` channels. The metrics inside this `StepResult` are recorded immediately by a metrics output sink; the actual full `StepResult` is passed along to a presentation manager.

The ui-thread has the job of pulling from the presentation manager and sending the results to the client window. `presented_model_frame` or `presented_model_frames` can be used to draw the model output.

Import `SlangPyUIThread` from `flashdreams.api_v2.slangpy_ui_thread`, subclass
it, and implement `draw_ui(ui, step_index, events)`. The `ui` argument exposes:

- `ui.screen`, the root [`slangpy.ui.Screen`](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#slangpy.ui.Screen)
  that receives top-level widgets.
- Every public type from `slangpy.ui`, including widget constructors such as
  `Window`, `Group`, `Text`, `Button`, `ComboBox`, sliders, drag controls, and
  input controls.

The [SlangPy UI API reference](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui)
is the source of truth for every available widget constructor, method,
property, flag, and callback. FlashDreams delegates these names directly to
`slangpy.ui`; it does not maintain a smaller wrapper API. See the
[`slangpy_ui_demo` examples](../../../integrations_v2/slangpy_ui_demo/README.md)
for examples and utilizing model-thread outputs as a part of the UI.

If a ui-thread is not registered, the runtime uses a default `UIThread` implementation (`BlitModelOutputToScreenThread`) that simply blits the model output to the screen (flattening channels into a single frame as if they were image layers).

This is a minimal session using the default ui-thread implementation:

```python

@dataclass
class ModelState:
    desc: SessionDesc

class ModelThread(IThread[ModelState]):
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
        self.register_model_thread(ModelThread, state=ModelState(self._desc))
```

`IThread.is_finished` returns `False` by default. Override it when the model
should end the run on its own, as an MP4-producing model may set if not limiting the maximum number of steps.

### Presentation Knobs Of Threads & Benchmarking
---------

The runtime buffers completed model steps and presents one frame per UI tick.
`SessionDesc.presentation_mode` controls what happens when model generation and
the UI run at different speeds:

- `BLOCK` waits when the buffer is full. If no new frame is ready, the UI may
  draw the current frame again.
- `DROP_OLDEST` drops old buffered work so the UI can show newer output.
- `LOSSLESS` waits when the buffer is full and shows each model frame once. When
  the current frame is the last available frame, the UI loop waits for a new
  frame instead of drawing the old one again. This is important for quality-evaluation as it allows the benchmark to capture the full model output.

Output sinks read floating-point frames as `[-1, 1]` and integer frames as
`[0, 255]`. This is not remappable via a `SessionDesc` setting, ui-thread should implement its own remapping logic.


Not built yet
-------------

- A third fixed-rate thread for game logic.
- A result field for the number of dropped frames.
- Slowing model generation before the presentation buffer fills.
