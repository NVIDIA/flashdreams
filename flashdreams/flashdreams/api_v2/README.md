<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

The protocols a FlashDreams v2 application implements. Everything here is a
contract; the runtime creates every other object and passes it in.

[ARCHITECTURE.md](../../../ARCHITECTURE.md) covers how an application, a session
and the runtime fit together. [`runtime_v2`](../runtime_v2/README.md) covers what
runs one, including the command line. This page is only what you implement, and
what each contract promises.

## What is in here

- `application.py`: `IApplication` parses arguments, holds what its sessions
  share, and creates them.
- `session.py`: `ISession` is one run, and registers the loops that do its work.
- `loop.py`: `ILoop` holds per-loop state, messaging and lifecycle;
  `IModelLoop` generates, `IUILoop` presents.
- `input_source.py`, `output_sink.py`, `client_window.py`: `IClientWindow` is
  both an `InputSource` and an `OutputSink`, grouping one client's input and
  output.
- `user_input_event.py`: base class for timestamped input events. The concrete
  event types belong to the runtime.

Three of these are things you write: an application, a session, and a model
loop. A UI loop is optional. Windows and sinks you only implement if you are
adding a new way to watch a run.

## `IApplication`

Lives as long as the process. It parses its own arguments in `init`, loads
whatever its sessions share, and creates them one at a time. Anything expensive
belongs here, loaded once, and released in `close`.

It can also answer `session_desc` before `init` runs, so a caller can ask what
this application would generate without paying to start it. Returning `None`
means it will generate whatever it is asked for.

Reject a description you cannot honour, from `create_session`, rather than
generating something else instead.

## `ISession`

One run. It registers a model loop in `init` and may register a UI loop. A
session that registers no UI loop gets
`flashdreams.runtime_v2.blit_model_output_to_screen_loop.BlitModelOutputToScreenLoop`,
which draws every model channel into one frame as if they were image layers.

Each loop is registered with the state it owns, and the call returns the loop:

| Runs on | Calls | Owns | Frame rate |
| --- | --- | --- | --- |
| The UI thread | `IUILoop.step` | UI-loop state, `run_session` state | `frames_per_second_for_ui` |
| The model thread | `IModelLoop.step` | Model-loop state and model logic | `frames_per_second_for_step` |

```python
self.register_model_loop(ModelLoop, state=ModelState(self._desc))
```

`state` is required for a model loop and optional for a UI loop. Each loop's rate
comes from the session description: the model loop steps at
`frames_per_second_for_step`, and the UI ticks at `frames_per_second_for_ui`.

The UI thread initially selects frames from model chunks at
`frames_per_second_for_step`. With nonblocking `BackpressureMode.DROP_OLDEST` backpressure, the oldest chunk not-finished being processed by the ui-thread will be discarded in favor of a new chunk returned by the model-thread if presentation-manager buffer is full. With `BackpressureMode.BLOCK` backpressure, instead of discarding a chunk the model-thread will wait for an open chunk-slot to store its result in the presentation-manager before progressing to its next `step`. The goal of this backpressure model is to allow independent computation and presentation of UI (for reactivity to user inputs) separate from the backend logic of the model-thread.```
`PresentationMode.CONTINUOUS` lets an `IUILoop` redraw every UI tick;
`PresentationMode.ON_DEMAND` runs it only when the selected model frame changes.
Interactive or clock-driven UIs should use continuous presentation.

## Loops

The two loops run on different threads, so neither should reach into the other's
state directly. `invoke_async` is the way across:

```python
new_prompt = str(text_from_ui)
invoke_async(
    self.state.model_loop,
    lambda state, new_prompt=new_prompt: state.set_prompt(new_prompt),
)
```

The call returns immediately and queues the operation against the target loop.
That loop takes a snapshot of its queue before its next `step` and runs only
what was in it. Operations must return `None`. Anything still queued at shutdown
is dropped, so two loops cannot keep each other alive by messaging back and
forth.

`ILoop.is_finished` returns `False` by default. Override it when the model should
end the run on its own, which is what a run writing an MP4 depends on, an MP4
window never sends a close event, so nothing else will stop it.

`ILoop.reset` raises `NotImplementedError` by default. A reset arrives as a
client event, and when one does, every loop's `reset` is called, its
`latest_result` is cleared, and the `step_index` handed to `step` starts again
at zero. A loop that does not override `reset` therefore fails the first time a
client asks for one, implement it, even if the body is `return`.

## What a step returns

The two loops have different return contracts, and the runtime enforces both:

- A model loop returns `list[StepResult]`, one entry per channel. A single
  `StepResult` or `None` raises `TypeError`.
- A UI loop returns one `StepResult`, or `None` to present nothing this tick.

Every channel in one model step must report the same `frame_count`, and a
mismatch raises `ValueError`. A step may generate several frames at once; the
runtime presents them one per UI tick rather than dropping all but the last.

A UI loop reads what the model produced through `presented_model_frame` and
`presented_model_frames`, which return `[C, H, W]` frames with one, three or
four channels. Four channels is RGBA, and composites over what is beneath it.

Output sinks read floating-point frames as `[-1, 1]` and integer frames as
`[0, 255]`. No `SessionDesc` setting remaps this; a UI loop that works in some
other range converts before returning.

CUDA results may cross from a producer stream to a dedicated presentation or
transfer stream. Constructing a `StepResult` for CUDA output automatically
records readiness on the current stream, so construct the result while the
actual producer stream is current. The presentation manager and runtime sinks
call `StepResult.read_output()` while their consumer stream is current.
When CUDA is available, `run_session` uses one highest-priority CUDA stream by
default for the complete UI-thread lifecycle. An explicit CPU presentation
manager opts out.

## A minimal application

Using the default UI loop, so there is only a model loop to write:

```python
class ModelLoop(IModelLoop[SessionDesc]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        del events
        frame = torch.zeros(
            (1, 3, self.state.video_height, self.state.video_width)
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
        return

class Session(ISession):
    def __init__(self, desc: SessionDesc) -> None:
        if desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("This session requires tchw output.")
        self._desc = desc

    @property
    def session_desc(self) -> SessionDesc:
        return self._desc

    def init(self) -> None:
        self.register_model_loop(ModelLoop, state=self._desc)
```

A loop's state can be any object; this one keeps the session description and
nothing else.

This loop runs forever. Override `is_finished` to make it stop.

## Writing a UI loop

`SessionDesc.backpressure_mode` handles the model-generation-loop producing
frames faster than the UI thread can consume them:

- `BackpressureMode.BLOCK` waits when the presentation queue is full. This keeps
  every generated frame and can slow the model thread to the UI thread's pace.
- `BackpressureMode.DROP_OLDEST` discards old buffered work so the UI can catch
  up to newer output. This favors low latency over preserving every frame.

`SessionDesc.presentation_mode` handles the UI loop ticking faster than the
model-generation-loop produces frames:

- `PresentationMode.CONTINUOUS` runs the UI every tick and may reuse
  the newest generated model frame.
- `PresentationMode.ON_DEMAND` runs the UI after the presentation manager
  advances to a new model frame.

Use `PresentationMode.ON_DEMAND` with `BackpressureMode.BLOCK` when every
generated model frame must be selected and written exactly once in order.

For full immediate Dear ImGui controls drawn over model output, subclass
`ImGuiUILoop` from `flashdreams.runtime_v2.imgui_ui_loop` and implement
`step_ui(imgui, step_index, events)` rather than `step`. Its `imgui` proxy
exposes `imgui_bundle.imgui` and an image-like pixel upload convenience form.
For SlangPy's smaller retained widget API, subclass `SlangPyUILoop` from
`flashdreams.runtime_v2.slangpy_ui_loop`. The
[`slangpy_ui_demo` integration](../../../integrations_v2/slangpy_ui_demo/README.md)
remains the reference for that retained API.

## Where to go next

- [ARCHITECTURE.md](../../../ARCHITECTURE.md) - how the layers fit together.
- [Runtime](../runtime_v2/README.md) - what runs an application, and the command
  line that does it.
- [Writing an integration](../../../integrations_v2/README.md) - the checklist
  for a new application.
- [`flashdreams.t2v_v2`](../t2v_v2/README.md) - the text-to-video API built on
  these protocols, and how to add a model to it.
