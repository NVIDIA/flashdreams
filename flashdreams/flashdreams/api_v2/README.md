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
- `user_input_event_data.py`: base class for input event data. The concrete
  types belong to the runtime.

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

```python
self.register_model_loop(ModelLoop, state=ModelState(self._desc))
```

`state` is required for a model loop and optional for a UI loop. Each loop's rate
comes from the session description: the model loop steps at
`frames_per_second_for_step`, and the UI ticks at `frames_per_second_for_ui`.

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

## A minimal application

Using the default UI loop, so there is only a model loop to write:

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
        self.register_model_loop(ModelLoop, state=ModelState(self._desc))
```

This loop runs forever. Override `is_finished` to make it stop.

## Writing a UI loop

For widgets drawn over the model output, subclass `SlangPyUILoop` from
`flashdreams.runtime_v2.slangpy_ui_loop` and implement
`step_ui(ui, step_index, events)` rather than `step`. The
[`slangpy_ui_demo` integration](../../../integrations_v2/slangpy_ui_demo/README.md)
is the reference, including one example that uses model output inside the UI.

## Where to go next

- [ARCHITECTURE.md](../../../ARCHITECTURE.md) - how the layers fit together.
- [Runtime](../runtime_v2/README.md) - what runs an application, and the command
  line that does it.
- [Writing an integration](../../../integrations_v2/README.md) - the checklist
  for a new application.
- [`flashdreams.t2v_v2`](../t2v_v2/README.md) - the text-to-video API built on
  these protocols, and how to add a model to it.
