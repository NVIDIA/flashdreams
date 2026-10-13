---
title: 'Demo Application API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This page documents the public application protocols in `flashdreams.api_v2`
and the runtime-owned value types those protocols exchange. For implementation
workflows, start with [Create a demo](../guides/create_demo.md) or
[Integrate a model with a demo](../guides/integrate_model.md).

## Application and session

### `flashdreams.api_v2.application.IApplication`

One application, for as long as the process runs.

Parses its own arguments and holds whatever its sessions share, such as a
checkpoint or a compiled pipeline. It outlives every session it creates, so
that shared state is loaded once here and released in `close`.

An application module implements this and `ISession`. The runtime
creates everything else and passes it in.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/api_v2/application.py#L14)

#### `init`

```python
def init(self, commandline_args: Sequence[str]) -> None
```

Parse application arguments and validate startup state.

#### `session_desc`

```python
def session_desc(self) -> SessionDesc | None
```

Return the description of a session this application would generate.

A caller has to describe a session before there is one to describe, and
only the application knows what its model was trained for. Asked before
`init`, so describing a session costs nothing.

**Returns**

The session to create when nobody asks for another, or `None`,
the default, from an application that generates whatever it is
asked for. Its caller describes the session instead.

#### `create_session`

```python
def create_session(self, session_desc: SessionDesc) -> ISession
```

Create one isolated, uninitialized session for `session_desc`.

**Parameters**

- `session_desc`: Session the runtime is asking for.

**Returns**

A session for `session_desc`, resolved to what this application can
actually produce.

**Raises**

- `ValueError`: The application cannot honour `session_desc`.

#### `close`

```python
def close(self) -> None
```

Release whatever the application holds.

Not abstract, and does nothing by default, so an application with nothing
to release does not implement it.

### `flashdreams.api_v2.session.ISession`

One application run with a model loop and a UI loop.

Register the model loop in `init`. If no UI loop is registered,
the runtime uses `BlitModelOutputToScreenLoop`.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/api_v2/session.py#L21)

- `_registered_ui_loop: IUILoop[Any] | None`

- `_registered_model_loop: IModelLoop[Any] | None`

#### `parallel_context`

```python
def parallel_context(self) -> ParallelContext | None
```

Return the model mesh before initialization, or `None` for a local session.

The runtime synchronizes model steps and inputs across this world.
Only rank zero owns a client window and output sinks.

#### `init`

```python
def init(self) -> None
```

Initialize state and register this session's loops.

A model loop is required. Registering a UI loop is optional; without one
the runtime uses `BlitModelOutputToScreenLoop`.

**Raises**

- `RuntimeError`: No model loop was registered by the time the runtime
    asks for the loops.

#### `session_desc`

```python
def session_desc(self) -> SessionDesc
```

Return the description used to configure the runtime.

#### `register_ui_loop`

```python
def register_ui_loop(self, loop_type: type[IUILoop[Any]], *, state: Any=None, **kwargs: Any) -> IUILoop[Any]
```

Create and register the UI loop, and return it.

Omit this call to use the default UI. The loop is paced at the session's
`frames_per_second_for_ui`.

**Parameters**

- `loop_type`: UI loop class to instantiate.
- `state`: State the loop owns. Unlike a model loop, a UI loop is
    allowed to hold none.
- `**kwargs`: Passed to `loop_type`.

**Raises**

- `RuntimeError`: The runtime already took the loops, or a UI loop was
    registered already.
- `TypeError`: `loop_type` does not derive from `IUILoop`.

#### `register_model_loop`

```python
def register_model_loop(self, loop_type: type[IModelLoop[Any]], *, state: Any, **kwargs: Any) -> IModelLoop[Any]
```

Create and register the model loop, and return it.

The loop is paced at the session's `frames_per_second_for_step`.

**Parameters**

- `loop_type`: Model loop class to instantiate.
- `state`: State the loop owns. Required, since a model loop with no
    state has nothing to generate from.
- `**kwargs`: Passed to `loop_type`.

**Raises**

- `RuntimeError`: The runtime already took the loops, or a model loop
    was registered already.
- `TypeError`: `loop_type` does not derive from `IModelLoop`.

#### `ui_loop`

```python
def ui_loop(self) -> IUILoop[Any]
```

Return the registered UI loop.

#### `model_loop`

```python
def model_loop(self) -> IModelLoop[Any]
```

Return the registered model-generation loop.

#### `close`

```python
def close(self) -> None
```

Release resources owned by the session.

## Model and UI loops

### `flashdreams.api_v2.loop.IModelLoop`

Loop that generates model results on the model thread.

`ILoop.step` must return `list[StepResult]` here, one entry per
channel, with every channel reporting the same `frame_count`. An empty
list means the step produced no presentable output.

Every admitted rank executes the same model step and ordered collectives.
The integration owns TP/CP partitioning and must gather all CP shards needed
for publication before the presenting rank returns non-empty results.
Workers return `[]` only after all required collectives, including gathers.
Gather before decoding when it needs the complete latent/view sequence;
decoding may run only on the presenting rank when the model permits it.
TP reductions that already replicate a complete value need no extra gather.

A run whose steps all present nothing still ends when the model loop does.
Returning a bare `StepResult` or `None` raises `TypeError`.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/api_v2/loop.py#L289)

#### `step`

```python
def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]
```

We expect IModelLoop's to return a list of StepResults, one result per channel.

#### `inference_state`

```python
def inference_state(self) -> ModelInferenceState
```

Return whether this model loop has started, is running, or finished.

### `flashdreams.api_v2.loop.IUILoop`

Loop whose output is sent to the client window.

`ILoop.step` must return `list[StepResult]` here: one frame to
present, or an empty list to present nothing this step. A list longer than
one raises `TypeError`. Model frames to draw come from
`presented_model_frame` and `presented_model_frames` rather
than from the model loop directly.
`has_pending_model_frames` and `presented_model_frame_count`
report whether more model frames are waiting and how many have been
selected.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/api_v2/loop.py#L424)

#### `step`

```python
def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]
```

Return zero or one `StepResult` to present this tick.

#### `register_session_ui_loop_objects`

```python
def register_session_ui_loop_objects(self, *, session_desc: SessionDesc, presentation_manager: PresentationManager) -> None
```

Store UI objects supplied when this loop is registered with a session.

**Parameters**

- `session_desc`: Description of the session this UI presents.
- `presentation_manager`: Buffer containing model frames.

#### `model_inference_state`

```python
def model_inference_state(self) -> ModelInferenceState
```

Return whether model inference has started, is running, or finished.

#### `request_new_session`

```python
def request_new_session(self, session_desc: SessionDesc) -> None
```

Ask the runtime to replace this session after the current UI step.

**Parameters**

- `session_desc`: Fully resolved description for the replacement session.

#### `request_hide_cursor`

```python
def request_hide_cursor(self, hide_cursor: bool) -> None
```

Request that the client window show or hide its cursor.

#### `request_lock_cursor_to_window`

```python
def request_lock_cursor_to_window(self, lock_cursor_to_window: bool) -> None
```

Request that the client window release or capture pointer motion.

#### `request_new_window_size`

```python
def request_new_window_size(self, new_window_size: tuple[int, int]) -> None
```

Request a positive downstream client-window width and height.

**Parameters**

- `new_window_size`: Requested `(width, height)` in pixels.

**Raises**

- `TypeError`: The size is not a pair of integers.
- `ValueError`: Either dimension is not positive.

#### `get_or_create_ui_loop_requests`

```python
def get_or_create_ui_loop_requests(self) -> UILoopRequests
```

#### `flush_ui_loop_requests`

```python
def flush_ui_loop_requests(self) -> UILoopRequests | None
```

return UI loop requests and clear the internal state of requests.

#### `presented_model_frame`

```python
def presented_model_frame(self, channel_index: int=0) -> Tensor | None
```

Return the current frame from one model-result channel.

**Parameters**

- `channel_index`: Channel to read, indexed as the model loop returned
    them.

**Returns**

A `[C, H, W]` frame with one, three or four channels, or `None`
before the first model result has been presented.

**Raises**

- `IndexError`: The presented result has no such channel.
- `ValueError`: The presentation stream is on a different CUDA device
    from the presented result.

#### `presented_model_frames`

```python
def presented_model_frames(self) -> tuple[Tensor, ...]
```

Return the current frame from every model-result channel.

**Returns**

One `[C, H, W]` frame per channel, bottom channel first, or an
empty tuple before the first model result has been presented.

**Raises**

- `ValueError`: The presentation stream is on a different CUDA device
    from a presented result.

#### `presented_model_frame_count`

```python
def presented_model_frame_count(self) -> int
```

Return how many model frames have been selected in this generation.

#### `has_pending_model_frames`

```python
def has_pending_model_frames(self) -> bool
```

Return whether another model frame is ready to present.

## Session description

### `flashdreams.runtime_v2.session_desc.SessionDesc`

Description of a session, passed to create one and to open a window on it.

The runtime fills this in to ask an application for a session, and the
session reports back what it resolved to. The same description then
configures the client window through `OutputSink.open`.

[View source](https://github.com/NVIDIA/flashdreams/blob/main/flashdreams/flashdreams/runtime_v2/session_desc.py#L35)

- `output_layout: VideoTensorLayout`

- `backpressure_mode: BackpressureMode`

- `presentation_mode: PresentationMode`

- `frames_per_second_for_ui: int`

- `frames_per_second_for_step: int`

- `video_width: int`

- `video_height: int`

- `metadata: dict[str, Any]`
