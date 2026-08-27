<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

The machinery that runs a v2 application: it finds one, asks it for a session,
runs that session's two loops on two threads, and delivers what they generate to
a file or a browser.

This is the implementation. For how the pieces fit together and why the seams are
where they are, read [ARCHITECTURE.md](../../../ARCHITECTURE.md) first; for the
protocols an application implements, see [`api_v2`](../api_v2/README.md).

## What is in here

Finding and starting an application:

- `application_registry.py` reads the `flashdreams.applications_v2` entry points
  to turn a slug into an `IApplication`, falling back to importing the slug as a
  module name.
- `cli.py` is `flashdreams-run-v2`: it splits its own arguments from the
  application's at `--`, decides which session to ask for, and builds the window.
- `application_runner.py` owns the application lifecycle around one run —
  `init`, `create_session`, `run_session`, `close`.
- `client_window_factory.py` turns `--mode` into a window, and owns the
  arguments each mode takes.

Running a session:

- `session_runner.py` is `run_session`, which everything above ends up calling.
  It is the only place that starts a generation thread; the WebRTC server and
  the video encoder run threads of their own, but neither touches a loop.
- `event_buffer.py` holds client input until both loops have read it.
- `presentation_manager.py` buffers generated frames between the two threads and
  decides which one the UI sees.
- `session_desc.py` describes the session being run: frame size, rates, layout,
  and the two policy knobs below.
- `step_result.py` is what one generation step produces.

Presenting it:

- `blit_model_output_to_screen_loop.py` is the UI loop a session gets when it
  registers none of its own.
- `slangpy_ui_loop.py` and `slangpy_ui_renderer.py` are the UI loop for
  applications that draw widgets over the model output.
- `mp4_client_window.py` and `webrtc_client_window.py` are the two windows.
- `mp4_output_sink.py`, `metrics_output_sink.py`, `video_encoder.py` and
  `video_tensor.py` are what output is written through.
- `serving/` is the HTTP and WebRTC server behind the browser window.

Input:

- `user_input_event.py` defines the concrete event types, `user_input_events.py` the
  batch of them a source hands over.

## The command line

`flashdreams-run-v2 SLUG` finds an application through the
`flashdreams.applications_v2` entry point group and runs it with
`ApplicationRunner`. Arguments after `--` go to the application, so
`flashdreams-run-v2 SLUG -- --help` describes the application rather than this
command. The split is stated rather than guessed because an application may
declare arguments this command also has.

`--mode` picks where the run goes, and each mode owns its own arguments in
`client_window_factory.py`:

| Mode | Takes | Input | Ends when |
| --- | --- | --- | --- |
| `mp4` (default) | `--output-path` | none | the session reports itself finished |
| `webrtc` | `--host`, `--port` | keyboard, mouse, focus, reset, close | the browser disconnects, or the session finishes |

These override whatever session the application asked for:

| Argument | Sets |
| --- | --- |
| `--pixel-width`, `--pixel-height` | Frame size to generate. |
| `--fps` | `frames_per_second_for_step`, which is also the rate an MP4 plays back at. |
| `--layout` | Tensor layout to generate results in. |
| `--backpressure-mode` | What the model thread does when the presentation queue is full. |
| `--presentation-mode` | Whether the UI runs continuously or on demand for new frames and requested input refreshes. |

Each defaults to asking for nothing, so a run that names none of them gets what
the application generates. There is no argument for the UI tick rate, and none
for `run_session`'s `steps` limit — a caller that needs to bound a run by steps
drives the runtime from Python.

`--stats-path` adds a `MetricsOutputSink`. It receives the **model** loop's
results as they are published, not the UI loop's output, so a benchmark measures
what the model generated while the window still sees one composited frame per
tick. See [`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
and the [benchmark README](../../tools/benchmarks/README.md) for the suite.
The runtime augments those model records with step wall time and
presentation-queue depth/publish-wait measurements under the reserved
`runtime_` metric prefix. UI and window timings are not folded into a later
model record because they describe a different frame.

## Starting and stopping a run

`ApplicationRunner.run` calls `init`, `create_session` and `run_session` in
order, and closes the application on the way out whether or not the run
succeeded. It also closes the window itself when the run never started, because
`run_session` is what otherwise owns the window, and a WebRTC window may already
be serving a browser before the application has finished loading.

`run_session` then opens the window and any metrics sink, collects one batch of
input, presents one tick, and only then starts the model thread — so a client that
closed during startup is never generated for. Its main loop services input and
presents frames until the shutdown event is set, or until the model thread has
finished and no frames are still pending.

On the way out it sets the shutdown event, joins the model thread, shuts both
loops down, clears the presentation buffer, unregisters the readers, closes every
sink it opened, and closes the session. Then it raises a loop's failure if one
was queued, otherwise its own, otherwise the first cleanup failure, logging the
rest.

## `EventBuffer`

Input arrives once per tick on the UI thread and is appended here. The buffer
keeps a flat list plus a cursor per registered reader: `read` returns everything
that reader has not seen and moves its cursor to the end, and
`collect_garbage` deletes the prefix every reader has passed. The UI loop is
reader 0 and the model loop is reader 1.

Appending also counts resets. Every `ResetUserInputEvent` in a batch bumps
the generation number that loops and the presentation manager compare against
their own.

A close event is handled twice over, deliberately: `run_session` sets the
shutdown event when it sees one in a collected batch, and `ILoop._begin_run` sets
it again when a loop reads one. Either path alone would end the run; both means
neither thread waits on the other to notice.

## `PresentationManager`

The model thread publishes a list of channels per step into a bounded queue
(`max_pending`, two by default). The UI thread calls `advance` once per tick,
which walks the frames within the chunk it is already showing before taking
another off the queue.

Frame cadence initially uses `frames_per_second_for_step`, then follows model
frame completions observed over the trailing two seconds. A late UI tick
reanchors the next deadline; it never drains multiple model frames into
back-to-back writes in one tick.

`SessionDesc.backpressure_mode` decides what publishing does when the queue is
full:

- `BLOCK` waits for room, in `put_timeout` slices so a shutdown is still
  noticed. Every generated frame survives, and the model thread is held back to
  the rate the UI can consume.
- `DROP_OLDEST` evicts queued work so the UI can catch up to the newest output,
  trading frames for latency.

`SessionDesc.presentation_mode` decides what the UI does when no new frame is
ready:

- `CONTINUOUS` runs the UI loop every tick, re-presenting the newest
  model frame.
- `ON_DEMAND` runs the UI loop when `advance` moves to a new frame, or when the
  UI requests an input-driven refresh through `should_redraw_for_input`. The
  default hook requests a refresh for every non-empty input batch; UI loops may
  override it to filter events. Time-driven status UIs may request an idle tick
  through `should_redraw_on_idle()` without enabling continuous output.

For output that has to be compared frame by frame, use `BLOCK` with
`ON_DEMAND`: together they keep every frame in the presentation manager
and present each in order. When no input refresh is requested, each is written
exactly once; an input refresh may intentionally re-present the current model
frame. Steps that could not be kept are counted in `dropped_for_space` and
`discarded_at_reset`, and logged when the run ends. Both count model steps
rather than frames, so one step of twelve frames counts once.

## Presenting and writing

A UI loop reads model frames through `presented_model_frame` and
`presented_model_frames`, composites whatever it wants, and returns one
`StepResult` that `run_session` writes to the window.

The default UI loop, `BlitModelOutputToScreenLoop`, composites every model
channel in list order as if they were image layers and reshapes the result into
the session's layout. `SlangPyUILoop` is the alternative, for widgets drawn over
the model output; it returns a `[1, C, H, W]` frame, so a session using it
declares a `tchw` layout.

`IClientWindow` is both an `InputSource` and an `OutputSink`, so a window is
written to with the same three calls as any sink: `open` with the session
description, `write` per result, `close` at the end. Both windows are thin
delegates over the thing that does the work — `Mp4ClientWindow` over
`Mp4OutputSink`, `WebRTCClientWindow` over `WebRTCServer` — and neither describes
the output shape, because the session already did.

They differ in what they can do rather than in how they are driven. The MP4
window reports no input and never sends a close, so a session written to a file
has to finish on its own. The WebRTC window turns browser keyboard, mouse and
focus events into input and a disconnecting browser into a close; because those
arrive on the server's own thread, it queues them and hands them over in batches
when the session asks.

The UI thread owns WebRTC cadence. Each `write` synchronously materializes one
owned video frame and admits it to a two-frame FIFO of unsent frames. The WebRTC
track returns queued frames to aiortc immediately: it neither sleeps to pace
them nor repeats the latest frame. If network or encoder congestion fills the
FIFO, the next write replaces only its oldest unsent frame; a frame already
handed to aiortc is never overwritten.

What a sink expects of the pixel values it is handed is part of the result
contract, in [`api_v2`](../api_v2/README.md#what-a-step-returns).

## Adding a mode

A mode is one way to watch a run. Subclass `ClientWindowMode` in
`client_window_factory.py`, declare the arguments only it takes, and add it to
`_MODES`; a command line offering the modes never has to know what any of them
are. `check_arguments` is called while the command can still print usage, and
`starting` and `finished` are what it prints — which is how a WebRTC run reports
a URL nobody could have guessed when the port was chosen by the operating system.
