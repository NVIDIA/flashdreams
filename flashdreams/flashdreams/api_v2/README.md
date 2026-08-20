<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Protocols for the FlashDreams API.

- `application.py` / `session.py`: `IApplication` creates an `ISession` from a
  `SessionDesc`, and the session reports what it resolved to. `session_desc`
  is the description the application would choose for itself, for a caller with
  none of its own.
- `input_source.py` / `output_sink.py` / `client_window.py`: `IClientWindow` is
  one client's input and output together. It is given the session's `SessionDesc`
  in `OutputSink.open`.
- `user_input_event_data.py`: base type for event payloads.

`flashdreams.runtime_v2.session_runner.run_session` drives a session against a
window until the session reports `is_finished` or the window reports a close, or
for a fixed number of steps a caller asks for. A caller holding an application
uses `flashdreams.runtime_v2.application_runner.ApplicationRunner` to get there,
which takes no step count: how long a run lasts is the application's business. A
run whose output is a file goes the same way, against
`flashdreams.runtime_v2.mp4_client_window.Mp4ClientWindow`, which reports no
input and encodes every result. Since it never reports a close, such a run needs
a session that finishes.

`flashdreams-run-v2` is that run from a shell: `flashdreams.runtime_v2.cli` finds
an application by slug, gives it the arguments after `--`, and hands it to
`ApplicationRunner` with the window `--mode` asked for, an MP4 file or a client
over WebRTC. Applications are found through the `flashdreams.applications_v2`
entry point group, or by the name of the package an integration ships when it
has registered nothing, which is
`flashdreams.runtime_v2.application_registry`'s job.

What the modes are belongs to
`flashdreams.runtime_v2.client_window_factory`, not to the command. A mode owns
the arguments only it takes, such as `--output-path` for a file or `--port` for
a browser, and what to say about where the run went: a URL to open before it
starts, or the file once there is something in it. So a new way of watching a
run is a mode added there, and the command is unchanged.

The session it asks for comes from `IApplication.session_desc`, with
`--pixel-width`, `--pixel-height`, `--fps`, and `--layout` overriding whatever
they name. That is the whole of what the command knows about the kind of
application it is running: a model answers with the clip its checkpoint was
trained for, and an application that generates whatever it is asked for answers
nothing and is described by those arguments alone.

`--stats-path` asks a run to record what it cost as well as what it generated.
`Mp4ClientWindow` takes that path and adds a `MetricsOutputSink` beside the MP4
writer, which records each step's measurements as the artifact
`flashdreams-benchmark` reads. The measurements are the model's own: a step reports what
it measured and this writes it down, converting milliseconds to seconds because
a report cannot compare two units. Nothing is measured unless a run asks, so an
ordinary run pays nothing for this.
[`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
is the suite that uses it, comparing every t2v model on one prompt and seed, and
[running it](../../tools/benchmarks/README.md) is written down beside the
harness.

`flashdreams.t2v_v2` is text-to-video on top of these protocols rather than part
of them: one `T2VApplication` owns the command line every t2v model needs, an
integration supplies only its own defaults, and `testing.check_t2v_model_impl`
is the check its tests run to cover the batch path in one call. See
[its README](../t2v_v2/README.md). The five `integrations_v2/t2v_*` packages are
the models behind it, and each is a factory of about forty lines.

Ownership
---------

Agreed design decisions. Change them by discussion.

- An application module implements `IApplication` and `ISession`. The runtime
  creates every other protocol here and passes it in.
- `IApplication` lasts as long as the process. It holds what its sessions share,
  such as a checkpoint or a compiled pipeline, and outlives every session it
  creates. It also says what session it would generate unasked, through
  `session_desc`, since only it knows what its model was trained for. The
  default says nothing, for an application that generates whatever it is asked
  for.
- `ISession` is one run: KV cache, game state, and anything else that must not
  carry into another run. It also says when that run is over, through
  `is_finished`. The default never finishes.
- `InputSource` and `OutputSink` belong to the runtime. The runner reads from the
  source and writes to the sink, so a session takes `UserInputEvents` in, returns
  a `StepResult`, and holds neither.
- `IClientWindow` pairs one client's input source and output sink. It is internal
  to the runtime, which is why it appears in no signature on `IApplication` or
  `ISession`. A window whose client disconnects reconnects itself rather than the
  runtime creating a second session.
- Application and session logic, including UI rendering, runs on the server side
  and is presented or streamed to a client window.
- The `UserInputEventData` types in `flashdreams.runtime_v2` cover the input
  modalities supported today, and integrations consume them. Nothing stops an
  integration subclassing the base class, and whether it should be able to is not
  settled, so this is a convention rather than something the code enforces.
- Ending and restarting a run are events on that same stream, not separate calls:
  a window reports `CloseUserInputEventData` when its client closes or goes away,
  and `ResetUserInputEventData` to start over. This is how native windowing
  systems deliver a close, ordered with the input around it, and `step_ui` is
  handed the batch it arrived in, so a session can react rather than just being
  abandoned.
- A reset does not split the input around it. The batch reaches the first step of
  the new generation whole, so a key held down when the client restarts is still
  held after, because it is the earlier edge that says so. A session that must
  not inherit that input ignores the older events itself.
- An `OutputSink` reads `StepResult.output` as one of two things: floats holding
  `[-1, 1]`, which is what FlashDreams models emit, or integers holding raw
  `0`-`255`. `SessionDesc` carries no range and a session cannot declare one, so
  this is a convention every sink follows.

Threading
---------

`run_session` uses two threads, and every window runs that way. Generation is on
the calling thread; the window gets a thread of its own, ticking at
`frames_per_second_for_ui` to read input, call `ISession.step_ui`, and write
finished results. A step that takes longer than one of those ticks does not hold
up input or output, because it is not on that thread. That is why `SessionDesc`
carries two rates: `frames_per_second_for_ui` is how often input is read and
results are presented, and `frames_per_second_for_step` is the rate the generated
frames are meant to play back at. Only the UI rate is read so far, generation
currently runs as fast as it can.

A window with no input to report, such as one writing an MP4, returns no events
and leaves `step_ui` at its default; the threading is unchanged.

Only the I/O thread touches the window, `open` and `close` included. That is what
a native window needs, and it keeps `IClientWindow` implementations free of
locking.

Writing happens on that thread too, so a window slower than generation leaves
results waiting. `run_session` bounds how many wait, with `max_pending`, and
`when_full` decides the rest: `WhenFull.BLOCK` holds generation back so every
result is presented, which is what a file output wants, and `WhenFull.DROP_OLDEST`
skips frames to keep latency down, which is what a realtime one wants. The caller
picks, since it is the caller that created the window.

Each waiting result carries the generation it was produced for, and a reset moves
on to the next one. Nothing from the generation the client abandoned is written,
whether it was already waiting or was still being generated when the reset
arrived, so what a client sees after restarting begins at the new step zero.

Reading input and presenting frames belong to `IClientWindow`, not to `ISession`.
`ISession.step_ui` is the second tick the I/O thread drives, so a session's UI work
keeps running while a step is in flight. It cannot produce output yet, so today it
can only update state.

Not built yet
-------------

- Pacing generation at `frames_per_second_for_step`, and a third thread at a fixed
  rate for game logic, which we expect to want and to stay optional.
- Reporting dropped results as data rather than a log line, so a caller can count
  them.
- Slowing generation before the window is saturated. The bounded queue only paces
  generation once results are already waiting.
- Input that keeps up with generation. Input is polled at the UI rate, so a run of
  fast steps can finish several of them between polls and hand them all the same
  batch. Pacing generation is what would fix it.
- An output path for `ISession.step_ui`, so UI work can reach the window rather
  than only updating session state.
