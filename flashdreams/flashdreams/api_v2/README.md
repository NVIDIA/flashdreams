<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Protocols for the FlashDreams API.

- `application.py` / `session.py`: `IApplication` creates an `ISession` from a
  `SessionDesc`, and the session reports what it resolved to.
- `input_source.py` / `output_sink.py` / `client_window.py`: an `OutputSink` is
  where results go, and is given the session's `SessionDesc` in
  `OutputSink.open`. `IClientWindow` is a producer of inputs and a consumer of
  outputs for a single client, populating an input source and targeting an
  output sink.
- `user_input_event_data.py`: base type for event payloads.

One loop runs every session:
`flashdreams.runtime_v2.session_runner.run_session` drives a session against a
window on two threads, for a fixed number of steps or until the window reports a
close. What changes between a run somebody is watching and a run that writes a
file is the window it is driven against, not the loop and not the session:

- An interactive run is given a window with a client on the other end, which
  reports what the client did and presents what the session generated.
- A run writing a file is given
  `flashdreams.runtime_v2.mp4_client_window.Mp4ClientWindow`, which reports no
  input and encodes every result through `Mp4OutputSink`. Since it never reports
  a close, `steps` is what ends the run.

A run that writes a file is there to produce an artifact that can be compared:
the same configuration in gives the same frames out, so a run can be scored
against other engines running the same model and against earlier runs of our own.
It gets that from being handed no input and from keeping every result, with
`WhenFull.BLOCK`, rather than from a loop of its own. A run with a client waiting
gives both up on purpose, dropping frames to stay in the present.

One thing does still follow a wall clock in both: how often `ISession.step_ui` is
called, which is why what a session does there must not change what `step`
generates. See `session.py`.

Ownership
---------

Agreed design decisions. Change them by discussion.

- An application module implements `IApplication` and `ISession`. The runtime
  creates every other protocol here and passes it in.
- `IApplication` lasts as long as the process. It holds what its sessions share,
  such as a checkpoint or a compiled pipeline, and outlives every session it
  creates.
- `ISession` is one run: KV cache, game state, and anything else that must not
  carry into another run.
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
- An `OutputSink` reads `StepResult.output` as one of two things: a floating point
  tensor holding `[-1, 1]`, which is what FlashDreams models emit, or an integer
  tensor holding raw `0`-`255` values. `SessionDesc` does not carry a range, and a
  session has no way to declare one, so every sink follows this convention instead.

Threading
---------

`run_session` uses two threads, whatever window it is driving, a file included.

`ISession.step` runs on the thread that called `run_session`, which is the
generation thread; the window gets an I/O thread of its own, ticking at
`frames_per_second_for_ui` to read input, call `ISession.step_ui`, and write
finished results. A step that takes longer than one of those ticks does not hold
up input or output, because it is not on the I/O thread.

The two rates in `SessionDesc` measure different things.
`frames_per_second_for_ui` is how often input is read and finished results are
presented, and `frames_per_second_for_step` is the rate the generated frames are
meant to play back at. Only the UI rate is read so far — generation currently
runs as fast as it can.

Only the I/O thread touches the window, `open` and `close` included. That is what
a native window needs, and it keeps `IClientWindow` implementations free of
locking.

Writing happens on that thread too, so a window slower than generation leaves
results waiting. `run_session` bounds how many wait, with `max_pending`, and
`when_full` decides the rest: `WhenFull.BLOCK` holds generation back so every
result is presented, which is what a window that must show all of them wants,
and `WhenFull.DROP_OLDEST` skips frames to keep latency down, which is what a
realtime program wants. The caller picks, since it is the caller that created the
window. A run writing a file leaves this alone: the default is what keeps every
frame, and dropping one would leave a hole in the file nothing reports.

A reset starts a new attempt at the run, which `run_session` calls a generation:
a counter it takes up by one each time, recorded on every result it hands to the
window. Nothing from the attempt the client abandoned is written, whether it was
already waiting or was still being generated when the reset arrived, so what a
client sees after restarting begins at the new step zero.

Reading input and presenting frames belong to `IClientWindow`, not to `ISession`.
Alongside reading input, each I/O tick calls `ISession.step_ui`, so a session's
UI work keeps running while a step is in flight. It cannot produce output yet, so
today it can only update state.

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
- Driving a session from a window's own event loop. A WebRTC or native window
  owns an event loop already, an asyncio one or an OS message pump, and stepping
  the session from it is what such a window wants. `run_session` polls instead,
  on a thread it starts itself.
- `flashdreams-run-v2`: a CLI that creates the requested output, loads an
  application module, and runs it against a client window or to a file. Named
  apart from the v1 `flashdreams-run`, which stays as it is. Until it exists, the
  caller wires that up and an integration ships no entry point.
- An output path for `ISession.step_ui`, so UI work can reach the window rather
  than only updating session state.
- Shared per-domain test entry points, so a model integration gets coverage from
  one call — for example `test_t2v_model_impl(model_config, expected_frame_stats)`
  returning a pass or fail result.
