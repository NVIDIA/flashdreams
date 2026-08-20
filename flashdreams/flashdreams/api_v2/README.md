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

There are two ways to run a session, interactively or as a batch. These are
styles of running a session rather than kinds of session, and a session is
written the same way for both:

- `flashdreams.runtime_v2.session_runner.run_session` drives a session against a
  window on two threads, for a fixed number of steps or until the window reports
  a close. This is the interactive path.
- `flashdreams.runtime_v2.batch_runner.run_batch` generates a fixed number of
  steps for the model generation thread and writes each to an `OutputSink`,
  which is `Mp4OutputSink` when the results are going to a file.

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

`run_session` uses two threads, and every interactive window runs that way.
`run_batch` generates a step and writes it on the model generation thread. The
rest of this section is about `run_session`.

Generation is on the calling thread; the window gets a thread of its own,
ticking at `frames_per_second_for_ui` to read input, call `ISession.step_ui`,
and write finished results. A step that takes longer than one of those ticks
does not hold up input or output, because it is not on that thread.

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
window. A batch run needs neither, since it writes each result as it comes.

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
- Driving a session from a window's own event loop. A WebRTC or native window
  owns an event loop already, an asyncio one or an OS message pump, and stepping
  the session from it is what such a window wants. `run_session` polls instead,
  on a thread it starts itself.
- `flashdreams-run`: a CLI that creates the requested output, loads an
  application module, and runs it interactively or as a batch. Until it exists,
  the caller wires that up and an integration ships no entry point.
- An output path for `ISession.step_ui`, so UI work can reach the window rather
  than only updating session state.
- Shared per-domain test entry points, so a model integration gets coverage from
  one call — for example `test_t2v_model_impl(model_config, expected_frame_stats)`
  returning a pass or fail result.
