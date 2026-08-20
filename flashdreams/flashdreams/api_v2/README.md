<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Protocols for the FlashDreams API.

- `application.py` / `session.py`: `IApplication` creates an `ISession` from a
  `SessionDesc`, and the session reports what it resolved to.
- `input_source.py` / `output_sink.py` / `client_window.py`: `IClientWindow` is
  one client's input and output together. It is given the session's `SessionDesc`
  in `OutputSink.open`.
- `user_input_event_data.py`: base type for event payloads.

Each kind of output has its own loop, and what they share is the session they
step rather than the loop that steps it:

- `flashdreams.runtime_v2.session_runner.run_session` drives a session against a
  window on two threads, for a fixed number of steps or until the window reports
  a close. This is the interactive path.
- `flashdreams.runtime_v2.batch_runner.run_batch` generates a fixed number of
  steps on one thread and writes each to a window that reports no input. This is
  the path for output that is a file, and `Mp4ClientWindow` is that window.

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
- A window reads `StepResult.output` by its dtype: a floating point tensor holds
  `[-1, 1]`, which is what FlashDreams models emit, and an integer tensor holds
  raw `0`-`255` values. `SessionDesc` does not carry a range, so this is the
  convention every window follows rather than something a session declares.

Threading
---------

`run_session` uses two threads, and every interactive window runs that way. A
batch run does not: `run_batch` generates a step and writes it on one thread,
because a file has no client to take input from or to keep up with. The rest of
this section is about `run_session`.

Generation is on the calling thread; the window gets a thread of its own,
ticking at `frames_per_second_for_ui` to read input, call `ISession.step_ui`,
and write finished results. A slow step does not hold up input or output, which
is why `SessionDesc` carries the two frame rates separately. Only the UI rate is
read so far — generation currently runs as fast as it can.

A window with no input to report returns no events and leaves `step_ui` at its
default; the threading is unchanged.

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
- The loops an interactive window needs. A WebRTC or native window owns an event
  loop of its own, an asyncio one or an OS message pump, and would drive the
  session from it. `run_session` polls instead, on a thread it starts itself.
- `flashdreams-run`: a CLI that creates the requested kind of client window,
  loads an application module, and runs the loop that window needs. Until it
  exists, the caller wires that up and an integration ships no entry point.
- An output path for `ISession.step_ui`, so UI work can reach the window rather
  than only updating session state.
- Several batch runs from one loaded application. `run_batch` initializes the
  application it is given and closes it again, so rendering two files today
  loads whatever the application holds twice.
- Shared per-domain test entry points, so a model integration gets coverage from
  one call — for example `test_t2v_model_impl(model_config, expected_frame_stats)`
  returning a pass or fail result.
