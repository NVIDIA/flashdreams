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

`flashdreams.runtime_v2.session_runner.run_session` drives a session against a
window, for a fixed number of steps or until the window reports a close.

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
- The `UserInputEventData` types cover the input modalities supported today, and
  the runtime owns that set. Integrations consume them rather than adding more.
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

Threading
---------

`run_session` uses two threads, and every window runs that way. Generation is on
the calling thread; the window gets a thread of its own, ticking at
`frames_per_second_for_ui` to read input, call `ISession.step_ui`, and write
finished results. A slow step does not hold up input or output, which is why
`SessionDesc` carries the two frame rates separately. Only the UI rate is read so
far — generation currently runs as fast as it can.

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
picks, since it is the caller that created the window. Results still waiting when
a reset arrives are discarded either way.

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
- Recognising a result as belonging to an abandoned generation. A reset throws away
  what is already waiting, but a step that was running when the reset arrived
  still has its result presented, because generation only learns about the reset
  on its next step. Tagging results with a generation would close that.
- `ApplicationRunner`: takes an `IApplication` and an `IClientWindow` and drives
  the main loop. `run_session` is what exists today; it drives a session the
  caller already created.
- `flashdreams-run`: a CLI that creates the requested kind of client window,
  loads an application module, and hands both to `ApplicationRunner`. Until it
  exists, the caller wires that up and an integration ships no entry point.
- An output path for `ISession.step_ui`, so UI work can reach the window rather
  than only updating session state.
- Shared per-domain test entry points, so a model integration gets coverage from
  one call — for example `test_t2v_model_impl(model_config, expected_frame_stats)`
  returning a pass or fail result.
