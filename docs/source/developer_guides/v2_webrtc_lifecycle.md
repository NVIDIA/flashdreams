<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V2 WebRTC application and session lifecycle

The v2 WebRTC path keeps an application and its model alive while it creates
one session at a time from browser requests. A session owns one rollout; it
does not own the server or reload the model.

The Self-Forcing entry command remains:

```bash
uv run --project integrations_v2/t2v_self_forcing flashdreams-run-v2 \
  t2v-self-forcing --mode webrtc --port 8080
```

Open the printed URL, enter a prompt, and select **New session**. Closing or
refreshing the page ends the active rollout, but the process and loaded model
remain available for the next request. Ctrl-C ends the application.

## Ownership

| Component | Owns | Does not own |
| --- | --- | --- |
| `cli.py` | Argument parsing, application and window construction, final runner cleanup | Model state, session state, browser events |
| `ApplicationRunner` | The initialized application and the serial application-level session loop | Browser event types or per-step scheduling |
| `run_session()` | Exactly one session, its generation loop, its I/O thread, and session cleanup | The application or a persistent window after a successful handoff |
| `WebRTCClientWindow` | The WebRTC server and the thread-safe browser-event queue | Application or session state |
| `WebRTCServer` | HTTP, one active peer connection, its data channel, and its media track | Runtime lifecycle decisions |
| `T2VApplication` | The loaded model pipeline shared by every session | A rollout cache or browser connection |
| `T2VSession` | One prompt and one rollout cache | Model loading or server lifetime |

The CLI constructs the window, then transfers its lifetime cleanup
responsibility to the runner. While the server is idle, the runner's calling
thread opens and polls the window. During a session, `run_session()` lends the
window to one dedicated I/O thread, which performs the meaningful close when
the window should end. The runner does not touch it again until that thread has
stopped and the old session has closed. Its final, idempotent close is a fallback
for setup failures and interrupted handoffs. This is sequential access, not
concurrent access.

## Data flow

1. `cli.py` selects the `t2v-self-forcing` application and the `webrtc` window
   mode. Arguments after `--` belong to the application.
2. `ApplicationRunner.init()` calls `T2VApplication.init()`. The pipeline is
   set up, moved to its device, evaluated, and retained by the application.
3. The WebRTC mode creates `WebRTCClientWindow`, which starts its server thread
   and exposes the browser URL.
4. `ApplicationRunner.run()` resolves the CLI's partial `SessionDescRequest`
   against the initialized application's default `SessionDesc`.
5. A window whose `keeps_open_between_sessions` capability is true opens with
   that resolved stream format before a session exists. This lets the browser
   negotiate WebRTC and send its first request without loading a session cache
   first. Other windows start the resolved session immediately.
6. The server validates every data-channel message and invokes the callback
   registered by `WebRTCClientWindow`. The callback only appends the event to a
   thread-safe queue.
7. While idle, `wait_for_new_session()` drains that queue. It translates the
   latest close/new-session transition into a complete `SessionDesc` and
   returns only that description. `ApplicationRunner` therefore never parses a
   `UserInputEvent`.
8. The application validates the description and creates a `T2VSession` over
   its resident pipeline. `T2VSession.init()` creates only the per-rollout
   prompt/cache state.
9. `run_session()` starts one I/O thread. That thread opens the same window for
   the actual session boundary, drains input each UI tick, calls `step_ui()`,
   and writes completed results. The calling thread runs model steps.
10. A new-session event stops that rollout and returns the requested next
    `SessionDesc`. A close or natural completion returns no replacement. In
    a persistent window the runner keeps it open, closes the old session, and
    either starts the replacement or waits for another browser request.
11. Ctrl-C closes the peer/server and then the application. Releasing the
    application drops the one resident pipeline after every session cache has
    already been released.

## Why the WebRTC window is opened twice

The two calls mark different boundaries:

- The first call prepares the fixed stream format so a browser can connect
  while no session exists.
- The per-session call discards source frames queued by the previous rollout.
  Event timestamps retain one window-lifetime monotonic origin, so opening a
  replacement cannot reorder buffered events from the old and new browser.

They do not create two servers or two peer connections. A connected peer is
reused. The application must either accept the already-resolved `SessionDesc`
or reject it; it cannot silently change width, height, layout, or playback rate
after the browser stream was prepared.

## Lifecycle event ordering

Close and new-session events can arrive in one drained batch during a page
refresh. Their order is meaningful, so the latest transition wins:

- close, then new session: start the new browser's request;
- new session, then close: cancel the request because that browser went away.

Any close or replacement also advances the session generation. A model step
that was already running may finish, but its result is tagged with the old
generation and is not written. The WebRTC media track similarly clears frames
that it has not yet handed to the encoder. At most one already-encoded frame
can still be in flight; a transport cannot recall a frame it has sent.

## WebRTC connection lifecycle

Offer negotiation is serialized, and only one browser is admitted. Data
channel and peer callbacks capture the peer that registered them, so callbacks
from an old page cannot close or mutate its replacement. Closing the active
data channel releases the peer immediately, allowing a refreshed page to
negotiate without an indefinite series of HTTP 409 responses.

The new-session button starts enabled and reads **Opening...** while signaling
is in progress. A click during that interval keeps the latest valid prompt in
the page and sends it as soon as the data channel opens. The UI therefore does
not make model loading or a brief reconnect look like an unavailable action.

The media track uses `frames_per_second_for_step`, the generated video's
playback rate. `frames_per_second_for_ui` controls only how often the runtime
polls input and presents completed work.

The media source queue is currently unbounded. T2V rollouts are finite, and a
session replacement clears frames that have not reached the encoder, so its
size remains bounded by a rollout in the current application. A future
long-running producer will need an explicit block-or-drop policy chosen for
that application's latency requirements.

## Why these boundaries are useful

- The core generation pipeline and checkpoint load once and stay resident
  across browser refreshes and sequential prompts. The existing Wan pipeline
  may release and reload its one-shot text encoder after prompt encoding to fit
  within GPU memory; it does not reload the diffusion model between sessions.
- Session cleanup is complete before the next session is created, so rollout
  caches cannot overlap accidentally.
- WebRTC code transports validated events but never makes application
  lifecycle decisions.
- `ApplicationRunner.run()` visibly owns the possibly multi-session run, while
  `run_session()` visibly owns exactly one rollout.
- Failure paths keep the first useful exception while still releasing the
  session I/O thread, WebRTC server thread, window, and partially initialized
  model.

`SessionDescRequest` remains separate from `SessionDesc` on purpose: the former
means “only the fields the CLI explicitly supplied,” while the latter is the
fully resolved contract shared by the application, session, and window. No
additional host, session-runner class, or WebRTC-specific lifecycle wrapper is
needed.

The bundled browser page is currently prompt-oriented because text-to-video is
the application that needs client-created sessions today. A generic UI schema
should be introduced only when another application has a concrete, different
request format.
