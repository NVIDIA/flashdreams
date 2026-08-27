# Make WebRTC presentation write-driven

## Summary

- Make the UI thread the single owner of presentation cadence.
- Materialize exactly one owned video frame synchronously in each WebRTC
  `window.write()` call.
- Remove WebRTC's second frame pacer and latest-frame repeat behavior; aiortc
  receives every admitted frame as soon as it asks for one.
- Retain at most two unsent frames during encoder or network congestion. A new
  write replaces only the oldest queued frame, never one already handed to
  aiortc.
- Preserve source timing with monotonic PTS and use a bounded final-frame drain
  during shutdown.

## CUDA presentation readiness

- Add an optional producer-recorded CUDA readiness event to `StepResult`.
- Record readiness when model output crosses into the presentation queue, and
  let UI loops join it from a dedicated consumer stream.
- Make built-in UI producers record their completed output and make MP4/WebRTC
  sinks wait before conversion or device-to-host transfer.
- Perform WebRTC CUDA conversion and transfer on a sink-owned high-priority
  stream, then finish materialization before `write()` returns.

## Scope

This PR contains only the generic presentation and WebRTC transport fix. The
Cam2V visual redesign, key highlighting, live-session HUD, and related browser
instrumentation are split into a separate branch/PR.

## Validation

- Focused WebRTC CPU tests, including a real aiortc connection.
- Session runner, native-window, and MP4 presentation tests.
- CUDA stream-ordering and WebRTC materialization tests on NVIDIA GB300.
- Repository pre-commit checks.
