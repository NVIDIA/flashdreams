## Summary

This PR makes v2 WebRTC presentation write-driven so the UI thread owns frame
composition and pacing.

## Changes

- Present generated chunks frame by frame at a UI-owned cadence without bursts.
- Materialize frames in `window.write()` and deliver them to aiortc without a
  second WebRTC pacer.
- Keep a bounded two-frame sender queue and replace the oldest unsent frame
  during congestion.
- Order model output, high-priority UI composition, and high-priority WebRTC
  transfer work with CUDA events instead of device-wide synchronization.
- Redraw the UI for fresh input while model generation continues, with reusable
  input timeline and keyboard-state handling.
- Handle transient WebRTC `disconnected` state without showing a false terminal
  connection error.
- Keep the native SlangPy Cam2V overlay and report model FPS over the most recent
  two seconds.

## Validation

- 51 focused Cam2V and WebRTC tests passed.
- CUDA presentation-stream tests passed.
- Formatting, lint, type, and lockfile checks passed.
