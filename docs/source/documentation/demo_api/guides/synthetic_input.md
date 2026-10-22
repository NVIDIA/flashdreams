---
title: 'Synthetic Input'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

# Synthetic Input

Synthetic input replays versioned `UserInputEvent` payloads through any
client-window mode. Use it for repeatable QA, CI, and benchmarks without
interactive device input.

## Run a replay

Save a descriptor such as this one as `inputs.json`:

```json
{
  "version": 1,
  "events": [
    {
      "on": "ui_loop",
      "at": 0,
      "event": {"type": "keyboard", "key": "w", "state": "Pressed"}
    }
  ]
}
```

Pass the file with `--synthetic-input-file PATH`. For predictable
frame-by-frame replay, use blocking backpressure and on-demand presentation:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG \
  --mode mp4 \
  --output-path replay.mp4 \
  --total-model-steps 120 \
  --synthetic-input-file inputs.json \
  --backpressure-mode block \
  --presentation-mode on_demand
```

`block` preserves every model output instead of dropping queued work, and
`on_demand` runs the steady-state UI loop only when a new model frame is
selected. Together they present each retained model frame once and in order
during inference, making `ui_loop` schedules consistent between runs. The UI
can still run before inference starts and after it finishes. Synthetic input
also works with other modes and presentation settings when this behavior is not
required.

## Descriptor format

The top-level object must contain only `version` and `events`. Version 1
supports `ui_loop` scheduling.
`at` is a zero-based input-poll index: `0` releases an event when the runtime
first polls the client window after opening the session, before the first UI
step. Events with the same `at` value retain their file order.

The descriptor cursor and poll index are application-relative, so a replacement
session continues from the next unreleased event. Event timestamps are
session-relative and restart when a replacement session opens. Version 1
delivers each descriptor event at most once for the application run; replacement
sessions never rewind the cursor or poll index.

## Event payloads

Each `event` object must contain a registered `UserInputEvent` `type` and may
contain only that event type's fields. Do not provide `timestamp`; the runtime
stamps an event when it is released. Optional fields use the defaults below.

For example, a mouse-button press maps directly to the `mouse` event fields,
while an event without additional fields needs only its registered type:

```json
{"type": "mouse", "action": "button", "button": 0, "pressed": true}
```

```json
{"type": "reset"}
```

| Event type | Field | Type | Required | Default | Accepted values |
| --- | --- | --- | --- | --- | --- |
| `close` | - | - | - | - | - |
| `focus` | `focused` | `boolean` | no | `false` | - |
| `fullscreen_enabled_optimal_window_size` | `width` | `integer` | yes | - | - |
| `fullscreen_enabled_optimal_window_size` | `height` | `integer` | yes | - | - |
| `game_wheel` | `action` | `string` | no | `"state"` | `"connected"`, `"disconnected"`, `"state"` |
| `game_wheel` | `index` | `integer` | no | `0` | - |
| `game_wheel` | `controller_id` | `string` | no | `""` | - |
| `game_wheel` | `steering` | `number` | no | `0.0` | - |
| `game_wheel` | `throttle` | `number` | no | `0.0` | - |
| `game_wheel` | `brake` | `number` | no | `0.0` | - |
| `game_wheel` | `clutch` | `number` | no | `0.0` | - |
| `game_wheel` | `buttons` | `array[boolean]` | no | `[]` | - |
| `gamepad` | `action` | `string` | no | `"state"` | `"connected"`, `"disconnected"`, `"state"` |
| `gamepad` | `index` | `integer` | no | `0` | - |
| `gamepad` | `controller_id` | `string` | no | `""` | - |
| `gamepad` | `mapping` | `string` | no | `""` | - |
| `gamepad` | `axes` | `array[number]` | no | `[]` | - |
| `gamepad` | `buttons` | `array[number]` | no | `[]` | - |
| `gamepad` | `pressed` | `array[boolean]` | no | `[]` | - |
| `keyboard` | `key` | `string` | yes | - | - |
| `keyboard` | `state` | `string` | yes | - | `"Released"`, `"Pressed"` |
| `mouse` | `action` | `string` | no | `"move"` | `"move"`, `"button"`, `"wheel"` |
| `mouse` | `x` | `number` | no | `0.0` | - |
| `mouse` | `y` | `number` | no | `0.0` | - |
| `mouse` | `button` | `integer` | no | `0` | - |
| `mouse` | `pressed` | `boolean` | no | `false` | - |
| `mouse` | `wheel_x` | `number` | no | `0.0` | - |
| `mouse` | `wheel_y` | `number` | no | `0.0` | - |
| `numeral_keypad` | `value` | `integer` | no | `0` | - |
| `query_string` | `query_string` | `string` | no | `""` | - |
| `reset` | - | - | - | - | - |
| `touch` | `action` | `string` | no | `"move"` | `"start"`, `"move"`, `"end"`, `"cancel"` |
| `touch` | `touch_id` | `integer` | no | `0` | - |
| `touch` | `x` | `number` | no | `0.0` | - |
| `touch` | `y` | `number` | no | `0.0` | - |
| `touch` | `pressure` | `number` | no | `0.0` | - |
| `touch` | `primary` | `boolean` | no | `false` | - |
| `unknown` | - | - | - | - | - |
| `xr_controller` | `action` | `string` | no | `"state"` | `"connected"`, `"disconnected"`, `"state"` |
| `xr_controller` | `handedness` | `string` | no | `"none"` | `"left"`, `"right"`, `"none"` |
| `xr_controller` | `controller_id` | `string` | no | `""` | - |
| `xr_controller` | `axes` | `array[number]` | no | `[]` | - |
| `xr_controller` | `buttons` | `array[number]` | no | `[]` | - |
| `xr_controller` | `pressed` | `array[boolean]` | no | `[]` | - |
| `xr_controller` | `position` | `[number, number, number] or null` | no | `null` | - |
| `xr_controller` | `orientation` | `[number, number, number, number] or null` | no | `null` | - |
