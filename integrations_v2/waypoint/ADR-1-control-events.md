<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ADR-1: Waypoint live control event semantics

Status: accepted for the V2 integration.

The Waypoint model consumes a set of held IDs from a fixed 256-entry button
vocabulary, relative mouse motion, and a ternary wheel direction. FlashDreams
V2 clients instead report keyboard strings, mouse-button indices, normalized
absolute pointer coordinates, and wheel deltas.

The canonical button mapping follows the official Overworld Biome client at
revision `e3bc3715ba32f787d1f9719f183d4fddf6917cbe`. It mirrors Windows virtual-key
codes:

- `A`–`Z` map to 65–90 and `0`–`9` map to 48–57. Browser key case is ignored.
- arrows map to `0x25`–`0x28`; Shift, Control, Space, Tab, and Enter map to
  `0x10`, `0x11`, `0x20`, `0x09`, and `0x0d`.
- V2 mouse indices left/middle/right (`0/1/2`) map to Waypoint IDs
  `0x01/0x04/0x02`.
- Unknown keys and additional mouse buttons are ignored. Escape remains a
  client/runtime command rather than model input.

Key and mouse-button press/release events update held state. Held state is
included in every generated action until a matching release, focus loss, or
reset. Focus loss and reset clear every held input and discard the pointer
origin so controls cannot stick.

Mouse movement uses normalized coordinates. The first move after focus/reset
establishes an origin and emits no motion. Later moves are converted to
presentation pixels as `(current - previous) * (width, height)`, multiplied by
the configured sensitivity, and accumulated across the event batch. Button and
wheel events update the origin without creating look motion. This intentionally
matches Biome's pixel-delta contract while adapting V2's absolute coordinates.

Vertical wheel deltas are summed across the batch and reduced to `-1`, `0`, or
`1`. Positive means wheel up, matching the V2 browser client's sign convention
and Waypoint's `down/stationary/up` ordering. Horizontal wheel input is ignored
because the checkpoint has one wheel channel.

Events are processed in V2 timestamp order. Edge state is persistent; mouse and
wheel accumulators are per generated action. A reset inside a batch clears both
persistent and accumulated state before later events in that batch are applied.

File-driven mode has complete precedence over live input: keyboard and mouse
events do not alter a controls-file rollout. Reset and close remain runtime
lifecycle events in either mode. Live mode samples one coalesced control for
every model-loop action.
