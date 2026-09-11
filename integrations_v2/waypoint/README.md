<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint 1.5 integration

This package implements the published
[Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
checkpoint and binds it to the shared
[Action2V application](../../apps/action2v/README.md). Waypoint owns the model,
checkpoint mapping, control encoding, pipeline, and caches; Action2V owns the
CLI, sessions, browser input, and output presentation.

## Names

- `waypoint-1.5-1b` is the pipeline config name: model family, upstream release
  1.5, and upstream `1B` checkpoint label.
- `action2v-waypoint-1-5-1b` is the application slug used by
  `flashdreams-run-v2`. `action2v` identifies the application contract; the
  remaining components identify its model config. Dots become hyphens in the
  command-line slug.
- `WAYPOINT_1_5` is the immutable checkpoint architecture contract.
- `WAYPOINT_1_5_CHECKPOINT` is the published DiT checkpoint URL.
- `PIPELINE_WAYPOINT_1_5` assembles the transformer, four-step scheduler,
  control encoder, and TAEHV decoder.
- `WAYPOINT_CONFIGS` indexes the available pipeline variants by config name.
- `WAYPOINT_ACTION2V_DEFAULTS` adds Action2V-specific defaults such as the
  application slug, first frame, display size, and playback rate.

`1B` preserves the upstream checkpoint name. The upstream model card reports
1.2B parameters, while the checkpoint contains 1,860,823,096 serialized tensor
elements; neither count is used to rename the artifact.

## Package layout

```text
config.py                    public pipeline config
impl/spec.py                 checkpoint architecture constants
impl/checkpoint.py           strict upstream state-dict mapping
impl/controls.py             checkpoint-level control representation
impl/input_mapping.py        browser events to Waypoint controls
impl/encoder.py              control encoder
impl/pipeline.py             seed, generate, and finalize flow
impl/scheduler.py            fixed Euler schedule
impl/decoder.py              TAEHV adapter
impl/transformer/            DiT and local/global K/V caches
apps/action2v/adapter.py      shared Action2V binding
tests/                       CPU, CUDA, and structure checks
```

The package depends inward on FlashDreams `infra` and shared recipes. Framework
code does not import this integration.

## Execution contract

1. The first RGB/RGBA image is resized to 1024x512 and repeated into four seed
   frames.
2. TAEHV encodes those frames and the transformer commits the seed action to
   its K/V history.
3. Each keyboard/mouse snapshot becomes one `WaypointControl`.
4. Four Euler evaluations generate one latent action; TAEHV decodes it into
   four RGB frames.
5. A separate sigma-zero evaluation commits the clean latent to the cache for
   the next action.

The public result is detached `[4, 3, 512, 1024]` TCHW data in `[-1, 1]`.
One latent action has shape `[B, 1, 32, 32, 64]` before 2x2 patchification.

Most transformer blocks retain 16 actions densely. Blocks 3, 7, 11, 15, 19,
and 23 use the sparse 128-action history. Provisional denoising evaluations may
replace only the current cache slot; only seed establishment and `finalize`
advance history.

Model weights are shared by the application. Each session owns its seed,
controls, RNG state, and model/codec caches. A lock serializes shared model and
RNG work.

The checkpoint has no prompt encoder or prompt/cross-attention weights, so this
integration intentionally exposes no text prompt. It also does not implement
the upstream 360P checkpoint, quantization, or multi-GPU inference.

For usage, see the [Action2V adapter README](apps/action2v/README.md) and the
[Waypoint user guide](../../docs/source/models/waypoint.rst). For tested
hardware, parity results, performance, and exact commands, see
[VALIDATION.md](VALIDATION.md).

## How Control Events Work

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

FlashDreams V2 owns typed event transport, ordering, fan-out, reset generation,
focus events, and client backends. `apps/action2v` owns the application and
session lifecycle plus a model-neutral `ActionSnapshot` accumulator. It retains
held key and mouse-button state, emits transient normalized mouse/wheel deltas,
and clears state on reset or focus loss.

Waypoint owns `WaypointActionMapper`, which converts snapshots to Windows
virtual-key IDs, pixel-scaled motion, ternary wheel input, and
`WaypointControl`. `WaypointControlEncoder` remains responsible for converting
that checkpoint-specific object to tensors. The legacy
`flashdreams.runtime.mapping.InputMapping` API is not used.
