<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint model package for FlashDreams

This package loads the published [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B) checkpoint through
FlashDreams.

## Model

- One action produces one 32-channel latent frame and four presented RGB frames.
- Waypoint 1.5 has no text-conditioning input in its published checkpoint
  configuration, so this integration will not expose a prompt encoder.
- User controls are 256 button IDs, mouse delta, and scroll movement.

## Integration baseline

- FlashDreams target: `8fd97fa38f04bc32c288760fa0fbf5da52464cea`
  (the post-#506 V2 runtime).
- Source integration: PR #464 at
  `0f1782346f33cc47cc0bd456c3c48c5f3b7145f4`.
- Published checkpoint:
  `Overworld/Waypoint-1.5-1B/model.safetensors`.
- Upstream parity implementation:
  [Overworldai/world_engine](https://github.com/Overworldai/world_engine).
- Closest reusable FlashDreams component: the shared Hunyuan Video 1.5 TAEHV
  streaming codec. Waypoint's DiT and cache topology are model-specific.

Runtime orchestration is intentionally outside this package. The V2 application
adapter owns argument parsing, sessions, input events, presentation, and file or
WebRTC output.

## Control files

Pass a JSON action timeline with ``--controls-file``. It uses every listed
action unless ``--actions N`` selects a prefix. The file format is:

```json
{
  "schema_version": 1,
  "actions": [
    {"buttons": [32], "mouse_dx": 0.1, "mouse_dy": 0.0, "scroll_wheel": 0},
    {},
    {"buttons": [1, 32]}
  ]
}
```

Every field within an action is optional. ``buttons`` is an array of model
button IDs, ``mouse_dx`` / ``mouse_dy`` are finite numbers, and
``scroll_wheel`` is ``-1``, ``0``, or ``1``. A control file takes precedence
over the repeated-control options, so it can be used without ``--buttons``,
``--mouse-dx``, ``--mouse-dy``, or ``--scroll``.
