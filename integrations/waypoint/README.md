<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint for FlashDreams

This package loads the published [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B) checkpoint through
FlashDreams.

## Model

- One action produces one 32-channel latent frame and four presented RGB frames.
- Waypoint 1.5 has no text-conditioning input in its published checkpoint
  configuration, so this integration will not expose a prompt encoder.
- User controls are 256 button IDs, mouse delta, and scroll movement.

## Run a rollout

Sync the workspace, then invoke the registered runner with a seed image and
repeated control:

```bash
uv sync
uv run flashdreams-run waypoint-1.5-1b --seed-image .\seed.jpg --actions 45 --buttons 32
```

The runner repeats the supplied button IDs, mouse displacement, and scroll value
for each action. Without ``--actions``, a repeated-control rollout emits four
actions. It writes ``outputs/waypoint-1.5-1b.mp4`` at 60 FPS by default. Pass
``--seed N`` to replay a rollout; otherwise the runner logs its generated seed.
Use ``uv run flashdreams-run waypoint-1.5-1b --help`` to list every override.

For the pinned public example seed and its 118-action control sequence:

```bash
uv run flashdreams-run waypoint-1.5-1b --example-data True
```

The seed image is cached under
``$FLASHDREAMS_CACHE_DIR/example_data/waypoint/``. The action timeline is the
versioned repository asset
``assets/example_data/waypoint/example_controls.json``.
The CLI requires an explicit Boolean value for this option.

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
