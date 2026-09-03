<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint Action2V

This adapter binds the Waypoint model in `flashdreams-waypoint` to the shared
Action2V application, session, model-loop, input, and output APIs.
Model modules are loaded once per application; the image-established
transformer/decoder cache, RNG stream, and live controls are isolated per
session.

The application uses the shared [`apps/action2v`](../../../../apps/action2v/README.md)
shell.

## Command-line options

Arguments after `--` belong to Waypoint; runtime options go before it. Waypoint
uses the shared Action2V options and adds no model-specific flags.

| Option | Default | Description |
| --- | --- | --- |
| `--first-frame PATH` | pinned public image | Override the RGB or RGBA image that establishes the world. |
| `--seed N` | random | Set the non-negative model RNG seed for reproducible generation. |
| `--device DEVICE` | `cuda` | Select the model device. |
| `--profile` | off | Enable synchronized pipeline-stage profiling. |
| `--mouse-sensitivity SCALE` | `1.0` | Scale normalized pointer motion before mapping it to Waypoint pixel deltas. |
| `-h`, `--help` | — | Print these options without loading checkpoints. |

Run `flashdreams-run-v2 --help` for runtime options such as output mode, MP4
path, metrics path, WebRTC host, and port.

## Run Waypoint

For live keyboard and mouse input, use the browser window:

```bash
uv run flashdreams-run-v2 action2v-waypoint-1-5-1b --mode webrtc \
    -- --seed 464
```

Arguments after `--` belong to Waypoint. Run
`flashdreams-run-v2 action2v-waypoint-1-5-1b -- --help` for the complete list. The first run downloads the public Waypoint 1.5 1B
and TAEHV checkpoints.

The application declares four-frame `TCHW` results on Waypoint's native
1024x512 canvas at 60 FPS playback. First-frame images are resized once to that native canvas; generated frames are
presented without another spatial resample.
