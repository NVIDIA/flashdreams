<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint 1.5 V2 application

This package adapts the independently authored `flashdreams-waypoint` model
package to FlashDreams' V2 application, session, model-loop, event, and output
APIs. Model modules are loaded once per application; the image-established
transformer/decoder cache, RNG stream, and live controls are isolated per
session.

## Run Waypoint

Write a deterministic example rollout and its step metrics:

```bash
flashdreams-run-v2 waypoint-1-5-1b \
    --output-path waypoint.mp4 --stats-path waypoint.metrics.json \
    -- --example-data --actions 40 --seed 464 --profile
```

Use a local image and a controls JSON file in the same finite MP4 mode:

```bash
flashdreams-run-v2 waypoint-1-5-1b --output-path waypoint.mp4 \
    -- --seed-image seed.png --controls-file controls.json --seed 464
```

For live keyboard and mouse input, use the browser window and omit a controls
file:

```bash
flashdreams-run-v2 waypoint-1-5-1b --mode webrtc \
    -- --seed-image seed.png --seed 464
```

Arguments after `--` belong to Waypoint. Run
`flashdreams-run-v2 waypoint-1-5-1b -- --help` for the complete list. The first
run downloads the public Waypoint 1.5 1B and TAEHV checkpoints.

The application declares four-frame `TCHW` results on Waypoint's native
1024x512 canvas at 60 FPS playback. Seed images are resized once to that native
canvas; generated frames are presented without another spatial resample. File
controls use blocking, new-results-only presentation so MP4 output retains
every generated frame.

Sessions created by one application intentionally start from the configured
seed, which makes the same control sequence reproducible across reset and
session recreation. Model modules are shared, and model/RNG work is serialized
by an application lock while each session restores its own RNG and cache state.
The current WebRTC runtime accepts one browser client per server process.

See [ADR-1-control-events.md](ADR-1-control-events.md) for the live keyboard and
mouse contract. Passing `--controls-file` selects finite deterministic mode;
omitting it selects live input. `--example-data` uses the pinned public seed
and bundled 118-action timeline. See [VALIDATION.md](VALIDATION.md) for CPU,
CUDA, real-checkpoint, MP4, performance, and official-reference parity evidence.
