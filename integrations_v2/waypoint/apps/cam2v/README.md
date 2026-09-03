<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint Cam2V

Install the Waypoint integration and launch its `cam2v-waypoint` application:

```bash
uv sync --package flashdreams-waypoint --inexact
uv run --no-sync flashdreams-run-v2 cam2v-waypoint --mode webrtc \
  --presentation-mode continuous --host 0.0.0.0 --port 8089 -- \
  --example-data
```

Use a local starting image with live controls:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-waypoint --mode webrtc \
  --presentation-mode continuous --host 0.0.0.0 --port 8089 -- \
  --image-path seed.png --seed 464
```

## Controls

| Input | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move |
| Mouse | Look around |
| `Shift` | Sprint |
| `Space` | Jump or context action |
| `R` | Reset the rollout to the starting image |

The on-screen HUD lists these controls and wraps held keys in brackets. Losing
window or browser focus clears held controls. Pass `--no-ui` to disable the
HUD. Interactive examples opt into continuous presentation for prompt HUD
updates. Waypoint otherwise defaults to on-demand presentation so finite
replays contain each generated frame exactly once.

For a deterministic file-driven MP4:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-waypoint \
  --presentation-mode on_demand \
  --output-path waypoint.mp4 --stats-path waypoint.metrics.json -- \
  --example-data --actions 40 --seed 464 --profile --no-ui
```

The application emits four seed frames followed by four frames per action at
Waypoint's native 1024 x 512 resolution. `--seed-image` remains an alias for
`--image-path`. Run `flashdreams-run-v2 cam2v-waypoint -- --help` for all
application arguments.

## Tests

```bash
uv sync --package flashdreams-waypoint --group test --inexact
uv run --no-sync pytest integrations_v2/waypoint/tests -m ci_cpu
```

Run `-m ci_gpu` instead for the CUDA attention and cache tests.
