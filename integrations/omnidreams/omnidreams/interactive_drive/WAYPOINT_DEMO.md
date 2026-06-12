<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Waypoint Demo

`interactive-drive` can drive the ego vehicle from a ClipGT scene-editor
waypoint trajectory instead of reading keyboard, game-controller, or steering
wheel drive input. This is useful for repeatable demos where the vehicle should
follow the same route every run.

The waypoint file must use the same ClipGT world coordinate frame as the scene
you launch. Extra metadata fields are allowed and ignored by the runtime, so
scene-editor exports can be passed directly to `--drive-trajectory`.

## Launch

Run from the flashdreams workspace root after installing the interactive-drive
extra:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
```

For the demo trajectory at
`/home/horde/omnidreams-scenes/065dcac9-ee67-4434-a835-c6b816c88e48/clipgt/drive_trajectory.json`,
launch the matching scene archive with:

```bash
uv run --package flashdreams-omnidreams interactive-drive \
  --scene /home/horde/omnidreams-scenes/clipgt-065dcac9-ee67-4434-a835-c6b816c88e48.usdz \
  --drive-trajectory /home/horde/omnidreams-scenes/065dcac9-ee67-4434-a835-c6b816c88e48/clipgt/drive_trajectory.json \
  --backend raster \
  --manifest example_world_model_perf.yaml \
  --recording-auto-start
```

To view the same route from a browser, use the MJPEG presenter:

```bash
uv run --package flashdreams-omnidreams interactive-drive \
  --stream-mjpeg :8080 \
  --scene /home/horde/omnidreams-scenes/clipgt-065dcac9-ee67-4434-a835-c6b816c88e48.usdz \
  --drive-trajectory /home/horde/omnidreams-scenes/065dcac9-ee67-4434-a835-c6b816c88e48/clipgt/drive_trajectory.json
```

Then open `http://<host-ip>:8080/` and select the
`clipgt-065dcac9-ee67-4434-a835-c6b816c88e48` scene. The MJPEG presenter waits
for the browser selection before starting the rollout.

When `--drive-trajectory` is set, the ego vehicle follows the waypoint route.
Keyboard and wheel drive controls are ignored for steering, throttle, and
brake, while runtime controls such as view switching, reset, exit scene, and
quit remain available.

If the selected manifest enables both `recording_enabled: true` and
`recording_auto_start: true`, a non-looping waypoint trajectory runs as an
unattended recording job: the demo starts the selected scene, records the
route, saves the recording bundle when the final waypoint is reached, and
then exits.

For faster batch capture, pass `--headless` to skip local presentation. Passing
`--recording-auto-start` (or the `--recording_auto_start` alias) also forces
headless mode and disables realtime frame pacing, so the demo generates frames
as fast as the backend can produce them. The recording bundle still uses the
manifest FPS, so `example_world_model_perf.yaml` recordings play back at
30 FPS even when captured faster than realtime.

## Sample Waypoint JSON

```json
{
  "schema": "clipgt-waypoint-trajectory-v1",
  "name": "drive-trajectory",
  "coordinate_frame": "clipgt_world_m",
  "scene_path": "/home/horde/omnidreams-scenes/065dcac9-ee67-4434-a835-c6b816c88e48/clipgt",
  "clipgt_dir": "/home/horde/omnidreams-scenes/065dcac9-ee67-4434-a835-c6b816c88e48/clipgt/clipgt",
  "speed_mps": 10.0,
  "lookahead_m": 6.0,
  "waypoint_tolerance_m": 2.1,
  "stop_at_end": true,
  "loop": false,
  "waypoints": [
    {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    {
      "x": -12.71737086420638,
      "y": -1.3575551854652854,
      "z": 0.0
    },
    {
      "x": -16.997800652350644,
      "y": 14.694056520075709,
      "z": 0.0
    },
    {
      "x": -19.940596131699827,
      "y": 31.013195087375724,
      "z": 0.0
    },
    {
      "x": -24.221025919844095,
      "y": 54.288032060410174,
      "z": 0.0
    }
  ]
}
```

## JSON Fields

| Field | Required | Default | Description |
|---|---:|---:|---|
| `schema` | No | none | Must be omitted or set to `clipgt-waypoint-trajectory-v1`. |
| `name` | No | `drive-trajectory` | Name printed in the launch log. |
| `waypoints` | Yes | none | At least two points. Each point can be an object with `x`, `y`, optional `z`, or an array like `[x, y, z]`. |
| `speed_mps` | No | `4.0` | Target forward speed while following the route. |
| `lookahead_m` | No | `6.0` | Distance ahead on the polyline used as the steering target. Larger values smooth steering but cut corners more. |
| `waypoint_tolerance_m` | No | derived from lookahead | Distance from the route end where a non-looping trajectory is considered finished. |
| `stop_at_end` | No | `true` | For non-looping routes, brake to a stop at the final waypoint. Set to `false` to keep driving after the final waypoint. |
| `loop` | No | `false` | Connect the final waypoint back to the first and keep following the route indefinitely. |
| `coordinate_frame`, `scene_path`, `clipgt_dir` | No | ignored | Bookkeeping from scene-editor exports. The runtime ignores these fields. |

The controller uses waypoint `x` and `y` for path following. The `z` value is
accepted for compatibility with scene-editor exports but does not affect
steering or speed.

## Tuning Notes

- Keep the first waypoint near the scene's starting ego pose to avoid an
  aggressive initial turn.
- Increase `lookahead_m` if the vehicle oscillates around the route.
- Decrease `lookahead_m` or `speed_mps` for tight turns.
- Use `loop: true` for closed courses. In loop mode the final waypoint is
  connected back to the first waypoint and `stop_at_end` has no effect.
