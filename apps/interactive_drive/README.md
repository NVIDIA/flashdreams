<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Interactive Drive

## Controls

### Keyboard

| Key | Action |
| --- | --- |
| `W` or Up arrow | Accelerate forward. |
| `S` or Down arrow | Accelerate in reverse. |
| `A` or Left arrow | Steer left. |
| `D` or Right arrow | Steer right. |
| Space | Brake; while held, throttle is suppressed. |
| `1` | Show the RGB view. |
| `2` | Show the HD-map conditioning view. |
| `3` | Show the PhysX collider view. |

### Controller

Standard-mapped gamepads use these controls:

| Control | Action |
| --- | --- |
| Left stick, push forward+tilt | Steer. |
| Right trigger | Accelerate in the selected gear. |
| Left trigger | Brake. |
| `R` (hold) | Select reverse gear; releasing returns to Drive. |
| `Start` / `+` (Plus) | Restart the current rollout. |

View selection does not currently have controller bindings. No other gamepad
sticks, axes, or buttons are used. A connected steering wheel uses its steering,
throttle, and brake inputs directly.

A long-running native-v2 driving demo. Its `InteractiveDriveUILoop` HUD contains
scene and variant selection,
driving telemetry, steering-wheel and pedal sprites, post-processing controls,
and a BEV minimap. Dear ImGui builds the immediate-mode HUD and SlangPy renders
it with GPU textures; the application does not use CSS.

The package also owns its model-neutral scene loading, simulation, rendering,
input handling, and wheel-configuration support. Its world-model binding is
supplied by an integration adapter.

## Usage

Install the application, then start it with no application arguments:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc --port 8089
```

Use `interactive-drive-omnidreams-perf`/`interactive-drive-omnidreams-fast-perf` instead for the native-accelerated, performance-tuned configurations.

Forward so that you may connect via `<ip>:8089` by adding `--host 0.0.0.0`

The default scene downloads on first use from the gated
`nvidia/omni-dreams-scenes` Hugging Face dataset. Application arguments are
optional and follow the `--` separator:

| Argument | Description |
| --- | --- |
| `--scene PATH` | Use a local USDZ scene instead of downloading the default scene. |
| `--prompt TEXT` | Override the prompt stored in the selected scene variant. |
| `--camera NAME` | Select a camera from the scene. Default: `camera_front_wide_120fov`. |
| `--variant NAME` | Select the scene's initial-frame and prompt variant. Default: `default`. |
| `--total-blocks N` | Stop after this many generated blocks; `0` runs until the session is stopped. Default: `0`. |
| `--fps N` | Set the application frame rate. Default: `30`. |
| `--width N` | Set the output width. Default: `1280` (`1168` for the perf app). |
| `--height N` | Set the output height. Default: `704` (`640` for the perf app). |
| `--view {rgb,hdmap,physx}` | Select the initial RGB, HD-map conditioning, or PhysX collider view. Default: `rgb`. |
| `--no-ui` | Present model output directly without creating the HUD or rendering its BEV minimap. |
| `--game-mode` | Enable the speed limit and collisions with scene actors and static map geometry. |
| `--postprocess-preset NAME` | Start with a registered video post-processing preset enabled. Default: none. |
| `--world-model-profile` | Enable synchronized world-model profiling. |
| `--world-model-device DEVICE` | Select the model device. Default: `cuda:0`. |
| `--world-model-seed N` | Pin the seed used for each rollout. |
| `--world-model-debug-condition-frame-dir PATH` | Override first-chunk condition frames for debugging. |

For example, use a local scene, select its rain variant, override its prompt,
and enable RTX super resolution:

```bash
uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc -- \
    --scene scene.usdz --variant rain --prompt "A rainy night drive" \
    --game-mode --postprocess-preset rtx-super-resolution
```

For example, render every generated frame once, in order, with the HUD disabled (for quality evaluation):

```bash
uv run flashdreams-run-v2 interactive-drive-omnidreams-perf \
    --mode mp4 --output-path artifacts/test/interactive-drive.mp4 \
    --backpressure-mode block --presentation-mode on_demand -- \
    --no-ui --total-blocks 60
```

This is frame-lossless presentation: the runtime neither drops nor repeats
generated frames. The MP4 itself is currently encoded as H.264 at CRF 18, so it
is not mathematically lossless at the pixel/codec level.

For example, render a native-window with game-mode collisions enabled:
```bash
uv run flashdreams-run-v2 interactive-drive-omnidreams-perf --mode native-window -- \
    --game-mode
```

The HUD's view button cycles through **RGB → HDMAP → PHYSX**.

When `--postprocess-preset` is set, the preset starts enabled and the HUD's
**Post-processing** checkbox can toggle it between generated chunks. Without a
preset, the checkbox is hidden. Run
`uv run flashdreams-run-v2 interactive-drive-omnidreams -- --help` to see the presets
registered in the current environment. The built-in `rtx-*` presets require
the optional NVIDIA VFX dependency, installable with
`uv pip install 'flashdreams[rtx-postprocess]'`, and supported RTX hardware.

The downloaded default scene is
`scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz`. Pass
`-- --scene scene.usdz` to use a local scene instead.

## Tests

```bash
uv run --no-sync pytest apps/interactive_drive -m ci_cpu -v
```

## Logging

set `LOGURU_LEVEL` to `DEBUG` to see more logging. Default is `INFO`.
