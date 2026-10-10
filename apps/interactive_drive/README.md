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
| `3` | Show the PhysX collider view, where the integration simulates colliders. |

An integration whose model has only ever seen a car going forwards sets
`reverse=False`, and every way of asking for reverse brakes instead.

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
it with GPU textures; the application does not use CSS. The minimap is the
overhead camera this app's rasterizer appends to the rig, so an integration
whose backend renders its own rig leaves it out with `bev=False`.

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
| `--scene PATH` | Use a local scene instead of downloading the default one. A USDZ for the built-in loader, or whatever an integration's own scene loader reads, which may be a directory. |
| `--sample PATH` | A recording of the same drive, for a model that takes its prompt, framing or opening frames from real footage rather than from the scene. Ignored by scenes that carry their own. |
| `--start-frame N` | Frame of that recording to open on. Default: `0`. |
| `--prompt TEXT` | Override the prompt the scene or sample conditions on, for this run and every restart in it. Without it, a restart keeps whatever the scene supplies. |
| `--camera NAME[,NAME...]` | Select a camera from the scene, or a comma-separated rig of them. Several cameras are conditioned on together and presented as a grid, as close to square as they go, in the order given. Default: `camera_front_wide_120fov`. |
| `--present-camera NAME` | Present one camera of the rig alone, at its own size, instead of the grid. The rest are still conditioned on. |
| `--variant NAME` | Select the scene's initial-frame and prompt variant. Default: `default`. |
| `--total-blocks N` | Stop after this many generated blocks; `0` runs until the session is stopped. Default: `0`. |
| `--fps N` | Set the application frame rate. Default: `30`. |
| `--width N` | Set the output width. Default: `1280` (`1168` for the perf app). |
| `--height N` | Set the output height. Default: `704` (`640` for the perf app). |
| `--view {rgb,hdmap,physx}` | Select the initial RGB, HD-map conditioning, or PhysX collider view. Default: `rgb`. The collider overlay is drawn from the rig's first camera only, since it debugs the physics rather than the conditioning, so in a grid it appears in the first cell and the other cells show their own conditioning. `physx` is offered only where the integration simulates colliders. |
| `--no-ui` | Present model output directly without creating the HUD or rendering its BEV minimap. |
| `--game-mode` | Enable the speed limit and collisions with scene actors and static map geometry. |
| `--postprocess-preset NAME` | Start with a registered video post-processing preset enabled. Default: none. |
| `--postprocess-device DEVICE` | Select the postprocessor device independently. Default: `cuda:0`. |
| `--world-model-device DEVICE` | Select the model device. Default: `cuda:0`. |
| `--raster-device DEVICE` | Select the Ludus raster device independently. Default: `cuda:0`. |
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
    --timeout unbound \
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

The HUD is composited into the generated frame, so its sharpness is the frame's.
A model generating something smaller than 1280x704 has its panels scaled down to
fit and then magnified again by whatever displays the stream, which is what makes
the text look soft. An integration meant to be looked at rather than measured
asks for `full_size_hud=True`, and the session then composites into the smallest
whole multiple of the drive that the HUD fits in: one 832x480 view becomes
1664x960, which is what four of them already tile to. It costs four times the
pixels to overlay and encode and does nothing for the drive itself, so it is off
by default.

The HUD's view button cycles through **RGB → HDMAP → PHYSX**, leaving PHYSX out
for an integration that sets `physics=False`: the collider overlay is drawn from
the physics world, so without one there is no frame to show.

### SwiftVR on two GPUs

Install OmniDreams, Interactive Drive, and the SwiftVR package from PR #606:

```bash
uv sync --package flashdreams-omnidreams --package flashdreams-swiftvr \
    --extra interactive-drive --inexact
```

The world model and Ludus rasterizer can share one GPU while SwiftVR runs on
another. This example assigns OmniDreams and Ludus to `cuda:1`, assigns the
SwiftVR 2x postprocessor to `cuda:0`, and generates 832x464 frames before
upscaling them to 1664x928:

```bash
uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams-optimized-gb300 \
    --mode webrtc --host 0.0.0.0 --port 8089 -- \
    --width 832 --height 464 \
    --world-model-device cuda:1 --raster-device cuda:1 \
    --postprocess-preset swiftvr-2x --postprocess-device cuda:0
```

Use `swiftvr-4x` for the PR #606 4x preset. CUDA ordinals are assigned after
`CUDA_VISIBLE_DEVICES` is applied, so set that environment variable explicitly
when physical GPU placement matters. The HUD displays the fixed launch-time
device and preset choices; its **Post-processing** checkbox enables or bypasses
the configured processor without reloading either model. Presentation stays on
the selected postprocessor GPU in both modes, so toggling does not change the
Vulkan/CUDA interop device.

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
