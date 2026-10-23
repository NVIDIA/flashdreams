<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Sim Auto

> Prepared-scene LiDAR rendering uses independently written Apache-2.0 geometry
> and projection implementations. See the [test specification](tests/specs/lidar_geometry_projection.md).

Drive a world model using a scene's HD map, prompt and opening images.
The model supports **1–11 RGB views**; **1, 4 and 7** are popular configurations.
This RC's prepared-scene demo supports ordered subsets of its seven supplied
cameras. **Start with one RGB view on one GPU.** Optional
[LiDAR generation](#run-with-lidar) uses the same scene but is substantially slower.

## Interactive Drive previews

Short, straightforward drives with the live UI and default residential scene.
Playback follows simulation time.

| One RGB view | Four RGB views | Seven RGB views |
| --- | --- | --- |
| ![One front-wide RGB camera](../../docs/source/_static/model_clips/cosmos3_nano_sim_auto/drive-1-view.avif) | ![Four RGB cameras](../../docs/source/_static/model_clips/cosmos3_nano_sim_auto/drive-4-view.avif) | ![Seven RGB cameras](../../docs/source/_static/model_clips/cosmos3_nano_sim_auto/drive-7-view.avif) |

| One RGB view + LiDAR | Seven RGB views + LiDAR |
| --- | --- |
| ![One front-wide RGB camera and generated LiDAR](../../docs/source/_static/model_clips/cosmos3_nano_sim_auto/drive-1-view-lidar.avif) | ![Seven RGB cameras and generated LiDAR](../../docs/source/_static/model_clips/cosmos3_nano_sim_auto/drive-7-view-lidar.avif) |

## Install

Use Linux, an NVIDIA GPU with **80 GB+ VRAM**, an **R580+ driver** and
**100 GB+ free storage**; see the [system requirements](../../README.md#system-requirements).
The map renderer also needs a CUDA 13 toolkit (`nvcc`) and a C++ compiler.

The sidebar and sensor layout require **Vulkan graphics with CUDA interop on the
server GPU**, including in WebRTC browser mode. CUDA support alone is not enough
for this UI. If Vulkan/CUDA interop is unavailable, use
[WebRTC without the UI](#run-without-the-ui).

Before starting, confirm your Hugging Face account can access both
[the model](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto) and
[the scene dataset](https://huggingface.co/datasets/nvidia/omni-dreams-scenes).
A browser must be able to reach the server over HTTP and WebRTC.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then check
out the repository and run from its root in the same terminal throughout:

```bash
git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
export HF_TOKEN="<your-hugging-face-token>"
uv sync --frozen --inexact --python 3.12 \
  --package flashdreams-cosmos3-nano-sim-auto --extra interactive-drive \
  --package flashdreams --group cuda13
```

Skip cloning for an existing checkout. You can use
`uv run --no-sync hf auth login` instead of setting `HF_TOKEN`.
If your CUDA 13 toolkit is not already configured, set its location and check it:

```bash
export CUDA_HOME="<your-CUDA-13-toolkit-directory>"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version
```

For a toolkit installed in this Python environment, its root is
`$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cu13`.

The first launch downloads the
[Cosmos3 Nano Sim Auto checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto),
tokenizer and video codecs, and compiles the map renderer. Model preparation also
caches `lidar_vae/config.json` and `lidar_vae/diffusion_pytorch_model.safetensors`
from the same HF repository and revision (about 596 MB for the LiDAR weights).
These downloads use `HF_HOME` / `HF_HUB_CACHE`. Allow extra time for this setup.
To use downloaded weights, add `--checkpoint /path/to/model.safetensors.index.json`
after `--` in a launch command.

## Scenes

Interactive Drive downloads and caches these prepared bundles from
[nvidia/omni-dreams-scenes](https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes/synthetic-first-frames)
during initialization:

- `clipgt-residential-iter1500` (default)
- `clipgt-residential-rain-iter1500`
- `clipgt-residential-snow-iter1500`

All three appear in the **Scene** picker. Each bundle supplies its own geometry,
camera calibration, prompt and matching opening images. The batch application
accepts the same built-in names and downloads only the selected scene.

ZIPs use the Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE`); extracted bundles
use `FLASHDREAMS_CACHE_DIR` (default: `~/.cache/flashdreams`). Subsequent runs
reuse the cached assets. To use a local bundle, pass `--scene /path/to/bundle`;
local paths bypass downloads. Keep the whole folder, including `scene.json`,
`sample.json`, geometry JSON files, `initial_rgb/`, its README and
`LIDAR-PROFILE-LICENSE.txt`.
Custom bundles must use the same seven-camera rig, coordinates and hashed asset
inventory; see the [prepared-scene format](impl/conditioning/world_scenario/prepared_scene.py).

## Run in a browser

These commands run on one GPU. To select a GPU, optionally prefix a launch
command with `CUDA_VISIBLE_DEVICES=<gpu-index>`.

Start with one front-wide view:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080
```

Open `http://<server-address>:8080/`; use `localhost` when running locally.
The residential, rain and snow bundles download automatically. First startup
may compile the map renderer; denoiser compilation is off by default.
A successful run advances from the opening image to generated frames.

Defaults select one front-wide RGB camera, the residential scene, opening frame
zero and model seed zero. Presentation is continuous with blocking backpressure.
Application options go after `--`; keep `--host` and `--port` explicit for a
remote browser at a fixed address.

Each scene has limited HD map coverage. Driving outside it can make generated
motion appear unresponsive to steering or acceleration. A red **Map coverage
warning** appears in the sidebar when coverage is lost; sparse map gaps can
also trigger it.

Press **R** or click **Restart rollout** to return to the scene's starting position
and opening images if the warning appears or generation quality degrades.
Restarting is manual and takes effect after the current generation step finishes.
The warning clears when you return to map coverage or the new rollout begins.

### Check driving and the UI

- Use **W/S** for forward/reverse, **A/D** to steer and **Space** to brake.
- Switch between residential, rain and snow in **Scene**; check weather and
  driving. Leave **Variant** at `default`.
- Try both **Layout** options, focus individual cameras/sensors, then return
  to the overview.
- Toggle **RGB/HDMAP** with **View** or **1/2**; check the map and RGB correspond.
- Press **R** or click **Restart rollout** to start a fresh drive from the scene's
  opening images. The shortcut also works with `--no-ui`.
- Resize the browser and try other visible controls, including post-processing
  if available. Check for frozen views, broken layouts or lost keyboard input.

Repeat these checks for each configuration. Stop with **Ctrl+C** between
launches and reload the browser after restarting the server.

For four views covering the front, right, rear and left:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 -- --views 4
```

For seven views on the same GPU:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 -- --views 7
```

Seven RGB views can be very slow on one GPU. Check that all seven update
consistently while driving, changing layouts and focusing cameras.
Leave `--start-frame` at its default of zero to align the map with the opening images.
On a machine with a display,
replace `--mode webrtc` with `--mode native-window` to open a desktop window.

`--views 1`, `4` and `7` select fixed camera presets in rig order. For a custom
ordered subset, use `--camera` instead, for example
`--camera camera_front_wide_120fov,camera_rear_tele_30fov`.
Do not combine `--views` and `--camera`.

To start in rain or snow, add `--scene clipgt-residential-rain-iter1500` or
`--scene clipgt-residential-snow-iter1500` after `--`. Each variant includes
its own prompt and matching opening images.

On Slurm, [`probes/launch.sbatch`](probes/launch.sbatch) runs Interactive Drive or
a recorded-trajectory video on `--nodes N` x 4 GPUs (`MODE=drive|batch`).
Drive mode shows the UI by default; append `--no-ui` after the script to hide it.
Set `SCENE` to choose a scene. In drive mode, `LIDAR=1` enables automatic HF
LiDAR weights; `LIDAR_CHECKPOINT` optionally overrides them.

For longer sessions, add `--compile-denoise default --decoder-compile --compile-control-encode`
after `--` to enable model compilation.
Cold compilation can take several minutes.

### Run without the UI

If the server cannot initialize Vulkan/CUDA interop, add `--no-ui` after `--` to
stream the model output directly. WebRTC needs no display on the server:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 -- --no-ui
```

Open `http://<server-address>:8080/`. **WASD**, **Space** and **1/2** still control
driving, braking and RGB/HDMAP selection. The sidebar, status overlays, scene
picker and interactive layout controls are unavailable. Select the scene and
cameras at launch with `--scene` and either `--views` or `--camera`; the same
`--no-ui` flag works with multiple cameras and `--lidar`. Press **R** to restart
the rollout. Stop with **Ctrl+C** and relaunch to change scenes.

Model inference and map rendering still require CUDA. `--mode native-window`
still requires Vulkan and a local display, even with `--no-ui`. With the UI
disabled, CUDA interop is optional; frames can instead be copied through CPU
memory.

## Save a short video

Generate a 49-frame RGB video along the scene's recorded trajectory on one GPU:

```bash
uv run --no-sync flashdreams-run-v2 cosmos3-nano-sim-auto \
  --mode mp4 --output-path artifacts/sim-auto.mp4 --timeout unbound -- \
  --views 1 --frames 49
```

Open `artifacts/sim-auto.mp4` to check the result.
In batch mode, `--views N` selects the first N cameras from the scene, rather than
the interactive presets; `--camera` selects the output view.

## Run with LiDAR

Add `--lidar` to generate RGB and LiDAR together using the same scene bundle.
The codec weights download automatically using the model's HF credentials and cache.
This is substantially slower than RGB-only on a single GPU.

For one front-wide RGB view with LiDAR:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 -- --lidar
```

For seven RGB views with LiDAR:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 -- --views 7 --lidar
```

Use `--views 4 --lidar` for four RGB cameras and LiDAR. Repeat the driving/UI
checks, including focusing LiDAR and switching between generated output and
the **HDMAP** control scan.
Check all cameras and LiDAR after scene changes, layout changes and restarts.

Open `http://<server-address>:8080/` to see RGB and generated point clouds.
Point clouds are the default. Choose another LiDAR display with `--lidar-view range` or
`--lidar-view intensity`; **HDMAP** displays the control scan.
See [LiDAR details](LIDAR.md) for codec formats and output timing.

## Runtime on one RTX PRO 6000

| Views | Demo as shipped (fps) | Model only (fps) |
|---:|---:|---:|
| 1 | 20.8 | 25.3 |
| 4 | 4.6 | 5.9 |
| 7 | 2.3 | 2.8 |

fps is frames generated per second for each camera: each chunk produces 12 frames,
so fps = 12 / median chunk time.

- Demo as shipped: the HD-map controls are drawn and encoded on the same GPU inside
  every chunk, as in Interactive Drive.
- Model only: the same chunk without that control preparation, i.e. the transformer
  and decoder alone.

The difference is the control preparation, about 20% of the chunk.

Setup: this release, Hugging Face GA checkpoint, scene `clipgt-residential-iter1500`
(7 cameras, 480p), 1205 frames per run, 100 chunks of which 95 are measured, bf16,
4 denoising steps, no `torch.compile` (the default), one RTX PRO 6000 Blackwell (96 GB),
free-boosting clock.

## Reproducibility

For repeatable runs, use the same code, model, scene and seed. Record your launch
command and software versions when sharing results.
