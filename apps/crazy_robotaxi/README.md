# Crazy Robotaxi

Crazy Robotaxi is an interactive FlashDreams V2 application built on the
OmniDreams world model and `omnidreams-game-engine`. Drive a taxi through
authored maps, collect fares, or race against the clock using a keyboard,
gamepad, or steering wheel.

## Requirements

Crazy Robotaxi uses the same model assets and GPU runtime as the OmniDreams
integration. Set `HF_TOKEN` to a token with access to the NVIDIA OmniDreams
repositories. See the [OmniDreams integration guide](../../integrations_v2/omnidreams/README.md)
for the supported platform, model preparation, and controller setup.

## Quick start

From the repository root:

```bash
export HF_TOKEN=<YOUR-HF-TOKEN>

uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams python \
  integrations_v2/omnidreams/impl/omnidreams_singleview/tools/sync_thirdparty.py sync

uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window
```

Native-window mode requires a local display and SlangPy's Vulkan/CUDA interop.
To use a browser client instead:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode webrtc --host 0.0.0.0 --port 8089
```

Open `http://127.0.0.1:8089/`, or use the host printed by the runner when
connecting remotely. The first run downloads model assets and may take time to
compile and autotune kernels.

Three OmniDreams runner configurations are registered:

| Runner | Configuration |
| --- | --- |
| `crazy-robotaxi-omnidreams` | Standard |
| `crazy-robotaxi-omnidreams-perf` | Performance optimized |
| `crazy-robotaxi-omnidreams-fast-perf` | Fast performance optimized |

The performance configurations require the native DiT sources to be prepared
once:

```bash
uv run --package flashdreams-omnidreams omnidreams-prepare --perf
```

Application arguments follow `--`. For example:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams-perf --mode webrtc -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/boulevard_district.robotaxi.yaml \
  --game-time-s 90
```

Run the application with `-- --help` to list all game options. Restarting a
game rebuilds its simulation and autoregressive cache without reloading the
model.

## Controls

### Keyboard

| Control | Action |
| --- | --- |
| `W` or Up Arrow | Drive forward |
| `S` or Down Arrow | Reverse |
| `A` or Left Arrow | Steer left |
| `D` or Right Arrow | Steer right |
| `Space` | Apply the handbrake and cancel throttle |
| `R` | Restart the current game |
| `Escape` | Return to the previous menu, then exit from the mode screen |
| `Enter` | Submit the focused leaderboard name |

Menu choices and leaderboard buttons can also be clicked with the mouse.

### Controller

| Control | Action |
| --- | --- |
| Left stick | Steer |
| Right trigger (`RT` / `R2` / `ZR`) | Throttle |
| Left trigger (`LT` / `L2` / `ZL`) | Brake |
| `R` / `RB` / `R1` (hold) | Select reverse gear |
| Start / Menu / Plus | Restart the current game |
| Steering wheel and pedals | Use normalized steering, throttle, and brake input |

A connected gamepad or wheel takes precedence over keyboard driving input.
Gamepads do not currently control menus, the handbrake, or live-edit actions.

## Race mode

Bundled maps can define ordered race courses. Start with the included raceway
and select the `grand-prix` course in the menu:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/flashdreams_raceway.robotaxi.yaml \
  --game-mode race
```

Race times are stored per map and course. Use `--race-times PATH` to choose a
different leaderboard file.

## Optional live-edit abilities

Live-edit features are disabled by default. Enable them with application
arguments:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window -- \
  --live-edit-coins \
  --live-edit-items \
  --live-edit-weather \
  --live-edit-style
```

When enabled, `C` toggles coins, `K` cycles style skins, `V` cycles weather,
and `O` spawns a crossing obstacle. Style mode downloads its additional model
assets on first use and caches them under `artifacts/crazy_robotaxi/live_edit`.
Text-edit and obstacle guidance require a non-native DiT configuration; the
application rejects incompatible configurations before generation begins.

## Authored maps

Maps are strict semantic `.robotaxi.yaml` documents. Validate or preview them
without loading a model:

```bash
uv run --package crazy-robotaxi crazy-robotaxi-map validate path/to/city.robotaxi.yaml
uv run --package crazy-robotaxi crazy-robotaxi-map compile path/to/city.robotaxi.yaml
uv run --package crazy-robotaxi crazy-robotaxi-map preview \
  path/to/city.robotaxi.yaml --output city.svg
uv run --package crazy-robotaxi crazy-robotaxi-map preview-spawn \
  path/to/city.robotaxi.yaml --spawn taxi_start --output taxi_start.png
```

The complete spawn-image authoring path needs both Qwen Image Edit 2511 and the
OmniDreams world model. Run it through the application integration:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window -- \
  --map path/to/city.robotaxi.yaml --generate-spawn-images
```

Crazy Robotaxi first renders the semantic road into Qwen's source image and
uses `prompt` plus `time_of_day` (`dawn`, `day`, `dusk`, or `night`) as the
scenery instruction. It restores the exact semantic road, boundaries, curbs,
and markings through a road-only mask, adds deterministic luminance grain only
to the dark asphalt pixels, then sends that result through eight stationary
OmniDreams chunks. The grain brightness follows `time_of_day`; markings and
Qwen-generated scenery remain unchanged. The last generated frame becomes the
managed spawn PNG and the new gameplay seed, which lets the world model add
asphalt texture without moving the camera or vehicle. The game cache and
simulation are reset afterward, so gameplay still begins at AR index zero.

Managed images are written under `<map-id>.spawn-images/`; the YAML is updated
and the map is recompiled after both the Qwen and OmniDreams stages. Existing
authored images outside that folder are never overwritten. Add
`--force-spawn-images` to regenerate an existing managed image. The default is
eight settling chunks (about two seconds of generated video); override it with
`--spawn-image-settle-blocks N`.
Generation is never triggered by ordinary map compilation or game startup.

The offline command below remains available when only a Qwen draft is wanted.
It cannot perform the final world-model settlement because it deliberately
does not construct a model pipeline:

```bash
uv run --package crazy-robotaxi crazy-robotaxi-map generate-spawns \
  path/to/city.robotaxi.yaml
```
