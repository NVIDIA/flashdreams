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

## Options and user configuration

Every selection menu has an **OPTIONS** button. The Options screen is generated
from the same typed settings tree used at startup, with pages for launch, game,
model, renderer, presentation, live edit, runtime, and diagnostics. **SAVE**
atomically updates the user YAML and **DISCARD** (or `Escape`) abandons the
draft. Presentation settings apply when saved; the screen displays **RESTART
REQUIRED** when other changes need a new process.

By default, settings are loaded from
`$XDG_CONFIG_HOME/crazy-robotaxi/config.yaml`, or
`~/.config/crazy-robotaxi/config.yaml` when `XDG_CONFIG_HOME` is unset. The file
is created only after the first save. Use `--config PATH` to select another
user-authored file. YAML values are sparse overrides on the selected model
preset, and retained comments survive Options saves. Explicit application CLI
arguments override YAML for the current run without rewriting the saved value;
the Options screen labels affected fields.

For example:

```yaml
schema_version: 1
launch:
  mode: race
  map: apps/crazy_robotaxi/crazy_robotaxi/maps/flashdreams_raceway.robotaxi.yaml
  race_course: grand-prix
game:
  taxi:
    seed: 1234
    rules:
      global_time_s: 90.0
model:
  preset: omnidreams-perf
  pipeline:
    diffusion_model:
      seed: 5678
presentation:
  show_fps: true
live_edit:
  weather:
    enabled: true
```

Setting all three launch selections skips their corresponding startup menus;
leaving any selection null keeps that menu in the normal flow. Model diffusion
and gameplay seeds are independent. Selecting mystery items automatically
enables style editing, while rain or snow items automatically enable weather
editing.

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
| `H` | Hide or show the HUD control tooltips |
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
and `O` spawns a crossing obstacle. The same enabled actions appear as buttons
in the live-edit HUD card alongside frame-aligned ability status. Weather cannot
change while a non-base style is active. Style mode downloads its additional
model assets on first use and caches them under
`artifacts/crazy_robotaxi/live_edit`.

Style, weather, and guided obstacles need the Python transformer hooks. When
one of those features is enabled, the application automatically disables native
DiT acceleration and logs the reason. Native VAE acceleration and the remaining
performance configuration stay enabled; pixel-only features such as coins,
items, and unguided obstacles keep native DiT acceleration.

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
