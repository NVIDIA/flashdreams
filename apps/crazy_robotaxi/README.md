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

Eight OmniDreams runner configurations are registered:

| Runner | Configuration |
| --- | --- |
| `crazy-robotaxi-omnidreams` | Standard |
| `crazy-robotaxi-omnidreams-perf` | Performance optimized |
| `crazy-robotaxi-omnidreams-fast-perf` | Fast performance optimized |
| `crazy-robotaxi-omnidreams-responsive` | Standard with responsive model history |
| `crazy-robotaxi-omnidreams-perf-responsive` | Performance schedule with responsive model history |
| `crazy-robotaxi-omnidreams-fast-perf-responsive` | Native FP8 VAE with responsive model history |
| `crazy-robotaxi-omnidreams-optimized-gb300-responsive` | GB300-optimized attention with responsive model history |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000-responsive` | RTX PRO 6000-optimized attention with responsive model history |

The `crazy-robotaxi-omnidreams-perf` and
`crazy-robotaxi-omnidreams-fast-perf` presets use native DiT acceleration and
require its CUDA sources to be prepared once:

```bash
uv run --package flashdreams-omnidreams omnidreams-prepare --perf
```

The five presets whose names end in `-responsive` disable native DiT and do
not require that setup.

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
  --live-edit-style \
  --live-edit-map-context
```

When enabled, `C` toggles coins, `K` cycles style skins, `V` cycles weather,
and `O` spawns a crossing obstacle. Style mode downloads its additional model
assets on first use and caches them under `artifacts/crazy_robotaxi/live_edit`.
Map context appends authored road and landmark descriptions plus topology,
curve, and vehicle-motion clauses to the active prompt. Complete combined
prompts are encoded and retained lazily, so the first visit to a new context
may pause briefly and maps with many unique contexts retain more GPU memory.

Prompt editing requires the Python transformer. When map context is enabled,
the application disables only native DiT acceleration and otherwise preserves
the selected preset, including its scheduler, seed, native VAE, and finalize
settings. Style/weather without map context and obstacle guidance still require
a compatible non-native DiT configuration.

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

Spawn variants can define both a full `prompt` for normal play and a shorter
`prompt_context` base for `--live-edit-map-context`; dynamic road and motion
clauses are appended only to the latter.
