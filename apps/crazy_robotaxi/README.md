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

The mode menu has **CONTROLS** and **OPTIONS** buttons. The Options screen is
generated from the same typed settings tree used at startup, with pages for
game, model, renderer, presentation, live edit, runtime, and diagnostics. **SAVE**
atomically updates the user YAML without leaving the screen. **EXIT** returns
to the mode menu and changes to **EXIT WITHOUT SAVING** while the draft is
dirty. **RESET TO DEFAULTS** resets the draft. Presentation settings apply when
saved; the screen displays **RESTART
REQUIRED** when other changes need a new process.

By default, settings are loaded from
`$XDG_CONFIG_HOME/crazy-robotaxi/config.yaml`, or
`~/.config/crazy-robotaxi/config.yaml` when `XDG_CONFIG_HOME` is unset. The file
is created only after the first save. Use `--config PATH` to select another
user-authored file. YAML values are sparse overrides on the selected runner's
defaults, and retained comments survive Options saves. Explicit application CLI
arguments override YAML for the current run without rewriting the saved value;
the Options screen labels affected fields.

For example:

```yaml
schema_version: 1
game:
  gamepad_button_style: PlayStation
  taxi:
    seed: 1234
    rules:
      global_time_s: 90.0
model:
  pipeline:
    diffusion_model:
      seed: 5678
presentation:
  show_fps: true
live_edit:
  weather:
    enabled: true
```

Mode, map, and race-course selections are intentionally CLI-only and do not
appear in the YAML or Options screen. Passing `--game-mode`, `--map`, and
`--race-course` skips their corresponding startup menus; omitted selections
remain in the normal menu flow. Model diffusion and gameplay seeds are
independent. Selecting mystery items automatically enables style editing, while
rain or snow items automatically enable weather editing.

## Controls

Open **CONTROLS** from the mode menu, then choose **KEYBOARD**, **GAMEPAD**, or
**WHEEL**. Each gameplay action has primary and secondary binding slots. Select
a slot and press the desired key or device control. `Escape` cancels capture;
`Backspace`, `Delete`, or **CLEAR** unbinds the slot. Reusing an existing binding
swaps it with the previous slot. **SAVE** writes the current device without
leaving its page, and **RESET TO DEFAULTS** affects only that device.

Bindings are stored as three independent sparse YAML documents under
`$XDG_CONFIG_HOME/crazy-robotaxi/controls/`, or
`~/.config/crazy-robotaxi/controls/` when `XDG_CONFIG_HOME` is unset:
`keyboard.yaml`, `gamepad.yaml`, and `wheel.yaml`. Use the CLI-only
`--controls-dir PATH` option to select another directory. Control changes take
effect after restarting the current application process.

### Keyboard

| Control | Action |
| --- | --- |
| `W` or Up Arrow | Drive forward |
| `S` or Down Arrow | Brake, then reverse after stopping |
| `A` or Left Arrow | Steer left |
| `D` or Right Arrow | Steer right |
| `Space` | Apply the handbrake and cancel throttle |
| `R` | Restart the current game |
| `H` | Hide or show the HUD control tooltips |
| `Escape` | Return to the previous menu, then exit from the mode screen (fixed) |
| `Enter` | Submit the focused leaderboard name (fixed) |

Menu choices and leaderboard buttons can also be clicked with the mouse.

### Controller

The Gamepad Controls screen uses one button-label convention at a time. Set
`game.gamepad_button_style` to `Xbox`, `PlayStation`, or `Nintendo Switch` in
the Options screen or user-authored settings YAML. Xbox labels are the default.

| Control | Action |
| --- | --- |
| Left stick | Steer |
| Right trigger (`RT` by default) | Throttle |
| Left trigger (`LT` by default) | Brake, then reverse after stopping |
| Menu button | Restart the current game |
| Steering wheel and pedals | Use normalized steering, throttle, and brake input |

A connected gamepad or wheel takes precedence over keyboard driving input.
Menu navigation remains mouse and keyboard controlled. Gamepad and wheel
handbrake, control-hint, and live-edit actions are supported but unbound by
default. Wheel bindings use the semantic steering, throttle, brake, clutch, and
button values supplied by the runtime; physical device calibration remains a
runtime concern.

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
and `O` spawns a crossing obstacle. The same enabled actions appear as buttons
in the live-edit HUD card alongside frame-aligned ability status. Weather cannot
change while a non-base style is active. Style mode downloads its additional
model assets on first use and caches them under
`artifacts/crazy_robotaxi/live_edit`.

Map context appends authored road and landmark descriptions plus topology,
curve, and vehicle-motion clauses to the active prompt. Complete combined
prompts are encoded and retained lazily, so the first visit to a new context
may pause briefly and maps with many unique contexts retain more GPU memory.

Style, weather, map context, and guided obstacles need the Python transformer
hooks. When one of those features is enabled, the application automatically
disables native DiT acceleration while preserving the selected preset's other
settings. Native VAE acceleration stays enabled; pixel-only features such as
coins, items, and unguided obstacles keep native DiT acceleration.

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

Spawns can define both a full `prompt` for normal play and a shorter
`prompt_context` base for `--live-edit-map-context`; dynamic road and motion
clauses are appended only to the latter.
