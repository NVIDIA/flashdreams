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

Twelve OmniDreams runner configurations are registered:

| Runner | Configuration |
| --- | --- |
| `crazy-robotaxi-omnidreams` | Standard |
| `crazy-robotaxi-omnidreams-optimized-gb300` | GB300-optimized attention |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000` | RTX PRO 6000-optimized attention |
| `crazy-robotaxi-omnidreams-perf` | Performance optimized |
| `crazy-robotaxi-omnidreams-fast-perf` | Fast performance optimized |
| `crazy-robotaxi-omnidreams-rtx-5090` | Performance schedule fitted to a 32 GB GeForce RTX 5090 at 1168x640 |
| `crazy-robotaxi-omnidreams-rtx-5090-fast` | RTX 5090 schedule with the native FP8 VAE at 1024x560 (real time) |
| `crazy-robotaxi-omnidreams-responsive` | Standard with responsive model history |
| `crazy-robotaxi-omnidreams-perf-responsive` | Performance schedule with responsive model history |
| `crazy-robotaxi-omnidreams-fast-perf-responsive` | Native FP8 VAE with responsive model history |
| `crazy-robotaxi-omnidreams-optimized-gb300-responsive` | GB300-optimized attention with responsive model history |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000-responsive` | RTX PRO 6000-optimized attention with responsive model history |

The five presets whose names end in `-responsive` disable native DiT.
`fast-perf-responsive` still uses the native FP8 VAE.

The two `rtx-5090` presets fit a 32 GB GeForce RTX 5090: they run the
Cosmos-Reason1 text encoder on the host CPU, use a 4-chunk temporal window,
and use SageAttention-3 FP8 attention on Linux (cuDNN FP8 on Windows, where
SageAttention-3 is unavailable).

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

Open **OPTIONS** from **SELECT GAME MODE**. The screen is generated from the
same typed settings tree used at startup, with pages for game, model, renderer,
presentation, live edit, runtime, and diagnostics. Hover an option's label or
editor to see its description, full YAML path, and available CLI flags.
Repeated labels such as **Seed:** and **Enabled:** are identified by their
surrounding menu headings.

Model architecture and embedding formats come from the selected runner preset.
Checkpoint overrides must match that architecture. Performance and scheduler
controls remain editable. Internal checkpoint hooks, benchmark-only fields, and
raster fields unused by the game are excluded. Deprecated overrides in existing
YAML files are ignored and removed on the next save. Unknown keys produce an
error.

### Editing and saving

A checkbox changes a Boolean, a drop-down presents a fixed set of choices, and
other fields accept text. Lists and structured values use YAML flow syntax, such
as `[1000, 500]`. A blank optional field means `None`.

**SAVE** atomically writes the draft and stays on the screen, retaining YAML
comments. **RESET TO DEFAULTS** resets the draft to the selected runner's
defaults; press **SAVE** to persist the reset. **EXIT** returns to the mode menu.
While edits are pending it says **EXIT WITHOUT SAVING** and discards them.
Presentation changes take effect on save. Other changes display **RESTART
REQUIRED FOR SETTINGS TO TAKE EFFECT** because they need a new process. A
**COMMAND-LINE OVERRIDE ACTIVE** note means a launch argument controls the
current run even if you save a different value.

### Settings ownership

Crazy Robotaxi settings belong to this app. Game engine settings come from
`omnidreams_game_engine`; OmniDreams and FlashDreams settings come from the model
integration and framework. All four kinds are saved in the same `config.yaml`.
Shared source names are kept intact, so readable menu labels may differ from
their YAML keys. The exported reference's **Defined by** column identifies the
project that first declared a field, even when this app overrides its default.

### Configuration file and CLI overrides

The screen displays the settings file path. By default it is
`$XDG_CONFIG_HOME/crazy-robotaxi/config.yaml`, or
`~/.config/crazy-robotaxi/config.yaml` when `XDG_CONFIG_HOME` is unset. The file
is created after the first save. Use `--config PATH` to select another file.
Only values different from the runner preset are saved, so **MODEL** values can
vary by runner. Paths entered relative to the settings file resolve relative to
its directory.

For offline YAML edits by users or agents, keys are the Python dataclass field
names, nested along the settings tree. Start at `CrazyRobotaxiUserSettings` in
[settings.py](crazy_robotaxi/settings.py) and follow nested types and inherited
fields; `model.pipeline` uses the selected runner's pipeline config. Only fields
exposed by `iter_setting_fields` are configurable. Structured list entries also
use their dataclass field names.

To derive a key from a menu label or section heading, remove the trailing colon,
lowercase it, and replace spaces with underscores. Keep each section as a nested
mapping: **PRESENTATION → Show Fps** becomes `presentation.show_fps`, a
`show_fps` key inside the `presentation` mapping. The tooltip gives the full
path. Units are in field names: `_m` means meters, `_s` seconds, `_mps` meters
per second, `_rad` radians, and `_deg` degrees.

Application CLI arguments follow the runner's `--`. Explicit CLI values override
saved YAML for the current run without rewriting it. Where both `--flag` and
`--no-flag` are listed, the latter explicitly turns the setting off.

Model diffusion and gameplay seeds are independent. `--seed` sets both;
`--game-seed` and `--model-seed` select them separately. If both are passed,
`--model-seed` takes precedence over `--seed` for the model, and `--seed` takes
precedence over `--game-seed` for taxi gameplay. `--profile-input-latency
[TRACE_PATH]` enables profiling and sets its trace path together; omitting the
path uses the default trace file. Selecting mystery items automatically enables
style editing, while rain or snow items automatically enable weather editing.

### Export the options reference

Use `--export-options-docs PATH` to generate a Markdown reference for the
selected runner:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams -- --export-options-docs /tmp/crazy-robotaxi-options.md
```

The export lists every editable option, grouped by menu heading, with its
on-screen label, full YAML key, defining project, available CLI flags, and tooltip
description.
It also includes the application CLI help. Export uses the runner preset and
exits after writing the file; it needs no model downloads, GPU session, or
presentation window. An em dash in the CLI column means to use Options or YAML.

### Launch arguments

These arguments select startup behavior and have no Options field:

| CLI flag | Purpose |
| --- | --- |
| `--config PATH` | Select the user settings YAML file. |
| `--game-mode` | Choose `taxi` or `race` and skip the mode menu. |
| `--map PATH` | Choose a map and skip map selection. |
| `--race-course ID` | Choose a race course; requires race mode. |
| `--force-map-recompile` | Rebuild the compiled map. |
| `--ui`, `--no-ui` | Enable or disable the ImGui UI. |
| `--controls-dir PATH` | Select the separate controls settings directory. |

`--no-ui` requires explicit `--game-mode`, `--map`, and `--total-blocks`
arguments; race mode also requires `--race-course`. Omitted mode, map, and
race-course selections remain in the normal menu flow. The runner name chooses
the starting model preset before application arguments are parsed.

## Controls

Open **CONTROLS** from the mode menu, then choose **KEYBOARD**, **GAMEPAD**, or
**WHEEL**. Each gameplay action has primary and secondary binding slots. Select
a slot and press the desired key or device control. `Escape` is a valid keyboard
binding. `Backspace`, `Delete`, or **CLEAR** unbinds the slot, while **CANCEL**
stops capture without changing it. Reusing an existing binding swaps it with the
previous slot. **SAVE** writes the current device without leaving its page, and
**RESET TO DEFAULTS** affects only that device.

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
**Gamepad Button Style** to `Xbox`, `PlayStation`, or `Nintendo Switch` in
the Options screen. Xbox labels are the default.

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

Each spawn can define both a full `prompt` for normal play and a shorter
`prompt_context` base for `--live-edit-map-context`; dynamic road and motion
clauses are appended only to the latter.
