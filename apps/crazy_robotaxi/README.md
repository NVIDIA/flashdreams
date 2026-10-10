# Crazy Robotaxi

Crazy Robotaxi is an interactive FlashDreams V2 application built on the
OmniDreams world model and `omnidreams-game-engine`. Drive a taxi through
authored maps, collect fares, or race against the clock using a keyboard,
gamepad, or steering wheel.

## Requirements

Crazy Robotaxi uses the same model assets and GPU runtime as the OmniDreams
integration. See the [repository requirements](../../README.md#system-requirements)
and [OmniDreams installation guide](../../integrations_v2/omnidreams/README.md#install).
Set `HF_TOKEN` to a Hugging Face token with access to the NVIDIA OmniDreams
model repositories.

The default menus and HUD use Dear ImGui rendered through SlangPy's
Vulkan/CUDA interop on the server, including in browser mode. Native-window
mode also needs a local display.

## Quick start

From the repository root:

```bash
export HF_TOKEN=hf_...

uv sync --package flashdreams-omnidreams
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window
```

To use a browser client:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode webrtc --host 0.0.0.0 --port 8089
```

Open `http://127.0.0.1:8089/`, or use the host printed by the runner when
connecting remotely. Choose **TAXI** or **RACE**, then a map; race mode also asks
for a course. **CONTROLS** and **OPTIONS** are available from the mode menu.
The application prepares maps and loads the world model before showing the
menus. The first launch downloads missing model assets and may take time to
compile and autotune kernels.

### Runner presets

Twelve OmniDreams runner configurations are registered. The
[exported options reference](#export-the-options-reference) includes their
inheritance, shared defaults, render sizes, and exact setting differences:

| Runner | Configuration |
| --- | --- |
| `crazy-robotaxi-omnidreams` | Standard |
| `crazy-robotaxi-omnidreams-optimized-gb300` | GB300-optimized attention |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000` | RTX PRO 6000-optimized attention |
| `crazy-robotaxi-omnidreams-perf` | Performance optimized |
| `crazy-robotaxi-omnidreams-fast-perf` | Fast performance optimized |
| `crazy-robotaxi-omnidreams-rtx-5090` | Performance schedule fitted to a 32 GB GeForce RTX 5090 at 1168x640 |
| `crazy-robotaxi-omnidreams-rtx-5090-fast` | Schedule fitted to 32 GB VRAM on a GeForce RTX 5090, with the native FP8 VAE at 1024x560 |
| `crazy-robotaxi-omnidreams-responsive` | Standard with responsive model history |
| `crazy-robotaxi-omnidreams-perf-responsive` | Performance schedule with responsive model history |
| `crazy-robotaxi-omnidreams-fast-perf-responsive` | Native FP8 VAE with responsive model history |
| `crazy-robotaxi-omnidreams-optimized-gb300-responsive` | GB300-optimized attention with responsive model history |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000-responsive` | RTX PRO 6000-optimized attention with responsive model history |

The five presets whose names end in `-responsive` disable native DiT.
`fast-perf-responsive` still uses the native FP8 VAE.

The two `rtx-5090` presets fit in 32 GB of VRAM on a GeForce RTX 5090. They
run the Cosmos-Reason1 text encoder on the host CPU and use a 4-chunk temporal
window. They prefer SageAttention-3 FP8 attention
when available, falling back to cuDNN when unavailable, including on Windows.

### Application arguments

Application arguments follow `--`. For example:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams-perf --mode webrtc -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/boulevard_district.robotaxi.yaml \
  --game-time-s 90
```

List all application arguments without loading the model:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams -- --help
```

Restarting a game with `R` rebuilds its simulation and autoregressive cache
while retaining the loaded model weights. Applying startup settings requires
closing and launching the application again.

For raw model frames without menus or HUD, pass `--no-ui` together with explicit
`--game-mode`, `--map`, and `--total-blocks` values. Race mode also requires
`--race-course`. With a `webrtc` or `mp4` client this skips the Vulkan UI;
model generation still needs the OmniDreams GPU runtime.

## Options and user configuration

Open **OPTIONS** from **SELECT GAME MODE**. The screen is generated from the
same typed settings tree used at startup, with pages for game, model, renderer,
presentation, live edit, runtime, and diagnostics. Hover an option's label or
editor to see its description, full YAML path, and available CLI flags.
Repeated labels such as **Seed:** and **Enabled:** are identified by their
surrounding menu headings.

Raster width and height control the generated video's resolution. Presentation
width and height control the display resolution, with the generated video
scaled to fit.

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

Add `--export-options-docs PATH` after `--` in a game launch command to generate
a Markdown reference for the selected runner. For example, using `null` mode:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode null -- \
  --export-options-docs /tmp/crazy-robotaxi-options.md
```

Crazy Robotaxi parses and handles the flag independently of the presentation
mode. Runtime arguments follow the usual rules for the chosen mode.

The export lists every editable option, grouped by menu heading, with its
on-screen label, full YAML key, defining project, available CLI flags, and tooltip
description.
It also includes the application CLI help. The game uses the runner preset and
exits after writing the file, before preparing maps or loading the model.
An em dash in the CLI column means to use Options or YAML.

OmniDreams exports also compare all twelve runner presets with their parents,
using values from the current configs. Preset-owned fields are marked as
non-editable. The guide explains active backends, native acceleration,
responsive history, and how saved overrides affect preset comparisons.

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

### Default keyboard bindings

| Control | Action |
| --- | --- |
| `W` or Up Arrow | Drive forward |
| `S` or Down Arrow | Brake, then reverse after stopping |
| `A` or Left Arrow | Steer left |
| `D` or Right Arrow | Steer right |
| `Space` | Apply the handbrake and cancel throttle |
| `R` | Restart the current game |
| `H` | Hide or show the HUD control tooltips |
| `M` | Switch between generated video and the HD-map conditioning view |
| `Escape` | Leave the current game for the map menu |
| `K` | Cycle styles when style editing is enabled |
| `V` | Cycle weather when weather editing is enabled |
| `C` | Toggle coins when coins are enabled |
| `O` | Spawn an obstacle when obstacles are enabled |

All bindings above can be changed in **CONTROLS**. Menu navigation uses fixed
`Escape` and gamepad `B` bindings (`Circle` with PlayStation labels), independent
of the gameplay return-to-menu binding. Backing out of the mode menu exits the
application. `Enter` submits the focused leaderboard name. Menu choices and
leaderboard buttons can be clicked with the mouse.

### Default gamepad bindings

The Gamepad Controls screen uses one button-label convention at a time. Set
**Gamepad Button Style** to `Xbox`, `PlayStation`, or `Nintendo Switch` in
the Options screen. Xbox labels are the default.

| Control | Action |
| --- | --- |
| Left stick, horizontal axis | Steer |
| `RT` | Throttle |
| `LT` | Brake, then reverse after stopping |
| `A` or `LB` | Apply the handbrake and cancel throttle |
| `START / MENU` | Restart the current game |
| `BACK / VIEW` | Leave the current game for the map menu |
| `RB` | Hide or show the HUD control tooltips |
| `X` | Switch between generated video and the HD-map conditioning view |
| D-pad Left | Cycle styles when style editing is enabled |
| D-pad Right | Cycle weather when weather editing is enabled |
| D-pad Up | Toggle coins when coins are enabled |
| D-pad Down | Spawn an obstacle when obstacles are enabled |

### Steering wheels and input switching

Wheel steering, throttle, and brake axes are bound by default. Other wheel
actions start unbound; assign them in **CONTROLS**. Wheel bindings use the
semantic axes and button values supplied by the runtime; physical device
calibration remains a runtime concern.

Driving input switches to the keyboard when a driving key is pressed and to
the gamepad or wheel when it receives deliberate driving input. Connecting an
idle controller alone does not take over keyboard driving.

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

Live-edit features are disabled by default. Enable them on the **LIVE EDIT**
Options page or with application arguments:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window -- \
  --live-edit-coins \
  --live-edit-items \
  --live-edit-weather \
  --live-edit-style \
  --live-edit-obstacle \
  --live-edit-map-context
```

When enabled, `C` toggles coins, `K` cycles style skins, `V` cycles weather,
and `O` spawns a crossing obstacle. The same enabled actions appear as buttons
in the live-edit HUD card alongside frame-aligned ability status. Weather cannot
change while a non-base style is active. Required additional checkpoints are
downloaded during application startup when the features are enabled, unless
explicit checkpoint paths are configured. Downloaded assets are cached under
`artifacts/crazy_robotaxi/live_edit` relative to the working directory.

Map context appends authored road and landmark descriptions plus topology,
curve, and vehicle-motion clauses to the active prompt. Complete combined
prompts are encoded and retained lazily, so the first visit to a new context
may pause briefly and maps with many unique contexts retain more GPU memory.

Style, weather, map context, and obstacles with a positive `guide_scale` need
the Python transformer hooks. Enabling any of them automatically disables
native DiT acceleration and logs the reason. Native VAE acceleration remains
available. Coins, nitro-only items, and unguided obstacles can retain native
DiT. The default item types include rain, snow, and mystery, which also enable
weather and style and therefore disable native DiT.

## Authored maps

Maps are strict semantic `.robotaxi.yaml` documents. See the
[map format guide](../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md) for
nodes, roads, profiles, traffic, spawns, and race courses. The map menu discovers
bundled maps and maps in the directory of the path supplied by `--map`.

From the repository root, validate, compile, or preview the bundled boulevard
map without loading a world model or using a GPU:

```bash
ROBOTAXI_MAP=apps/crazy_robotaxi/crazy_robotaxi/maps/boulevard_district.robotaxi.yaml

uv run --package crazy-robotaxi crazy-robotaxi-map validate "$ROBOTAXI_MAP"
uv run --package crazy-robotaxi crazy-robotaxi-map compile "$ROBOTAXI_MAP"
uv run --package crazy-robotaxi crazy-robotaxi-map preview \
  "$ROBOTAXI_MAP" --output boulevard.svg
uv run --package crazy-robotaxi crazy-robotaxi-map preview-spawn \
  "$ROBOTAXI_MAP" --spawn original_area_start --output boulevard_spawn.png
```

Compiled scene archives are cached under
`$FLASHDREAMS_CACHE_DIR/omnidreams-game-engine/game-maps`, defaulting to
`~/.cache/flashdreams/omnidreams-game-engine/game-maps`. Use
`compile --force-map-recompile` to rebuild an archive. The application accepts
the source YAML directly and compiles it as needed.

Each spawn can define both a full `prompt` for normal play and a shorter
`prompt_context` base for `--live-edit-map-context`; dynamic road and motion
clauses are appended only to the latter.
