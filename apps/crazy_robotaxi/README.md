# Crazy Robotaxi

Crazy Robotaxi is an interactive FlashDreams V2 application built on the
OmniDreams world model and `omnidreams-game-engine`. Drive a taxi through
authored maps, collect fares, race against the clock, or explore freely using a
keyboard, gamepad, or steering wheel.

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
connecting remotely. Choose **TAXI**, **RACE**, or **FREE-ROAM**, then a map;
race mode also asks for a course. **CONTROLS** and **OPTIONS** are available from
the mode menu.
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

## Headless driving tests

The app registers `--mode robotaxi-debug` through FlashDreams' output-mode
entry point. It runs the same simulation, rules, Ludus conditioning, and
world model as the interactive demo, with no window or ImGui rendering.
Select the map and game mode through the existing CLI arguments; race mode
also requires an explicit course.

From the repository root, run a three-second check of Raceway's starting
straight:

```bash
uv run --package flashdreams-omnidreams python \
  apps/crazy_robotaxi/tests/run_headless_scenario.py raceway_start \
  --output-dir /tmp/robotaxi-raceway-start
```

Each run needs a fresh output directory and `ffmpeg`/`ffprobe` on `PATH`.
The runner uses the real model, checks all videos' frame counts and telemetry
alignment, then summarizes the physical observations. Other scenarios are
`flat_acceleration` and `curb_impact` use the dedicated flat/curb test
map under `tests/maps/`. `hill_climb` drives from a flat approach onto a 32 m
hill using `debug_hill.robotaxi.yaml`. Physics measurements are reported for
inspection; the capture check does not assert that the vehicle handles correctly.
`--slug` selects another model
preset, `--config` selects application settings, and `--snapshot-every`
controls PNG sampling. Assertions fail with a nonzero exit status and
leave the capture available for inspection.

For an arbitrary map and driving script, invoke the demo directly:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams-perf --mode robotaxi-debug \
  --debug-output-dir /tmp/robotaxi-custom-test --debug-snapshot-every 15 -- \
  --map apps/crazy_robotaxi/tests/maps/debug_flat.robotaxi.yaml \
  --game-mode race --race-course debug-lap \
  --drive-script apps/crazy_robotaxi/tests/scenarios/curb_impact.yaml --seed 42
```

`--drive-script` supplies commands directly to the simulation, bypassing
keyboard input and its wall-clock sampling. A script contains only a
non-empty `steps` list:

```yaml
steps:
  - {frames: 60, throttle: 1}
  - {frames: 18, throttle: 1, steer: -0.7, steer_is_direct: true}
  - {frames: 42, throttle: 1}
  - {frames: 60}
```

Every step sets its controls independently; omitted controls are neutral.
`frames` is a positive integer at 30 frames/second. `throttle` and `brake`
range from 0 to 1, and `steer` from -1 (right) to 1 (left). Boolean controls
are `handbrake`, `reverse`, `stop`, and `steer_is_direct`. These are raw
`DriverCommand` values with manual driving enabled. By default steering
changes the steering angle over time; `steer_is_direct: true` sets a target
angle, with the game's normal steering response rate.

The script includes the initial pose as frame zero, as ordinary gameplay
does. Its frame count sets the run duration, so omit `--total-blocks`.
Scripted runs disable hidden prewarm blocks and use blocking presentation
queues to preserve every frame. Models still generate full chunks: a short
last chunk is padded with neutral controls internally, and its unused tail
is excluded from every exported view and telemetry record. If the game ends
or a runtime timeout interrupts the script, the manifest reports an
incomplete capture.

The output directory contains:

- `generated.mp4` and `hdmap.mp4`: separate, synchronized videos at the model
  resolution and 30 frames/second.
- `physics.mp4`: an overhead view of the measured car pose, contact bounds,
  inset native chassis, curb boxes, heading, and recent path. A curb-response
  flag marks the game's proximity rebound. The model can miss a maneuver in
  generated images; use this view and telemetry to inspect the simulation.
- `frames/{generated,hdmap,physics}/*.png`: synchronized screenshots at
  the requested interval, plus contact frames and the first/final frames. Use
  `--debug-snapshot-every 1` to save every frame.
- `telemetry.jsonl`: one record per exported frame, including the applied
  command, simulation timestamp, game state, camera pose, vehicle state,
  curb/actor responses, collider snapshots, and physical vehicle state **before ground alignment**.
  `ground_error_m` measures that physical pose against the compiled road
  mesh, so render alignment cannot conceal suspension bounce or falling
  below the road.
- `manifest.json`: requested/written frame counts, contact frames, view names, completion status,
  dimensions, map/course, script, model preset, and seeds.
- `inputs/`: copies of the authored map and resolved driving script.

CPU regression checks for controls, final-chunk trimming, output alignment,
and map compilation are runnable without model weights:

```bash
uv run pytest -m ci_cpu apps/crazy_robotaxi/tests/test_headless_debug.py \
  apps/crazy_robotaxi/tests/test_application.py
```

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

### Pacific Breakwater

Select **Pacific Breakwater** for a closed coastal circuit with waterfront
straights, marina bends, technical esses and a cape hairpin. All four lanes
follow the race direction, with no NPC traffic. Choose `coastal-sprint` for
one lap or `coastal-grand-prix` for three laps.

The bundled 1280x704 initial frame uses the compiled spawn's calibrated front
camera, with a visible silver hood. Its road surface, curbs and white markings
are projected directly from the semantic map. Generated scenery and asphalt
texture are composited around that geometry; the generated road layout is
discarded.
The full spawn prompt establishes the coastal setting for normal play; the
shorter base and nine road fragments keep lighting and hood framing consistent
while adding local landmarks with dynamic prompts enabled. The runtime supplies
turn direction and vehicle motion.

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams --mode native-window -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/pacific_breakwater.robotaxi.yaml \
  --game-mode race --race-course coastal-sprint --live-edit-dynamic-prompts
```

`--live-edit-dynamic-prompts` activates the road fragments and disables native DiT
acceleration, as described below. Omit it to use the full static spawn prompt.

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
  --live-edit-dynamic-prompts
```

When enabled, `C` toggles coins, `K` cycles style skins, `V` cycles weather,
and `O` spawns a crossing obstacle. The same enabled actions appear as buttons
in the live-edit HUD card alongside frame-aligned ability status. Weather cannot
change while a non-base style is active. Required additional checkpoints are
downloaded during application startup when the features are enabled, unless
explicit checkpoint paths are configured. Downloaded assets are cached under
`$FLASHDREAMS_CACHE_DIR/crazy-robotaxi/live_edit`.

Dynamic prompts append selected fragments for the current road context, next
map-node type, next map-node context, upcoming road-curve direction, and taxi
motion state. Each fragment can be included independently in the live settings.
Complete combined prompts are encoded and retained lazily, so the first visit
to a new combination may pause briefly and sessions with many unique
combinations retain more GPU memory.

Style, weather, dynamic prompts, and obstacles with a positive `guide_scale` need
the Python transformer hooks. Enabling any of them automatically disables
native DiT acceleration and logs the reason. Native VAE acceleration remains
available. Coins, nitro-only items, and unguided obstacles can retain native
DiT. The default item types include rain, snow, and mystery, which also enable
weather and style and therefore disable native DiT.

## Authored maps

The existing YAML map format supports optional node `pose.z_m` heights. Roads
transition smoothly between level node surfaces; paint, curbs, collision and
checkpoints follow the resulting ground mesh. See
[Road elevation](../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md#road-elevation)
for authoring rules. FlashDreams Raceway demonstrates hills after its flat
starting section.

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
`prompt_context` base for `--live-edit-dynamic-prompts`; selected dynamic
fragments are appended only to the latter.
The static `prompt` persists throughout driving: describe the camera,
visual style and regional scenery without fixing the current road
placement, lane count, upcoming turn or landmark position. Spawn-specific
geometry belongs in the initial image;
location-specific descriptions belong in the dynamic map fragments.
Keep weather, haze, fog, lighting and time of day out of both map prompts and
scenery fragments. Weather effects belong to the separate live-edit weather
controls.

The bundled maps each have a dedicated hood-view initial frame. FlashDreams
Raceway shows grandstands, pit garages and a city skyline; Boulevard District
shows a leafy residential boulevard with yellow traffic separation. Both
frames use the compiled spawn's road projection, with map-relative assets
also used by the map and course menus. Their spawn prompts match the scenery.

**Harbor Market** adds a waterfront taxi district with a connected street grid,
four-lane market avenue, ferry turnaround, two parking courts and 32 traffic
vehicles. Red-brick market halls, green awnings, cafes and dockside workshops
give each neighborhood its own dynamic prompt fragments.

**Mesa Gardens** adds a desert resort taxi district with a curved outer loop,
cross-town avenues, garden streets, two terminal turnarounds, a resort parking
court and 28 traffic vehicles. Its fragments describe cream stucco shops,
terracotta arcades, botanical gardens and desert landscaping. Both maps are
available from the map menu for Taxi and Free Roam modes.

## Map-frame authoring

`crazy-robotaxi-compose-frames` combines the saved generated artwork with the
map's compiled camera and semantic geometry. The source artwork and original
generation notes live in `crazy_robotaxi/artwork/`; the compositor does
not invoke image generation.

```bash
uv run --package crazy-robotaxi --extra authoring crazy-robotaxi-compose-frames \
  boulevard_district --output-dir /tmp/robotaxi-frames
```

Supported maps are `boulevard_district`, `flashdreams_raceway`,
`pacific_breakwater`, `harbor_market`, and `mesa_gardens`. The command writes
`*_initial_frame.png`, `*_alignment.png`, and `*_spawn_geometry.png` into the
explicit output directory. Review a candidate before copying its initial frame
into the game's map directory.

The compositor projects road surfaces, curbs, and runtime-loaded lane markings,
reuses the generated asphalt texture, warps scenery to the road edges, and
blends the car hood. It uses the engine's ground intersections, including
authored elevation. Registration landmarks are specific to the saved artwork.

Its assertions check static paint coverage only. They do not establish that
Omnidreams respects vehicle motion; inspect the candidate and validate motion
before promoting it to a runtime frame.
