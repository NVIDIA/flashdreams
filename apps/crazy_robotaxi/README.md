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

## Headless driving and diagnosis

`--mode robotaxi-debug` runs the normal game and input path without a desktop
window. By default it captures the ordinary UI rendered offscreen. Add
`--no-ui` after `--` for paired generated, HD-map, and physics views. Select an
explicit `--map` and `--game-mode`; race mode also needs `--race-course`.

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams-perf --mode robotaxi-debug \
  --debug-output-dir /tmp/robotaxi-capture --debug-cpu-sample-ms 20 -- \
  --map apps/crazy_robotaxi/tests/maps/debug_flat.robotaxi.yaml \
  --game-mode free-roam
```

Each capture needs a fresh output directory. Both presentation modes record
`telemetry.jsonl` for every simulation frame, including applied commands, native
physics states before gameplay adjustments, camera poses, ground errors, collider
snapshots, and game state. Observations are queued before model generation, so a
generation failure retains the physics evidence. Hidden warmup is excluded.

By default, both modes save `generated.mp4`, `hdmap.mp4`, `physics.mp4`, and
sampled/contact/final screenshots. Normal UI mode additionally saves
`presentation.mp4` and an atomically updated `latest.png`; `--no-ui` selects the
paired diagnostic presentation. Pass `--no-debug-capture-images` before `--` to
keep telemetry without videos or PNGs; this also removes the `ffmpeg` requirement.

`presentation.jsonl` records UI ticks, actual presentation timestamps, and source
frame identifiers. Join its `source.capture_frame` to `telemetry.jsonl` to find
repeated or skipped presentations. `capture_frame` stays unique across restarts;
`rollout_epoch`, `frame`, `model_step`, `autoregressive_index`, and `chunk_frame`
retain simulation/chunk identity. `media.jsonl` maps encoded video indices to
capture or presentation frame IDs, including when images are missing. Videos
use nominal frame rates; use presentation timestamps to reconstruct wall time.

A background writer performs disk writes, encoding, and physics drawing. Its
queue holds at most 256 observation/media jobs, four model-image chunks, and 60
UI images; images are copied into owned RGB byte storage before enqueueing.
Producers never wait for disk. Capture still costs observation collection and
image conversion/copy work. Overflow preserves gameplay timing and records
missing inclusive frame ranges in `manifest.json`. `--debug-strict-capture`
fails the run on capture gaps or writer errors. Normal shutdown drains accepted
work before closing the files.

The manifest separates file finalization, capture coverage, and scripted-run
completion. It includes produced/saved counts, missing ranges, generated frame
count, writer errors, pending input IDs, and termination reason. `complete`
requires complete capture and consumed inputs; scripted runs must also reach
their close event or a configured block/game endpoint. `inputs/` preserves the
map, settings, and input file. Rejected live commands remain visible in the input
log and do not themselves invalidate an otherwise complete capture.

### Timestamped inputs

Both live agents and YAML files use the same event schema and queue. Times are
integer microseconds on a monotonic wall clock that starts **after gameplay
initialization and model warmup**. It continues through restarts. Controls retain
their state until a release event, just as a person holding a key would.

Send JSON datagrams to `<output-dir>/control.sock`. Bind a local Unix datagram
reply socket before sending to receive acknowledgments. A `status` request
returns `loading` with `elapsed_us: null`, or `ready` with the current
`elapsed_us`. Schedule commands ahead of that time, allowing for transport and
UI polling latency:

```python
import json
import socket

with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as agent:
    agent.bind("/tmp/robotaxi-agent.sock")  # Fresh path; remove after closing.
    agent.settimeout(2)
    game = "/tmp/robotaxi-capture/control.sock"
    agent.sendto(b'{"id":"clock","type":"status"}', game)
    clock = json.loads(agent.recv(4096))
    assert clock["status"] == "ready"  # Query again while loading.
    event = {"id": "forward", "timestamp_us": clock["elapsed_us"] + 500_000,
             "type": "keyboard", "key": "w", "pressed": True}
    agent.sendto(json.dumps(event).encode(), game)
    print(json.loads(agent.recv(4096)))  # accepted or rejected
```

Every command requires a unique string `id`, an integer `timestamp_us`, and a
`type`. Supported payloads are `keyboard` with `key` and Boolean `pressed`,
`focus` with Boolean `focused`, and `close`. `close` ends the capture through the
normal runtime lifecycle. The input socket is local; it is not a network service.

Acknowledgments are also flushed to `input_events.jsonl`, so a disconnected or
slow agent can recover them:

- `accepted`: the application received a valid command before its deadline.
- `rejected`: invalid, duplicate, or late input. `reason: late` means its timestamp
  was earlier than `received_us`; the command is never applied. Receipt is when
  the application reads the socket, not when the sender writes it.
- `dispatched`: the command's time has arrived and the UI submitted it to the
  ordinary runtime event buffer. `dispatched_us` records polling delay.
- `consumed`: the model's input consumer read it, with `consumed_us` and
  `model_step`. Close events end at dispatch because the UI handles shutdown.
- `cancelled`: shutdown occurred before a pending command was consumed.

A command accepted on time remains valid if the game later stalls. Inputs are
never dispatched early; they enter at the first UI poll at or after the timestamp.
The model still consumes inputs at chunk boundaries, so consumption can be late
and a short press may be sampled later. This deliberately preserves native input
behavior, allowing headless captures to expose responsiveness regressions.
Acknowledgments measure that latency; they do not claim sub-chunk physics updates.
The queue accepts at most 100,000 unique command IDs per capture.

For unattended runs, add `--drive-script path/to/inputs.yaml` after `--`. The file
preloads the same commands before the ready clock starts:

```yaml
events:
  - {id: accelerate, timestamp_us: 0, type: keyboard, key: w, pressed: true}
  - {id: release, timestamp_us: 3000000, type: keyboard, key: w, pressed: false}
  - {id: end, timestamp_us: 4000000, type: close}
```

Include a close event or use the runtime timeout to bound the run. Scripted input
uses the same prewarm, presentation, keyboard handling, and sampling as live
input. Wall-clock duration does not guarantee a fixed number of generated frames.
The scenario runner uses the dedicated debug map for `flat_acceleration` and
`curb_impact`, captures paired views, and checks videos with `ffprobe`:

```bash
uv run --package flashdreams-omnidreams python \
  apps/crazy_robotaxi/tests/run_headless_scenario.py flat_acceleration \
  --output-dir /tmp/robotaxi-flat
```

### Model-thread instrumentation

`model_steps.jsonl` records the model thread's native OS thread ID and per-chunk
wall/CPU times for simulation, rules, conditioning, generation, cache finalization,
postprocessing, and the overall model step. Compare wall time with thread CPU time
to distinguish CPU work from waiting. These wall timers do not synchronize GPU
kernels and are not CUDA kernel durations; existing pipeline profiling remains
available for GPU measurements.

`--debug-cpu-sample-ms 20` additionally records Python stacks in `cpu_samples.jsonl`.
The UI samples the model thread, including loading, with no extra runtime thread.
Use the native thread ID to correlate with OS CPU monitoring. Stack samples show
the Python call site when native code is running or waiting, not native C++ stacks.
UI stalls or native code holding the GIL delay sampling; inspect actual
`sampled_ns` intervals rather than assuming the requested cadence. Sampling is
disabled by default.

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
discarded. Its generation prompt and compositing notes are saved beside the
image as `pacific_breakwater_initial_frame.prompt.txt`.
The full spawn prompt establishes the coastal setting for normal play; the
shorter base and nine road fragments keep lighting and hood framing consistent
while adding local landmarks with dynamic prompts enabled. The runtime supplies
turn direction and vehicle motion.

After changing this map or its spawn, rebuild the frame from the repository
root with `python apps/crazy_robotaxi/tools/compose_map_frames.py pacific_breakwater`
in the app environment. This uses the checked-in artwork and existing OpenCV,
Pillow, NumPy and Shapely dependencies. The map-specific compositor checks
visible paint pixels against the projected marking mask and saves a semantic
geometry preview plus a cyan alignment overlay beside the seed image.
Dash segments are 3 metres with 6-metre gaps along the semantic divider paths.

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

YAML maps with `schema_version: 2` support optional node `pose.z_m` heights.
Schema-v1 flat maps remain supported. Roads
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
visual style, weather and regional scenery without fixing the current road
placement, lane count, upcoming turn or landmark position. Spawn-specific
geometry belongs in the initial image and its offline generation prompt;
location-specific descriptions belong in the dynamic map fragments.

The bundled maps each have a dedicated hood-view initial frame. FlashDreams
Raceway shows grandstands, pit garages and a city skyline; Boulevard District
shows a leafy residential boulevard with yellow traffic separation. Both
frames use the compiled spawn's road projection, with map-relative assets
also used by the map and course menus. Their spawn prompts match the scenery.

To rebuild these frames from the saved artwork, run in the app environment:

```bash
python apps/crazy_robotaxi/tools/compose_map_frames.py flashdreams_raceway
python apps/crazy_robotaxi/tools/compose_map_frames.py boulevard_district
```

The compositor projects semantic road surfaces and white/yellow markings,
retains the generated hood and scenery, and checks visible paint coverage.
Each map's `*_alignment.png` shows the semantic lines in cyan;
`*_spawn_geometry.png` and `*_initial_frame.prompt.txt` retain the geometry
reference and generation provenance. Inspect the new frame after map edits.
