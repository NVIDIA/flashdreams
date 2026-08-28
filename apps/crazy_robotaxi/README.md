# Crazy Robotaxi

Crazy Robotaxi is a FlashDreams V2 application built on the model/UI-loop API
and `omnidreams-game-engine`. FlashDreams owns input collection, reset
generations, presentation buffering, client windows, and the two-thread
runtime. The UI loop uses FlashDreams' Dear ImGui renderer for the game HUD and
composites it over model frames for native-window, WebRTC, and file clients.
Camera-projected waypoint rings, beacons, and labels are rendered as a
background ImGui draw list on the UI thread; they are not windows or controls. The
model loop publishes immutable snapshot-and-pose metadata for each generated
frame, and the UI loop caches the projected marker geometry while that frame
remains visible. The optional raw BEV view is a second model result displayed
inside a real ImGui `Map` window. `CrazyRobotaxiImGuiUILoop` returns the video
back buffer and the base loop composites one ImGui overlay containing the
background waypoints, HUD, and BEV window.

## Controls

### Keyboard

| Control | Action |
| --- | --- |
| `W` or Up Arrow | Drive forward. |
| `S` or Down Arrow | Reverse. |
| `A` or Left Arrow | Steer left. |
| `D` or Right Arrow | Steer right. |
| `Space` | Apply the handbrake and cancel throttle. |
| `R` | Restart the current game without reloading the model. |
| `Escape` | Go back: game → map, course → map, map → mode, then exit from the mode screen. |
| `C` | Toggle collectible coins when `--live-edit-coins` is enabled. |
| `K` | Cycle style skins when `--live-edit-style` is enabled. |
| `V` | Cycle weather when `--live-edit-weather` is enabled. Weather changes are ignored while a skin is active. |
| `O` | Spawn a crossing-obstacle event when `--live-edit-obstacle` is enabled. |
| `Enter` | Submit the leaderboard name while the Driver name field is focused. |

Menu choices and the leaderboard Submit button can also be clicked with the mouse.

### Controller

| Control | Action |
| --- | --- |
| Left stick, push forward+tilt | Steer. |
| Right trigger (`RT` / `R2` / `ZR`) | Throttle. |
| Left trigger (`LT` / `L2` / `ZL`) | Brake. |
| Start / Menu / Plus | Restart the current game without reloading the model. |
| Steering wheel and pedals | Use normalized steering, throttle, and brake input directly. |

A connected gamepad or wheel takes precedence over keyboard driving input.
Disconnecting it returns control to any keyboard keys still held. Gamepads do
not currently bind reverse, handbrake, menu navigation, or live-edit actions;
use the keyboard or mouse for those controls.

```bash
uv sync --package crazy-robotaxi
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams \
  --mode native-window \
  --window-title "Crazy Robotaxi"
```

`crazy-robotaxi-omnidreams` is the default app name.
`crazy-robotaxi-omnidreams-perf` is the performance-optimized app name.
`crazy-robotaxi-omnidreams-fast-perf` is the fast-performance-optimized app name.

Native-window mode keeps the composited frame on the GPU and presents it in a
local GLFW window. It requires a local display plus SlangPy's Vulkan/CUDA
interop support. For a browser client instead, run:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams --mode webrtc --port 8089
```

Open `http://127.0.0.1:8089/`. Add `--host 0.0.0.0` when the browser connects
through another host, matching Interactive Drive.

Application arguments follow `--`:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams-perf --mode webrtc -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/boulevard_district.robotaxi.yaml \
  --game-time-s 90
```

At startup, choose Taxi or Race mode, then choose a discovered map. Race mode
shows each map's authored race courses on the map-selection screen and starts
the selected course. The configured `--map` is listed first; the menu also
discovers bundled maps and `.robotaxi.yaml` maps beside that configured file.
Map compilation and scene loading begin only after the menu choice.

Use `flashdreams-run-v2 crazy-robotaxi-omnidreams -- --help` for the complete
application options. Restart rebuilds simulation, game rules, traffic,
conditioning, and the autoregressive cache while retaining the loaded model.
The HUD keeps an always-visible bearing arrow and overlays visible targets or
an off-map direction arrow on the BEV.

## Race mode

Authored maps may define ordered race courses using the same schema and map
files as the original Crazy Robotaxi branch. Put the bundled demonstration map
first in the selection screen with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams --mode native-window -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/demo_race_track.robotaxi.yaml \
  --game-mode race
```

Choose Race on the first screen, then choose its course on the second. Race
progression uses swept, direction-aware gate crossings, supports looped and
point-to-point courses, and persists times per map/course. Override the race
leaderboard path with `--race-times PATH`.

## Live-edit gameplay

The source game's flag-gated abilities are available through V2 application
arguments. `C` toggles collectible coins, `K` cycles style skins, `V` cycles
weather, and `O` spawns a crossing-obstacle event. Effect items can trigger
rain, snow, a timed mystery skin, or a physics-authoritative nitro boost.

To launch with coins, effect items, weather effects, and the style LoRA:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams --mode native-window -- \
  --live-edit-coins \
  --live-edit-items \
  --live-edit-weather \
  --live-edit-style
```

On first launch, style mode downloads the v6 style LoRA, its recommended v5
corrector and gate, and the base-world clean-forcing corrector into
`artifacts/crazy_robotaxi/live_edit`. Later launches reuse those files. The
`--live-edit-style-lora`, `--live-edit-style-corrector`,
`--live-edit-gate-alpha-json`, and `--live-edit-base-corrector` options override
individual assets. Weather remains LoRA-free unless a nonzero weather-corrector
gain is configured. Text-edit and obstacle guidance require a non-native DiT
preset because those paths intercept the Python transformer conditioning
seam; the application reports this before generation if an incompatible
native preset is selected. All abilities are disabled by default.

## Maximum-performance preset

OmniDreams exposes `standard`, `perf`, and `fast-perf` pipeline configs. Crazy
Robotaxi selects those model-owned configs directly. `fast-perf` retains perf's
required native FP8 DiT, cuDNN attention, CUDA-graphed LightTAE decoder, and
two-step schedule, while preferring the native FP8 LightVAE encoders.

Prepare the native DiT sources once from the repository root:

```bash
uv run --package flashdreams-omnidreams omnidreams-prepare --perf
```

Run the fast config directly:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams-fast-perf \
  --mode native-window
```

On its first launch, `fast-perf` downloads the public OmniDreams sample,
calibrates the native FP8 LightVAE, and atomically caches the result at
`artifacts/native_vae/lightvae_fp8_state.pt`. Later launches reuse that file.
Set `OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` to override the cache location.

The native DiT sources must also have been prepared as described above. GPU
throughput and quality should be validated on the
target machine before treating it as a regression baseline.

## Performance diagnostics

Crazy Robotaxi emits lightweight model-thread, engine, overlay, and PhysX
timings during normal play. Capture synchronized model-step and GPU-stage
diagnostics while reproducing a chunk pause with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams --mode webrtc \
  --stats-path /tmp/crazy-robotaxi-stats.json -- \
  --profile-pipeline
```

Pipeline profiling is diagnostic and disabled during normal play because its
CUDA synchronization creates a CPU spin hotspot and prevents pipeline overlap.
With `--profile-pipeline`, the runtime also warns when a model step exceeds the
duration of the frames it produces. The live `model_step_wall_ms`,
`model_step_cpu_ms`, `engine_cpu_ms`,
`simulation_cpu_ms`, `rules_cpu_ms`, `conditioning_cpu_ms`, and `physx_*_ms`
metrics separate a throughput miss from model-thread CPU work; the pipeline's
`encode_ms`, `diffuse_ms`, `decode_ms`, and `finalize_ms` metrics identify the
corresponding GPU stage. Waypoint projection and ImGui draw submission are
UI-thread work and are not included in model-step metrics. The JSON sink
normalizes `_ms` metric names to `_s`. Exclude the first chunks when judging
steady state because compilation and graph capture are startup costs.

The PhysX split includes `physx_traffic_prepare_ms`,
`physx_barrier_rebound_ms`, `physx_traffic_update_ms`,
`physx_state_materialize_ms`, and `physx_bridge_other_ms`. Together they locate
adapter work within `physx_bridge_ms` without enabling synchronized GPU-stage
profiling.

The BEV is HUD-only data, so Crazy Robotaxi caps its raster resolution to the
actual ImGui map-image extent while preserving the authored aspect ratio. At
the default 1280x704 output this changes the default square BEV from 1024x1024
to 234x234. The renderer's uint8 pixels remain uint8 through the game-engine
contract; only the much smaller displayed image crosses to the CPU for ImGui's
current pixel-upload helper. A fully GPU-resident BEV texture still requires a
CUDA-tensor image hook in the V2 ImGui renderer.

The saved [performance investigation](../../docs/design/crazy_robotaxi_v2_performance.md)
records the current baseline, what the existing captures prove, and the exact
like-for-like `fast-perf` rerun needed after presentation changes.

Presentation remains fixed at 30 fps. Disabling diagnostic synchronization
removes an avoidable pause source, but it does not make a model preset whose
steady-state throughput is below 30 fps meet that rate; use the
`crazy-robotaxi-omnidreams-perf` app when that tradeoff is appropriate.

Profile interactive input separately with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams-perf --mode webrtc -- \
  --profile-input-latency
```

This opt-in adds a UI-thread key indicator and logs the time from V2 UI event
receipt to the first presented model frame carrying that transition. The normal
HUD does not construct these widgets when the flag is absent. The indicator's
physical-key-to-browser delay still includes WebRTC transport latency, while
the reported `UI TO MODEL FRAME` value isolates the synchronous app/model
portion. See the
[V2 latency handoff](../../docs/design/crazy_robotaxi_v2_input_latency.md) for
the remaining synchronous model-step boundary.

All model presets generate four neutral, hidden blocks before publishing the
first gameplay frame. The responsive ImGui HUD shows the current warmup block,
an animated activity marker, and elapsed time while compilation and autotuning
run. After warmup, the app recreates simulation, rules, conditioning, and the
autoregressive cache, so warmup does not consume game time, move the taxi,
advance the visible AR index, or count toward `--total-blocks`. Cache-bound
CUDA graphs safely re-arm against the new gameplay cache, so shorter first-use
hitches can remain. This moves the multi-second pauses ahead of presentation;
it does not reduce total cold-start time. Disable it for comparisons or startup
debugging with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi-omnidreams --mode webrtc -- \
  --prewarm-blocks 0
```

## Authored maps

Maps are strict semantic `.robotaxi.yaml` documents. Validate or compile them
without loading a model:

```bash
uv run crazy-robotaxi-map validate path/to/city.robotaxi.yaml
uv run crazy-robotaxi-map compile path/to/city.robotaxi.yaml
uv run crazy-robotaxi-map preview path/to/city.robotaxi.yaml --output city.svg
uv run crazy-robotaxi-map preview-spawn path/to/city.robotaxi.yaml \
  --spawn taxi_start --output taxi_start.png
```

The architectural source of truth for this rewrite is
[`../../docs/design/crazy_robotaxi_v2_architecture.md`](../../docs/design/crazy_robotaxi_v2_architecture.md).
