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

```bash
uv sync --package crazy-robotaxi
uv run flashdreams-run-v2 crazy-robotaxi \
  --mode native-window \
  --window-title "Crazy Robotaxi"
```

Native-window mode keeps the composited frame on the GPU and presents it in a
local GLFW window. It requires a local display plus SlangPy's Vulkan/CUDA
interop support. For a browser client instead, run:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc
```

Application arguments follow `--`:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc -- \
  --map /path/to/city.robotaxi.yaml \
  --model-preset perf \
  --game-time-s 90
```

Use `flashdreams-run-v2 crazy-robotaxi -- --help` for the complete application
options. Drive with W/A/S/D or the arrow keys; Space is the handbrake. The V2
client's reset event rebuilds simulation, game rules, traffic, conditioning,
and the autoregressive cache together while retaining the loaded model.
Leaderboard name entry is owned by the Dear ImGui UI and submitted to the model
loop through V2's asynchronous loop-message contract.

## Layered configuration

Crazy Robotaxi accepts strict, partial `schema_version: 1` configuration files:

- `--engine-config` covers the map request, model preset, renderer, and V2
  runtime diagnostics.
- `--game-config` covers taxi/race mode, rules, input reduction, vehicle
  physics, persistence, and live-edit abilities.

Settings resolve from packaged typed defaults, then YAML, then explicitly
supplied application flags. Omitted YAML fields retain their lower-precedence
values, unknown keys are errors, and relative paths inside YAML resolve beside
that file. The legacy `--renderer-config` option remains available as an
explicit compatibility layer.

The checked-in `example_engine_config.yaml` and `example_game_config.yaml`
show the complete V2 surface:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode native-window -- \
  --engine-config \
    apps/crazy_robotaxi/crazy_robotaxi/configs/example_engine_config.yaml \
  --game-config \
    apps/crazy_robotaxi/crazy_robotaxi/configs/example_game_config.yaml \
  --model-preset perf
```

Here `--model-preset perf` wins over the value in the engine YAML. Maps remain
separate authoritative `.robotaxi.yaml` documents referenced by path; layered
configuration does not copy or reinterpret map content.

## Race mode

Authored maps may define ordered race courses using the same schema and map
files as the original Crazy Robotaxi branch. Start the bundled demonstration
course with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode native-window -- \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/demo_race_track.robotaxi.yaml \
  --game-mode race
```

Race progression uses swept, direction-aware gate crossings, supports looped
and point-to-point courses, and persists times per map/course. Select a
non-default course with `--race-course ID` and override its leaderboard with
`--race-times PATH`.

## Live-edit gameplay

The source game's flag-gated abilities are available through V2 application
arguments. `C` toggles collectible coins, `K` cycles style skins, `V` cycles
weather, and `O` spawns a crossing-obstacle event. Effect items can trigger
rain, snow, a timed mystery skin, or a physics-authoritative nitro boost.

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode native-window -- \
  --live-edit-coins \
  --live-edit-items \
  --live-edit-weather \
  --live-edit-obstacle
```

Style mode additionally requires `--live-edit-style-lora PATH`. Optional
style/base/weather correctors retain the source branch's fused and unfused
deployment modes. Text-edit and obstacle guidance require a non-native DiT
preset because those paths intercept the Python transformer conditioning
seam; the application reports this before generation if an incompatible
native preset is selected. All abilities are disabled by default.

## Original demo performance preset

The opt-in `original-perf` preset reproduces the performance knobs from the
current `example_world_model_perf.yaml` without importing or depending on the
`interactive_drive` application. It uses the compiled LightVAE/LightTAE path,
the required native FP8 DiT with cuDNN attention, skips the KV-cache finalize
pass, and uses denoising timesteps `[1000, 100]`. For direct comparison with
that manifest, select its 1168x640 resolution as shown below.

Prepare the native DiT sources once from the repository root:

```bash
uv run --package flashdreams-omnidreams omnidreams-prepare --perf
```

Then run Crazy Robotaxi with the runtime resolution arguments before the
application-argument separator:

```bash
uv run flashdreams-run-v2 crazy-robotaxi \
  --mode native-window \
  --pixel-width 1168 \
  --pixel-height 640 \
  -- \
  --model-preset original-perf
```

The resolution flags are optional: without them, the preset uses the V2
session's default resolution and adapts the game renderer to match. The preset
still fails if the required native DiT cannot be built or loaded. It keeps the
PyTorch LightVAE encoder, so it does not require a `lightvae-fp8-state.pt`
file. Use the existing `perf` and `native-perf` presets for their previous
behavior. GPU throughput and quality still need to be validated on the target
machine before treating this preset as a regression baseline.

## Candidate maximum-performance preset

The opt-in `fast-perf` preset combines `original-perf`'s required native FP8
DiT and cuDNN attention with `native-perf`'s native FP8 LightVAE image and
per-chunk encoders. It retains the compiled/CUDA-graphed LightTAE decoder and
the original demo's two-step schedule. This combination is a performance
candidate, not yet a validated default; compare its steady-state timings and
output quality with `original-perf` on the target GPU.

It requires a calibrated LightVAE FP8 state. After generating that state, run:

```bash
OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH=artifacts/native_vae/lightvae-fp8-state.pt \
  uv run flashdreams-run-v2 crazy-robotaxi \
  --mode native-window \
  --pixel-width 1168 \
  --pixel-height 640 \
  -- \
  --model-preset fast-perf
```

The native DiT sources must also have been prepared as described above. As
with `original-perf`, the resolution flags are optional and the app renderer
follows the V2 session dimensions.

## Performance diagnostics

Crazy Robotaxi emits lightweight model-thread, engine, overlay, and PhysX
timings during normal play. Capture synchronized model-step and GPU-stage
diagnostics while reproducing a chunk pause with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc \
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
like-for-like `original-perf` rerun needed after presentation changes.

Presentation remains fixed at 30 fps. Disabling diagnostic synchronization
removes an avoidable pause source, but it does not make a model preset whose
steady-state throughput is below 30 fps meet that rate; use `--model-preset
perf` when its quality/performance tradeoff is appropriate.

Profile interactive input separately with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc -- \
  --model-preset perf \
  --profile-input-latency
```

This opt-in adds a UI-thread key indicator and logs the time from V2 UI event
receipt to the first presented model frame carrying that transition. It also
shows resolved, redundant, and capacity-dropped transition counts. The normal
HUD does not construct these widgets when the flag is absent. The indicator's
physical-key-to-browser delay still includes WebRTC transport latency, while
the reported `UI TO MODEL FRAME` value isolates the synchronous app/model
portion. See the
[V2 latency handoff](../../docs/design/crazy_robotaxi_v2_input_latency.md) for
the known runtime boundary.

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
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc -- \
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
