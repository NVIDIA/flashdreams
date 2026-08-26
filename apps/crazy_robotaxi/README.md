# Crazy Robotaxi

Crazy Robotaxi is a FlashDreams V2 application built on the model/UI-loop API
and `omnidreams-game-engine`. FlashDreams owns input collection, reset
generations, presentation buffering, client windows, and the two-thread
runtime. The UI loop uses FlashDreams' SlangPy renderer for the game HUD and
composites it over model frames for local and WebRTC clients.
Camera-projected waypoint rings, beacons, and labels are rendered as a
frame-aligned world layer on the UI thread; they are not ImGui windows. The
model loop publishes immutable snapshot-and-pose metadata for each generated
frame, and the UI loop rasterizes each presented waypoint layer once and
caches it while that frame remains visible. The optional BEV view is published
as an RGBA model-result layer. `CrazyRobotaxiSlangPyUILoop` composites video,
waypoints, and BEV through V2's presentation manager before the base loop
applies the SlangPy ImGui HUD.

```bash
uv sync --package crazy-robotaxi
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
Leaderboard name entry is owned by the SlangPy UI and submitted to the model
loop through V2's asynchronous loop-message contract.

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
corresponding GPU stage. Waypoint rasterization is UI-thread work and is not
included in model-step metrics. The JSON sink normalizes `_ms` metric names to
`_s`. Exclude the first chunks when judging steady state because compilation
and graph capture are startup costs.

Presentation remains fixed at 30 fps. Disabling diagnostic synchronization
removes an avoidable pause source, but it does not make a model preset whose
steady-state throughput is below 30 fps meet that rate; use `--model-preset
perf` when its quality/performance tradeoff is appropriate.

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
