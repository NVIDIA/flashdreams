# Crazy Robotaxi

Crazy Robotaxi is a FlashDreams V2 application built on the model/UI-loop API
and `omnidreams-game-engine`. FlashDreams owns input collection, reset
generations, presentation buffering, client windows, and the two-thread
runtime. The UI loop uses FlashDreams' SlangPy renderer for the game HUD and
composites it over model frames for local and WebRTC clients.
Camera-projected waypoint rings, beacons, and labels are carried as a
frame-aligned world layer; they are not ImGui windows. The optional BEV view
is likewise published as a frame-aligned RGBA model-result layer. The
`CrazyRobotaxiSlangPyUILoop` composites video, waypoints, and BEV through V2's
presentation manager before the base loop applies the SlangPy ImGui HUD.

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

Crazy Robotaxi emits model, engine, overlay, and complete model-step timing
metrics. Capture them while reproducing a chunk pause with:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc \
  --stats-path /tmp/crazy-robotaxi-stats.json
```

The runtime also warns when a model step exceeds the duration of the frames it
produces. The live `model_step_wall_ms`, `model_step_cpu_ms`, `engine_cpu_ms`,
`simulation_cpu_ms`, `rules_cpu_ms`, `conditioning_cpu_ms`,
`waypoint_overlay_cpu_ms`, and `physx_*_ms` metrics separate a throughput miss
from app-side CPU work; the pipeline's `encode_ms`, `diffuse_ms`, `decode_ms`,
and `finalize_ms` metrics identify the corresponding GPU stage. The JSON sink
normalizes `_ms` metric names to `_s`. Exclude the first chunks when judging
steady state because compilation and graph capture are startup costs.

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
