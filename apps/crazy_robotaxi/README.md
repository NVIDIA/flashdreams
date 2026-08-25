# Crazy Robotaxi

Crazy Robotaxi is a FlashDreams V2 application built on the model-thread
`omnidreams-game-engine`. FlashDreams owns input collection, reset generations,
presentation buffering, client windows, and the two-thread runtime.
The UI thread uses FlashDreams' SlangPy ImGui renderer for the game HUD and
composites it over model frames for local and WebRTC clients.

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
Leaderboard name entry is owned by the ImGui UI and submitted to the model
thread through V2's asynchronous thread-message contract.

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
