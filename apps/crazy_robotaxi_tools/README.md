# Crazy Robotaxi tools

Offline commands live in this package. Install it separately from the game.

## Map commands

```bash
uv run --package crazy-robotaxi-tools crazy-robotaxi-map --help
uv run --package crazy-robotaxi-tools crazy-robotaxi-map preview \
  apps/crazy_robotaxi/crazy_robotaxi/maps/boulevard_district.robotaxi.yaml \
  --output /tmp/boulevard.svg
```

The CLI supports `validate`, `compile`, `preview`, and `preview-spawn`.
Preview exporters are supplied by `omnidreams-game-engine-tools`; map loading
and compilation use the same engine code as the game.

## Headless captures

From the repository root, run a three-second check of Raceway's starting
straight:

```bash
uv run --package flashdreams-omnidreams --with-editable apps/crazy_robotaxi_tools \
  --with-editable apps/omnidreams_game_engine_tools crazy-robotaxi-scenario raceway_start \
  --output-dir /tmp/robotaxi-raceway-start
```

Each run needs a fresh output directory and `ffmpeg`/`ffprobe` on `PATH`.
The runner uses the real model, checks all videos' frame counts and telemetry
alignment, then summarizes the physical observations. Other scenarios are
`flat_acceleration` and `curb_impact`, which use the dedicated flat/curb test
map under `crazy_robotaxi_tools/maps/`. `hill_climb` drives from a flat approach onto a 32 m
hill using `debug_hill.robotaxi.yaml`. Physics measurements are reported for
inspection; the capture check does not assert that the vehicle handles correctly.
`--slug` selects another model
preset, `--config` selects application settings, and `--snapshot-every`
controls PNG sampling. Assertions fail with a nonzero exit status and
leave the capture available for inspection.

