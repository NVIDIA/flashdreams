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
