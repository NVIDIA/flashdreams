#!/usr/bin/env bash
# Quick driver for the Ludus condition rasterizer benchmark.
# Mirrors the chunk shape used by WorldModelRenderBackend so the
# numbers can be compared directly to the demo's [world-model] raster_ms.
set -euo pipefail

xvfb-run -a uv run --no-sync --package flashdreams-omnidreams \
  python integrations/omnidreams/scripts/bench_ludus_raster.py \
  --frames-per-chunk 8 \
  --num-chunks 30 \
  --warmup-chunks 5 \
  --width 1280 --height 704 \
  "$@"
