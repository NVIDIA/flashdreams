#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-50052}"
VIEWER_PORT="${VIEWER_PORT:-8081}"

uv run --package flashdreams-realesrgan uplift-server \
  --port "${PORT}" \
  --viewer_port "${VIEWER_PORT}" \
  --viewer_jpeg_backend auto \
  --viewer_jpeg_quality 75 \
  --viewer_frame_stride 1 \
  --viewer_playback_fps 30 \
  --compile \
  --compile_mode reduce-overhead \
  --warmup_height 496 \
  --warmup_width 896 \
  --warmup_frames 3 \
  "$@"
