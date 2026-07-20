#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 INPUT_VIDEO [flashdreams-feed-frames args...]" >&2
  exit 2
fi

INPUT_VIDEO="$1"
shift

SERVER="${SERVER:-localhost:50052}"
SCALE="${SCALE:-2}"
MAX_CHUNKS="${MAX_CHUNKS:-20}"
TARGET_FPS="${TARGET_FPS:-30}"
CONTINUOUS_CHUNK_FRAMES="${CONTINUOUS_CHUNK_FRAMES:-8}"
INPUT_FORMAT="${INPUT_FORMAT:-jpeg}"
INPUT_JPEG_QUALITY="${INPUT_JPEG_QUALITY:-90}"
LOOP_MODE="${LOOP_MODE:-chunk}"

uv run --package flashdreams-realesrgan flashdreams-feed-frames \
  --server "${SERVER}" \
  --input "${INPUT_VIDEO}" \
  --scale "${SCALE}" \
  --max_chunks "${MAX_CHUNKS}" \
  --target_fps "${TARGET_FPS}" \
  --continuous_chunk_frames "${CONTINUOUS_CHUNK_FRAMES}" \
  --loop_mode "${LOOP_MODE}" \
  --input_format "${INPUT_FORMAT}" \
  --input_jpeg_quality "${INPUT_JPEG_QUALITY}" \
  "$@"
