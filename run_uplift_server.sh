#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./run_uplift_server.sh [flashvsr|realesrgan] [uplift-server args...]

Environment:
  UPSAMPLER             flashvsr or realesrgan. Default: flashvsr
  PORT                  gRPC port. Default: 8090
  VIEWER_HOST           HTTP viewer bind host. Default: 0.0.0.0
  VIEWER_PORT           HTTP viewer port. Defaults: 8091
  VIEWER_JPEG_BACKEND   auto, cuda, or pillow. Default: auto
  VIEWER_JPEG_QUALITY   JPEG quality. Default: 75
  VIEWER_FRAME_STRIDE   Publish every Nth viewer frame. Default: 1
  VIEWER_PLAYBACK_FPS   Steady viewer playback FPS. Default: 30
  WARMUP_HEIGHT         Real-ESRGAN warmup input height. Default: 704
  WARMUP_WIDTH          Real-ESRGAN warmup input width. Default: 1280
  COMPILE               Set to 1 to pass --compile. Default: 0
  CUDA_GRAPH            FlashVSR only: set to 1 to pass --cuda_graph. Default: 0

Examples:
  ./run_uplift_server.sh flashvsr --attention_mode auto --sparse_ratio 1.5
  ./run_uplift_server.sh realesrgan --default_scale 2
  UPSAMPLER=realesrgan VIEWER_PORT=8081 ./run_uplift_server.sh --compile
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "flashvsr" || "${1:-}" == "realesrgan" || "${1:-}" == "real-esrgan" ]]; then
    UPSAMPLER="$1"
    shift
else
    UPSAMPLER="${UPSAMPLER:-flashvsr}"
fi

case "${UPSAMPLER}" in
    flashvsr)
        PACKAGE="flashdreams-flashvsr"
        MODULE="flashvsr.grpc.uplift_server"
        LABEL="FlashVSR"
        DEFAULT_VIEWER_PORT=8081
        ;;
    realesrgan|real-esrgan)
        PACKAGE="flashdreams-realesrgan"
        MODULE="realesrgan.grpc.uplift_server"
        LABEL="Real-ESRGAN"
        DEFAULT_VIEWER_PORT=8081
        ;;
    *)
        echo "unsupported UPSAMPLER=${UPSAMPLER}; expected flashvsr or realesrgan" >&2
        exit 2
        ;;
esac

PORT="${PORT:-8090}"
VIEWER_HOST="${VIEWER_HOST:-0.0.0.0}"
VIEWER_PORT="${VIEWER_PORT:-${DEFAULT_VIEWER_PORT}}"
VIEWER_JPEG_BACKEND="${VIEWER_JPEG_BACKEND:-auto}"
VIEWER_JPEG_QUALITY="${VIEWER_JPEG_QUALITY:-75}"
VIEWER_FRAME_STRIDE="${VIEWER_FRAME_STRIDE:-1}"
VIEWER_PLAYBACK_FPS="${VIEWER_PLAYBACK_FPS:-30}"
WARMUP_HEIGHT="${WARMUP_HEIGHT:-704}"
WARMUP_WIDTH="${WARMUP_WIDTH:-1280}"
COMPILE="${COMPILE:-0}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/torchinductor-${UPSAMPLER}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.cache/triton-${UPSAMPLER}}"

server_args=(
    --port "${PORT}"
    --viewer_host "${VIEWER_HOST}"
    --viewer_port "${VIEWER_PORT}"
    --viewer_jpeg_backend "${VIEWER_JPEG_BACKEND}"
    --viewer_jpeg_quality "${VIEWER_JPEG_QUALITY}"
    --viewer_frame_stride "${VIEWER_FRAME_STRIDE}"
    --viewer_playback_fps "${VIEWER_PLAYBACK_FPS}"
)

if [[ "${UPSAMPLER}" == "realesrgan" || "${UPSAMPLER}" == "real-esrgan" ]]; then
    server_args+=(
        --warmup_height "${WARMUP_HEIGHT}"
        --warmup_width "${WARMUP_WIDTH}"
    )
fi

if [[ "${COMPILE}" == "1" || "${COMPILE}" == "true" ]]; then
    server_args+=(--compile)
fi

if [[ "${UPSAMPLER}" == "flashvsr" && ( "${CUDA_GRAPH}" == "1" || "${CUDA_GRAPH}" == "true" ) ]]; then
    server_args+=(--cuda_graph)
fi

echo "Starting ${LABEL} uplift server on :${PORT}"
if [[ "${VIEWER_PORT}" != "0" ]]; then
    echo "Starting ${LABEL} HTTP viewer on ${VIEWER_HOST}:${VIEWER_PORT}"
    echo "Using ${LABEL} viewer JPEG backend: ${VIEWER_JPEG_BACKEND}"
fi

# Use the module path rather than the shared ``uplift-server`` console script:
# duplicate script names across workspace packages resolve nondeterministically
# in one uv-managed venv.
exec uv run --package "${PACKAGE}" python -m "${MODULE}" "${server_args[@]}" "$@"
