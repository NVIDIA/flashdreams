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
  ./feed_uplift_video.sh /path/to/input.mp4 [flashdreams-feed-frames args...]

Environment:
  UPLIFT_SERVER        Server address. Default: 127.0.0.1:8090
  UPLIFT_TARGET_FPS    Feed rate. Default: 30

Examples:
  ./feed_uplift_video.sh /tmp/clip.mp4
  UPLIFT_SERVER=galaxy-ts4-108:8090 ./feed_uplift_video.sh /tmp/clip.mp4
  ./feed_uplift_video.sh /tmp/clip.mp4 --no_pace
  ./feed_uplift_video.sh /tmp/clip.mp4 --max_chunks 20
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

INPUT_VIDEO="$1"
shift

SERVER="${UPLIFT_SERVER:-${FLASHVSR_UPLIFT_SERVER:-127.0.0.1:8090}}"
TARGET_FPS="${UPLIFT_TARGET_FPS:-${FLASHVSR_TARGET_FPS:-30}}"

export PYTHONPATH="integrations/flashvsr:flashdreams${PYTHONPATH:+:${PYTHONPATH}}"

echo "Feeding ${INPUT_VIDEO} continuously to uplift server at ${SERVER}"
exec uv run --package flashdreams --extra serving flashdreams-feed-frames \
    --server "${SERVER}" \
    --input "${INPUT_VIDEO}" \
    --target_fps "${TARGET_FPS}" \
    "$@"
