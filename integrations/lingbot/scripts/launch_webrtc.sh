#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8092}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG_NAME="${CONFIG_NAME:-lingbot-world-v2-14b-causal-fast-taehv-window15-sink3}"
WARMUP_CHUNKS="${WARMUP_CHUNKS:-0}"
FPS="${FPS:-16}"
EXAMPLE_IDX="${EXAMPLE_IDX:-0}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-352}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

exec uv run --no-sync python -m lingbot.webrtc.server \
  --host "${HOST}" \
  --port "${PORT}" \
  --config_name "${CONFIG_NAME}" \
  --device "${DEVICE}" \
  --warmup_chunks "${WARMUP_CHUNKS}" \
  --fps "${FPS}" \
  --video-height "${VIDEO_HEIGHT}" \
  --video-width "${VIDEO_WIDTH}" \
  --example-idx "${EXAMPLE_IDX}"
