#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the LingBot v2 Cam2V WebRTC demo from the venv setup.sh created.
#
# Uses the v2 CLI (`flashdreams-run-v2 <application-slug>`); the old
# `run_direct.py` / `python -m lingbot.runner` entry points no longer exist
# -- the v1 runner and its RUNNER_CONFIGS were removed in the v2 refactor.
#
#   bash run.sh                                              # native 832x464
#   LIGHT=1 bash run.sh                                      # cheaper 512x288
#   APP=cam2v-lingbot-world-fast-taehv-window15-sink3 bash run.sh   # low VRAM
#   PORT=8090 bash run.sh
#   WIDTH=640 HEIGHT=352 FPS=10 bash run.sh                  # explicit sizing
#   BLOCKS=2000 bash run.sh                                  # much longer run
#   PRESET=noir-alley-combat bash run.sh                     # start on a preset
#
# Settings are environment variables, so they go before the command:
# `LIGHT=1 bash run.sh`, not `bash run.sh LIGHT=1`. Arguments are passed
# through to the application instead.
#
# LIGHT=1 lowers the generated frame size and rate to 512x288 @ 12fps for
# GPUs that cannot keep up at native size (override with WIDTH/HEIGHT/FPS).
# There is no encoder choice to
# make here: the v2 WebRTC server hands raw frames straight to aiortc, which
# encodes in software (VP8 unless the browser negotiates H.264). The NVENC
# hardware encoder only exists in the v1 serving stack
# (flashdreams/serving/webrtc/nvenc.py), which v2 does not use -- so the only
# way to cut encode cost today is to give the encoder fewer pixels and fewer
# frames, which is what these knobs do.
#
# Note this changes what the *model generates*, not just what gets encoded:
# smaller frames are cheaper end to end but also lower quality. Keep width
# and height multiples of 16 (the model's default is 832x464 @ 16fps).
#
# Application slugs (see pyproject.toml entry points):
#   cam2v-lingbot
#   cam2v-lingbot-world-fast
#   cam2v-lingbot-world-fast-taehv-window15-sink3
#   cam2v-lingbot-world-v2-14b-causal-fast
#   cam2v-lingbot-world-v2-14b-causal-fast-taehv-window15-sink3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
APP="${APP:-cam2v-lingbot}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8089}"
WIDTH="${WIDTH:-}"
HEIGHT="${HEIGHT:-}"
FPS="${FPS:-}"
# The application stops once the rollout generates this many blocks, and
# nothing else is sized from it -- it is purely the stop condition. The 20
# it defaults to ends the run after about fifteen seconds of video, which
# reads as the server quitting on its own. This is set high enough that a
# run keeps going until you stop it: at roughly a second a block, 100000 is
# over a day. Lower it (BLOCKS=20) for a quick smoke test.
BLOCKS="${BLOCKS:-100000}"
# Start the rollout on one of the built-in presets. The page can swap prompts
# mid-run, but not the frame the rollout was initialized from, so picking a
# preset in the browser changes the events and the wording while the world
# still looks like whatever the session started on. Naming it here starts the
# session on that preset's own image and prompt instead.
PRESET="${PRESET:-}"

# LIGHT=1 opts into a smaller, cheaper stream; the default is whatever the
# model generates natively (832x464 for Lingbot). It only supplies
# defaults, so an explicit WIDTH/HEIGHT/FPS still wins.
if [ "${LIGHT:-0}" = "1" ]; then
  WIDTH="${WIDTH:-512}"
  HEIGHT="${HEIGHT:-288}"
  FPS="${FPS:-12}"
fi

if [ -d "$VENV" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  # Already inside a venv that has the packages installed (a `pip install -e`
  # into some other environment is a normal way to get here), so use it.
  echo "Using the active environment: $VIRTUAL_ENV"
elif command -v flashdreams-run-v2 >/dev/null 2>&1; then
  echo "Using flashdreams-run-v2 already on PATH."
else
  echo "ERROR: no environment found." >&2
  echo "  Run 'bash setup.sh' to build $VENV," >&2
  echo "  or activate the environment holding flashdreams-lingbot," >&2
  echo "  or point VENV at it: VENV=/path/to/venv bash run.sh" >&2
  exit 1
fi

# The GPU is shared; show what's already resident before claiming memory.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU state ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  echo
fi

# Runtime args go before the `--`; everything after it belongs to the app.
RUNTIME_ARGS=(--mode webrtc --host "$HOST" --port "$PORT")
if [ -n "$WIDTH" ]; then
  RUNTIME_ARGS+=(--pixel-width "$WIDTH")
fi
if [ -n "$HEIGHT" ]; then
  RUNTIME_ARGS+=(--pixel-height "$HEIGHT")
fi
if [ -n "$FPS" ]; then
  RUNTIME_ARGS+=(--fps "$FPS")
fi

echo "Starting $APP on $HOST:$PORT ..."
if [ -n "$WIDTH$HEIGHT$FPS" ]; then
  echo "Stream: ${WIDTH:-default}x${HEIGHT:-default} @ ${FPS:-default}fps"
fi
echo "Open http://localhost:$PORT once the model reports ready."
echo
# Anything passed to this script is forwarded to the application after
# these arguments, so `bash run.sh --no-ui` drops the camera-controls
# overlay the model draws into its own frames.
APP_ARGS=(--total-blocks "$BLOCKS")
if [ -n "$PRESET" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
  PRESET_JSON="$HERE/apps/cam2v/web/scene_presets.json"
  PRESET_IMAGE=$("$PY" -c "
import json, pathlib, sys
slug = sys.argv[1].strip().lower()
root = pathlib.Path(sys.argv[2]).parent
for preset in json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8')):
    if preset['name'].strip().lower().replace(' ', '-') == slug:
        image = preset.get('image', '')
        print((root / image) if not image.startswith('http') else '')
        break
" "$PRESET" "$PRESET_JSON")
  PRESET_PROMPT=$("$PY" -c "
import json, pathlib, sys
slug = sys.argv[1].strip().lower()
for preset in json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8')):
    if preset['name'].strip().lower().replace(' ', '-') == slug:
        print(preset.get('prompt', ''))
        break
" "$PRESET" "$PRESET_JSON")
  if [ -z "$PRESET_PROMPT" ]; then
    echo "ERROR: no preset named '$PRESET' in $PRESET_JSON" >&2
    exit 1
  fi
  # --example-data stays on: the resolver prefers an explicit prompt and
  # image over the example's own, while the example still supplies the
  # intrinsics and poses it requires and a preset does not carry.
  APP_ARGS+=(--example-data --prompt "$PRESET_PROMPT")
  if [ -n "$PRESET_IMAGE" ]; then
    APP_ARGS+=(--image-path "$PRESET_IMAGE")
  fi
  echo "Starting on preset '$PRESET'"
else
  APP_ARGS+=(--example-data)
fi

exec flashdreams-run-v2 "$APP" "${RUNTIME_ARGS[@]}" -- "${APP_ARGS[@]}" "$@"
