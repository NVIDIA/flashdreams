#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Start the Crazy Robotaxi hosted demo in a detached tmux session with an
# auto-restart loop.
#
# usage: start_demo.sh [TOKEN]
#   TOKEN            optional shared stream token (arg beats $DEMO_TOKEN;
#                    a random one is generated when neither is given)
#   $DEMO_HOST       bind host (default 127.0.0.1; set to the internal
#                    hostname to serve players on the VPN)
#   $DEMO_PORT       bind port (default 8630)
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="robotaxi-demo"
DEMO_HOST="${DEMO_HOST:-127.0.0.1}"
DEMO_PORT="${DEMO_PORT:-8630}"
TOKEN="${1:-${DEMO_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
    TOKEN="$(openssl rand -hex 16 2>/dev/null \
        || python3 -c 'import secrets; print(secrets.token_hex(16))')"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists." >&2
    echo "Run stop_demo.sh first (or 'tmux attach -t $SESSION' to inspect it)." >&2
    exit 1
fi

mkdir -p "$KIT/logs"
LOG="$KIT/logs/robotaxi-demo.log"

tmux new-session -d -s "$SESSION" \
    "bash '$KIT/run_loop.sh' '$DEMO_HOST' '$DEMO_PORT' '$TOKEN' '$LOG'"

echo "Started tmux session '$SESSION' (log: $LOG)."
echo "The world model takes ~4 minutes to warm up before frames flow."
echo
echo "Player URL:"
echo "  http://$DEMO_HOST:$DEMO_PORT/?token=$TOKEN"
