#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Auto-restart loop for the Crazy Robotaxi demo, run inside the
# `robotaxi-demo` tmux session by start_demo.sh. If the app exits it is
# relaunched after a 10 s delay, capped at 5 restarts per rolling hour.
#
# usage: run_loop.sh HOST PORT TOKEN LOGFILE
set -u

HOST="$1"
PORT="$2"
TOKEN="$3"
LOG="$4"

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(dirname "$KIT")"
F=/localhome/local-wenqingw/projs/flashdreams/integrations/omnidreams

RESTART_DELAY_S=10
MAX_RESTARTS_PER_HOUR=5
restart_epochs=()

cd "$W"
while true; do
    echo "=== [$(date '+%F %T')] launching crazy-robotaxi on $HOST:$PORT ===" >> "$LOG"
    uv run crazy-robotaxi \
        --stream-mjpeg "$HOST:$PORT" \
        --stream-token "$TOKEN" \
        --map "$W/apps/crazy_robotaxi/crazy_robotaxi/maps/boulevard_district.robotaxi.yaml" \
        --variant default \
        --live-edit-corrector-mode fused \
        --live-edit-style \
        --live-edit-style-lora "$F/edit_sft/outputs/lora_style_v6_step1600.pt" \
        --live-edit-style-corrector "$F/edit_sft/outputs/lora_style_corrector_v5_valpeak.pt" \
        --live-edit-gate-alpha-json "$F/edit_sft/outputs/gate_style_v5.json" \
        --live-edit-base-corrector "$F/drift_correction/outputs/lora_v2_v3_valpeak.pt" \
        --live-edit-skin-guidance-chunks 6 \
        --live-edit-coins \
        ${COIN_SPRITE:+--live-edit-coin-sprite "$COIN_SPRITE"} \
        --live-edit-weather \
        --live-edit-obstacle \
        >> "$LOG" 2>&1
    code=$?
    echo "=== [$(date '+%F %T')] app exited with code $code ===" >> "$LOG"

    # Prune restarts older than an hour, then enforce the hourly cap.
    now=$(date +%s)
    pruned=()
    for t in "${restart_epochs[@]:-}"; do
        [[ -n "$t" && $((now - t)) -lt 3600 ]] && pruned+=("$t")
    done
    restart_epochs=("${pruned[@]:-}")
    live=0
    for t in "${restart_epochs[@]:-}"; do
        [[ -n "$t" ]] && live=$((live + 1))
    done
    if (( live >= MAX_RESTARTS_PER_HOUR )); then
        echo "=== [$(date '+%F %T')] $live restarts in the last hour;" \
             "giving up (cap $MAX_RESTARTS_PER_HOUR/h). ===" >> "$LOG"
        exit 1
    fi
    restart_epochs+=("$now")

    echo "=== [$(date '+%F %T')] restarting in ${RESTART_DELAY_S}s" \
         "(restart $((live + 1))/$MAX_RESTARTS_PER_HOUR this hour) ===" >> "$LOG"
    sleep "$RESTART_DELAY_S"
done
