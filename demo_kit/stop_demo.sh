#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stop the Crazy Robotaxi hosted demo: kill the tmux session and any
# leftover app process, then report port and GPU state.
set -u

SESSION="robotaxi-demo"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Killed tmux session '$SESSION'."
else
    echo "No tmux session '$SESSION' found."
fi

# The run loop is gone, but the app itself may still be shutting down.
if pgrep -u "$USER" -f "crazy-robotaxi" > /dev/null; then
    pkill -u "$USER" -f "crazy-robotaxi"
    echo -n "Waiting for crazy-robotaxi to exit"
    for _ in $(seq 1 30); do
        pgrep -u "$USER" -f "crazy-robotaxi" > /dev/null || break
        echo -n "."
        sleep 1
    done
    echo
fi
if pgrep -u "$USER" -f "crazy-robotaxi" > /dev/null; then
    echo "WARNING: crazy-robotaxi still running; sending SIGKILL."
    pkill -9 -u "$USER" -f "crazy-robotaxi"
    sleep 2
fi

if pgrep -u "$USER" -f "crazy-robotaxi" > /dev/null; then
    echo "ERROR: crazy-robotaxi process still alive:" >&2
    pgrep -a -u "$USER" -f "crazy-robotaxi" >&2
    exit 1
fi
echo "No crazy-robotaxi processes left."

if command -v nvidia-smi > /dev/null; then
    echo "GPU memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
fi
