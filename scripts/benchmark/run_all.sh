#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run and plot every default benchmark:
#   ./scripts/benchmark/run_all.sh
# Run and plot every benchmark with supported exhaustive sweeps:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 ./scripts/benchmark/run_all.sh

set -euo pipefail
shopt -s globstar nullglob

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

status=0
for run_script in "$script_dir"/**/run.sh; do
    "$run_script" "$@" || status=$?
done

exit "$status"
