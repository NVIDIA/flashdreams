#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run and plot the default Cosmos/Wan policy benchmark:
#   ./scripts/benchmark/flashdreams/accelerated/multi_head_attention/run.sh
# Replot its saved results:
#   uv run python -m scripts.benchmark.flashdreams.accelerated.multi_head_attention.plot
#
# Run and plot the exhaustive policy benchmark:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 \
#     ./scripts/benchmark/flashdreams/accelerated/multi_head_attention/run.sh
# Replot its saved results:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 \
#     uv run python -m \
#     scripts.benchmark.flashdreams.accelerated.multi_head_attention.plot

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
artifact_dir=artifacts/benchmark/flashdreams/accelerated/multi_head_attention
if [[ "${FLASHDREAMS_RUN_FULL_BENCHMARK:-0}" == "1" ]]; then
    artifact_dir="$artifact_dir/full"
fi
mkdir -p "$artifact_dir"

uv run --project flashdreams --group test pytest \
    flashdreams/benchmarks/accelerated/multi_head_attention \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json="$artifact_dir/benchmark.json"

uv run python -m \
    scripts.benchmark.flashdreams.accelerated.multi_head_attention.plot
