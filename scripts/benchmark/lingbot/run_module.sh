#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run and plot the representative module benchmark:
#   ./scripts/benchmark/lingbot/run_module.sh
# Replot its saved results:
#   uv run python -m scripts.benchmark.lingbot.plot_module
#
# Run and plot all 1,369 full-block self/cross pairs:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 \
#     ./scripts/benchmark/lingbot/run_module.sh
# Replot their saved results:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 \
#     uv run python -m scripts.benchmark.lingbot.plot_module

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
artifact_dir=artifacts/benchmark/lingbot
if [[ "${FLASHDREAMS_RUN_FULL_BENCHMARK:-0}" == "1" ]]; then
    artifact_dir="$artifact_dir/full"
fi
mkdir -p "$artifact_dir"

uv run --project integrations/lingbot --group test pytest \
    integrations/lingbot/benchmarks/test_modules.py \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json="$artifact_dir/module.json"

uv run python -m scripts.benchmark.lingbot.plot_module
