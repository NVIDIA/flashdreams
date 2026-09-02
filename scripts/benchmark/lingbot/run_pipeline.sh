#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run and plot with the default benchmark artifacts:
#   ./scripts/benchmark/lingbot/run_pipeline.sh
# Run and plot alongside the full benchmark artifacts:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 \
#     ./scripts/benchmark/lingbot/run_pipeline.sh

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
artifact_dir=artifacts/benchmark/lingbot
if [[ "${FLASHDREAMS_RUN_FULL_BENCHMARK:-0}" == "1" ]]; then
    artifact_dir="$artifact_dir/full"
fi
mkdir -p "$artifact_dir"

uv run --project integrations/lingbot --group test pytest \
    integrations/lingbot/benchmarks/test_network.py \
    integrations/lingbot/benchmarks/test_pipeline.py \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json="$artifact_dir/pipeline.json"

uv run python -m scripts.benchmark.lingbot.plot_pipeline
