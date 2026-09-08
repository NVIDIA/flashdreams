#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run and plot all default OmniDreams benchmarks:
#   ./scripts/benchmark/omnidreams/run.sh
# Run and plot all full-mode OmniDreams benchmarks:
#   FLASHDREAMS_RUN_FULL_BENCHMARK=1 ./scripts/benchmark/omnidreams/run.sh

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_module.sh" "$@"
"$script_dir/run_pipeline.sh" "$@"
