#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p artifacts/benchmark/lingbot

uv run --project integrations/lingbot --group test pytest \
    integrations/lingbot/benchmarks \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json=artifacts/benchmark/lingbot/benchmark.json
