#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
mkdir -p artifacts/benchmark/flashdreams/accelerated/quantization

uv run --project flashdreams --group test pytest \
    flashdreams/benchmarks/accelerated/quantization/test_quantizer_benchmark.py \
    flashdreams/benchmarks/accelerated/quantization/test_quantized_gemm_benchmark.py \
    flashdreams/benchmarks/accelerated/quantization/test_quantized_linear_benchmark.py \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json=artifacts/benchmark/flashdreams/accelerated/quantization/benchmark.json

uv run python -m \
    scripts.benchmark.flashdreams.accelerated.quantization.plot_quantizer
uv run python -m \
    scripts.benchmark.flashdreams.accelerated.quantization.plot_gemm
uv run python -m \
    scripts.benchmark.flashdreams.accelerated.quantization.plot_quantized_linear
