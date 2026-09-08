# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for quantization benchmark matrices and plots."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.accelerated.quantization.linear import WeightGranularity
from flashdreams.accelerated.quantization.quantizer import Granularity
from scripts.benchmark.flashdreams.accelerated.quantization.plot_gemm import (
    _load_matrix as load_gemm_matrix,
)
from scripts.benchmark.flashdreams.accelerated.quantization.plot_gemm import (
    main as plot_gemm,
)
from scripts.benchmark.flashdreams.accelerated.quantization.plot_quantized_linear import (
    _load_matrix as load_linear_matrix,
)
from scripts.benchmark.flashdreams.accelerated.quantization.plot_quantized_linear import (
    main as plot_linear,
)
from scripts.benchmark.flashdreams.accelerated.quantization.plot_quantizer import (
    _load_results as load_quantizer_results,
)
from scripts.benchmark.flashdreams.accelerated.quantization.plot_quantizer import (
    main as plot_quantizer,
)

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).parents[5]


def _benchmark_globals(filename: str) -> dict[str, Any]:
    return runpy.run_path(
        str(
            _REPO_ROOT
            / "flashdreams"
            / "benchmarks"
            / "accelerated"
            / "quantization"
            / filename
        )
    )


def _record(
    group: str,
    param: str,
    median: float,
    extra_info: dict[str, object] | None = None,
) -> dict[str, object]:
    record = {"group": group, "param": param, "stats": {"median": median}}
    if extra_info is not None:
        record["extra_info"] = extra_info
    return record


def test_projection_geometry_matrix() -> None:
    expected_shapes = [
        (4096, 4096, 4096, "square-4096"),
        (4800, 2048, 2048, "mha-query-output"),
        (4800, 2048, 6144, "mha-fused-qkv"),
        (28800, 2048, 4096, "mha-cross-fused-kv"),
    ]
    gemm = _benchmark_globals("test_quantized_gemm_benchmark.py")
    linear = _benchmark_globals("test_quantized_linear_benchmark.py")

    assert [tuple(case.values) for case in gemm["_GEMM_SHAPES"]] == expected_shapes
    assert [tuple(case.values) for case in linear["_LINEAR_SHAPES"]] == expected_shapes


def test_quantizer_dequantization_scale_matrix_and_plot(tmp_path: Path) -> None:
    quantizer = _benchmark_globals("test_quantizer_benchmark.py")
    assert [case.values[0] for case in quantizer["_DEQUANTIZATION_SCALE_COUNTS"]] == [
        1,
        2,
    ]

    records = []
    for operation, scale_counts in (("quantize", (1,)), ("dequantize", (1, 2))):
        for format in ("float8_e4m3fn", "float8_e5m2", "int8"):
            for granularity in ("slice", "tensor"):
                for scale_count in scale_counts:
                    for implementation in ("torch", "triton"):
                        extra_info: dict[str, object] = {
                            "implementation": implementation
                        }
                        if scale_count > 1:
                            extra_info["scale_count"] = scale_count
                        records.append(
                            _record(
                                f"{operation}-{format}-{granularity}",
                                implementation,
                                0.001,
                                extra_info,
                            )
                        )
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(json.dumps({"benchmarks": records}), encoding="utf-8")

    values, _ = load_quantizer_results(input_path)
    assert ("dequantize", "int8", "slice", 1, "triton") in values
    assert ("dequantize", "int8", "slice", 2, "triton") in values

    output_dir = tmp_path / "quantizer"
    plot_quantizer([str(input_path), "--output-dir", str(output_dir)])
    assert (output_dir / "quantize.png").is_file()
    assert (output_dir / "dequantize.png").is_file()


def test_effective_gemm_dtype_classification() -> None:
    gemm = _benchmark_globals("test_quantized_gemm_benchmark.py")
    effective_gemm_dtype = gemm["_effective_gemm_dtype"]

    assert callable(effective_gemm_dtype)
    assert (
        effective_gemm_dtype(torch.float32, torch.float8_e4m3fn, Granularity.SLICE)
        is torch.bfloat16
    )
    assert (
        effective_gemm_dtype(torch.float32, torch.float8_e4m3fn, Granularity.TENSOR)
        is torch.float32
    )
    assert (
        effective_gemm_dtype(torch.float32, torch.int8, Granularity.SLICE)
        is torch.float32
    )


def test_effective_linear_dtype_classification() -> None:
    linear = _benchmark_globals("test_quantized_linear_benchmark.py")
    effective_gemm_dtype = linear["_effective_gemm_dtype"]

    assert callable(effective_gemm_dtype)
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.float8_e4m3fn,
            WeightGranularity.PER_OUT_CHANNEL,
            Granularity.TENSOR,
        )
        is torch.bfloat16
    )
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.float8_e4m3fn,
            WeightGranularity.TENSOR,
            Granularity.SLICE,
        )
        is torch.bfloat16
    )
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.float8_e4m3fn,
            WeightGranularity.TENSOR,
            Granularity.TENSOR,
        )
        is torch.float32
    )
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.int8,
            WeightGranularity.PER_OUT_CHANNEL,
            Granularity.SLICE,
        )
        is torch.float32
    )


def test_plot_rows_preserve_geometry_source_and_effective_dtypes(
    tmp_path: Path,
) -> None:
    geometry = "mha-query-output"
    gemm_configuration = "float8_e4m3fn-slice"
    linear_configuration = (
        "float8_e4m3fn-weight-per_out_channel-input-tensor-full-precision-x"
    )
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    _record(
                        f"quantized-gemm-{geometry}-bf16-end-to-end",
                        f"bf16-full-precision-{geometry}",
                        0.002,
                    ),
                    _record(
                        f"quantized-gemm-{geometry}-bf16-end-to-end",
                        f"fp32-{gemm_configuration}-{geometry}",
                        0.001,
                    ),
                    _record(
                        f"quantized-linear-{geometry}-bf16",
                        f"bf16-nn-linear-{geometry}",
                        0.003,
                    ),
                    _record(
                        f"quantized-linear-{geometry}-bf16",
                        f"fp32-{linear_configuration}-{geometry}",
                        0.0015,
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    gemm_rows, _, gemm_values, _ = load_gemm_matrix(input_path)
    linear_rows, _, linear_values, _ = load_linear_matrix(input_path)

    gemm_moved_row = ("end-to-end", geometry, "bf16", "fp32")
    linear_moved_row = (geometry, "bf16", "fp32")
    assert gemm_rows == [
        ("end-to-end", geometry, "bf16", "bf16"),
        gemm_moved_row,
    ]
    assert gemm_values[(gemm_moved_row, "full-precision")] == pytest.approx(2.0)
    assert gemm_values[(gemm_moved_row, gemm_configuration)] == pytest.approx(1.0)
    assert linear_rows == [(geometry, "bf16", "bf16"), linear_moved_row]
    assert linear_values[(linear_moved_row, "nn-linear")] == pytest.approx(3.0)
    assert linear_values[(linear_moved_row, linear_configuration)] == pytest.approx(1.5)

    gemm_output_dir = tmp_path / "gemm"
    linear_output_dir = tmp_path / "linear"
    plot_gemm([str(input_path), "--output-dir", str(gemm_output_dir)])
    plot_linear([str(input_path), "--output-dir", str(linear_output_dir)])

    assert (gemm_output_dir / "end_to_end.png").is_file()
    assert (linear_output_dir / "quantized_linear_quantize_input.png").is_file()
