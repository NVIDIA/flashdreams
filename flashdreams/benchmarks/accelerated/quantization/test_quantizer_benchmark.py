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

"""Benchmarks comparing Torch and Triton tensor quantization.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/quantization/test_quantizer_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    dequantize,
    quantize,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Quantizer benchmarks require CUDA.",
    ),
]

_SHAPE = (4096, 4096)
"""Matrix shape representative of large projection activations and weights."""

_SEED = 42

_WARMUP_ROUNDS = 5
"""Warmup calls used to absorb Triton compilation and CUDA initialization."""

_BENCHMARK_ROUNDS = 50
"""Measured calls used for each Torch and Triton comparison."""

_IMPLEMENTATIONS = (
    pytest.param(False, "torch", id="torch"),
    pytest.param(True, "triton", id="triton"),
)

_DEQUANTIZATION_SCALE_COUNTS = (
    pytest.param(1, id="one-scale"),
    pytest.param(2, id="two-scales"),
)
"""Single-scale and row/column-scaled GEMM restoration cases."""


def _record_case(
    benchmark: BenchmarkFixture,
    operation: str,
    format: torch.dtype,
    granularity: Granularity,
    implementation: str,
    scale_shapes: tuple[tuple[int, ...], ...],
) -> None:
    """Attach metadata for one quantize or dequantize case."""
    source_dtype = torch.float16 if operation == "quantize" else format
    output_dtype = format if operation == "quantize" else torch.float16
    benchmark.extra_info.update(
        {
            "operation": operation,
            "implementation": implementation,
            "shape": _SHAPE,
            "axis": -1,
            "source_dtype": str(source_dtype),
            "output_dtype": str(output_dtype),
            "quantized_dtype": str(format),
            "granularity": granularity.value,
            "scale_count": len(scale_shapes),
            "scale_shapes": scale_shapes,
            "scale_dtype": str(torch.float32),
            "seed": _SEED,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
        }
    )


@pytest.mark.parametrize("format", DTYPE_MAX)
@pytest.mark.parametrize("granularity", Granularity)
@pytest.mark.parametrize("use_triton,implementation", _IMPLEMENTATIONS)
def test_quantize_benchmark(
    benchmark: BenchmarkFixture,
    format: torch.dtype,
    granularity: Granularity,
    use_triton: bool,
    implementation: str,
) -> None:
    """Benchmark quantization through one Torch or Triton implementation."""
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    original = torch.randn(
        _SHAPE,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    format_name = str(format).removeprefix("torch.")
    benchmark.group = f"quantize-{format_name}-{granularity.value}"
    expected_scale_shape = (
        (1, 1) if granularity is Granularity.TENSOR else (_SHAPE[0], 1)
    )
    _record_case(
        benchmark,
        "quantize",
        format,
        granularity,
        implementation,
        (expected_scale_shape,),
    )

    def synchronized_quantize() -> tuple[Tensor, Tensor]:
        output = quantize(
            original,
            format,
            granularity,
            axis=-1,
            use_triton=use_triton,
        )
        torch.cuda.synchronize()
        return output

    torch.cuda.synchronize()
    quantized, scale = benchmark.pedantic(
        synchronized_quantize,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert quantized.shape == _SHAPE
    assert quantized.dtype is format
    assert scale.shape == expected_scale_shape
    assert torch.isfinite(quantized.float()).all()
    assert torch.isfinite(scale).all()


@pytest.mark.parametrize("format", DTYPE_MAX)
@pytest.mark.parametrize("granularity", Granularity)
@pytest.mark.parametrize("use_triton,implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("scale_count", _DEQUANTIZATION_SCALE_COUNTS)
def test_dequantize_benchmark(
    benchmark: BenchmarkFixture,
    format: torch.dtype,
    granularity: Granularity,
    use_triton: bool,
    implementation: str,
    scale_count: int,
) -> None:
    """Benchmark dequantization through one Torch or Triton implementation."""
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    original = torch.randn(
        _SHAPE,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    quantized, row_scale = quantize(
        original,
        format,
        granularity,
        axis=-1,
        use_triton=False,
    )
    scales = (
        (row_scale,) if scale_count == 1 else (row_scale, row_scale.mT.contiguous())
    )
    format_name = str(format).removeprefix("torch.")
    benchmark.group = f"dequantize-{format_name}-{granularity.value}"
    _record_case(
        benchmark,
        "dequantize",
        format,
        granularity,
        implementation,
        tuple(tuple(scale.shape) for scale in scales),
    )

    def synchronized_dequantize() -> Tensor:
        output = dequantize(
            quantized,
            *scales,
            dtype=torch.float16,
            use_triton=use_triton,
        )
        torch.cuda.synchronize()
        return output

    torch.cuda.synchronize()
    dequantized = benchmark.pedantic(
        synchronized_dequantize,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert dequantized.shape == _SHAPE
    assert dequantized.dtype is torch.float16
    assert torch.isfinite(dequantized).all()
