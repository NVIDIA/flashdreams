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

"""Benchmarks for regular and quantized nonpersistent linear inference.

``full-precision-x`` rows include activation quantization in the timed region,
while ``prequantized-x`` rows prepare activations and scales before timing.
Weight quantization always happens during module construction.

Rows are grouped by the linear GEMM's effective output dtype. Rowwise FP8
scaling emits BF16 and casts afterward when another output dtype is requested.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/quantization/test_quantized_linear_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    quantize,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Quantized linear benchmarks require CUDA.",
    ),
]

_LINEAR_SHAPES = (
    pytest.param(4096, 4096, 4096, "square-4096", id="square-4096"),
    pytest.param(4800, 2048, 2048, "mha-query-output", id="mha-query-output"),
    pytest.param(4800, 2048, 6144, "mha-fused-qkv", id="mha-fused-qkv"),
    pytest.param(
        28800,
        2048,
        4096,
        "mha-cross-fused-kv",
        id="mha-cross-fused-kv",
    ),
)
"""Legacy square linear layer and representative MHA projection geometries."""

_SEED = 42
_WARMUP_ROUNDS = 5
"""Warmup calls used to absorb kernel initialization and autotuning."""

_BENCHMARK_ROUNDS = 50
"""Measured calls used for each linear comparison."""

_DTYPE_FORMATS = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}

_ORIGINAL_DTYPES = (
    pytest.param(torch.float16, "fp16", id="fp16"),
    pytest.param(torch.bfloat16, "bf16", id="bf16"),
    pytest.param(torch.float32, "fp32", id="fp32"),
)
"""Source activation and output formats used by every benchmark case."""

_LINEAR_CASES = (
    pytest.param(None, None, None, None, id="nn-linear"),
    *(
        pytest.param(
            quantized_dtype,
            weight_granularity,
            input_granularity,
            prequantized,
            id=(
                f"{str(quantized_dtype).removeprefix('torch.')}"
                f"{'-x-float8_e4m3fn' if quantized_dtype is torch.float8_e5m2 else ''}"
                f"-weight-{weight_granularity.value}"
                f"-input-{input_granularity.value}"
                f"-{'prequantized-x' if prequantized else 'full-precision-x'}"
            ),
        )
        for quantized_dtype in DTYPE_MAX
        for weight_granularity in WeightGranularity
        for input_granularity in Granularity
        for prequantized in (False, True)
    ),
)
"""Regular linear baseline and every quantized linear inference configuration."""


def _effective_gemm_dtype(
    original_dtype: torch.dtype,
    quantized_dtype: torch.dtype | None,
    weight_granularity: WeightGranularity | None,
    input_granularity: Granularity | None,
) -> torch.dtype:
    """Return the dtype produced by GEMM before any output-only cast."""
    if (
        quantized_dtype is not None
        and quantized_dtype is not torch.int8
        and (
            weight_granularity is WeightGranularity.PER_OUT_CHANNEL
            or input_granularity is Granularity.SLICE
        )
    ):
        return torch.bfloat16
    return original_dtype


@pytest.mark.parametrize("m,k,n,geometry", _LINEAR_SHAPES)
@pytest.mark.parametrize(
    "quantized_dtype,weight_granularity,input_granularity,prequantized",
    _LINEAR_CASES,
)
@pytest.mark.parametrize("original_dtype,original_format", _ORIGINAL_DTYPES)
@torch.inference_mode()
def test_quantized_linear_benchmark(
    benchmark: BenchmarkFixture,
    original_dtype: torch.dtype,
    original_format: str,
    quantized_dtype: torch.dtype | None,
    weight_granularity: WeightGranularity | None,
    input_granularity: Granularity | None,
    prequantized: bool | None,
    m: int,
    k: int,
    n: int,
    geometry: str,
) -> None:
    """Benchmark regular and quantized linear inference."""
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    inputs = torch.randn(
        (m, k),
        device="cuda",
        dtype=original_dtype,
        generator=generator,
    )
    weight = torch.randn(
        (n, k),
        device="cuda",
        dtype=original_dtype,
        generator=generator,
    )

    operation: Callable[[], Tensor]
    if quantized_dtype is None:
        assert (
            weight_granularity is None
            and input_granularity is None
            and prequantized is None
        )
        linear = nn.Linear(
            k,
            n,
            bias=False,
            device="cuda",
            dtype=original_dtype,
        ).requires_grad_(False)
        linear.weight.copy_(weight)

        def operation() -> Tensor:
            return linear(inputs)

    else:
        assert weight_granularity is not None
        assert input_granularity is not None
        assert prequantized is not None
        quantized_linear = QuantizedNonPersistentLinear(
            weight,
            None,
            weight_granularity,
            quantized_dtype,
        )
        if prequantized:
            quantized_inputs, input_scale = quantize(
                inputs,
                quantized_dtype,
                input_granularity,
                axis=-1,
            )

            def operation() -> Tensor:
                return quantized_linear(
                    quantized_inputs,
                    input_scale,
                    out_dtype=original_dtype,
                )

        else:

            def operation() -> Tensor:
                return quantized_linear(
                    inputs,
                    input_granularity,
                    out_dtype=original_dtype,
                )

    effective_dtype = _effective_gemm_dtype(
        original_dtype,
        quantized_dtype,
        weight_granularity,
        input_granularity,
    )
    effective_format = _DTYPE_FORMATS[effective_dtype]
    benchmark.group = f"quantized-linear-{geometry}-{effective_format}"
    activation_dtype = original_dtype if quantized_dtype is None else quantized_dtype
    weight_dtype = (
        original_dtype
        if quantized_dtype is None
        else torch.float8_e4m3fn
        if quantized_dtype is torch.float8_e5m2
        else quantized_dtype
    )
    implementation = (
        "torch.nn.Linear" if quantized_dtype is None else "QuantizedNonPersistentLinear"
    )
    input_preparation = (
        "none"
        if quantized_dtype is None
        else "prequantized"
        if prequantized
        else "timed-quantization"
    )
    benchmark.extra_info.update(
        {
            "implementation": implementation,
            "input_preparation": input_preparation,
            "source_dtype": str(original_dtype),
            "source_format": original_format,
            "effective_dtype": str(effective_dtype),
            "effective_format": effective_format,
            "activation_dtype": str(activation_dtype),
            "weight_dtype": str(weight_dtype),
            "output_dtype": str(original_dtype),
            "quantized_dtype": (
                None if quantized_dtype is None else str(quantized_dtype)
            ),
            "weight_granularity": (
                None if weight_granularity is None else weight_granularity.value
            ),
            "input_granularity": (
                None if input_granularity is None else input_granularity.value
            ),
            "prequantized": bool(prequantized),
            "activation_quantization_timed": quantized_dtype is not None
            and not prequantized,
            "bias": False,
            "geometry": geometry,
            "batch_size": m,
            "in_features": k,
            "out_features": n,
            "m": m,
            "k": k,
            "n": n,
            "seed": _SEED,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
        }
    )

    def synchronized_linear() -> Tensor:
        output = operation()
        torch.cuda.synchronize()
        return output

    torch.cuda.synchronize()
    output = benchmark.pedantic(
        synchronized_linear,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output.shape == (m, n)
    assert output.dtype is original_dtype
    assert torch.isfinite(output).all()

    reference = F.linear(inputs, weight)
    relative_error = (
        output.float() - reference.float()
    ).norm() / reference.float().norm()
    tolerance = (
        0.0
        if quantized_dtype is None
        else (
            torch.finfo(quantized_dtype).eps
            if quantized_dtype.is_floating_point
            else 4 / DTYPE_MAX[quantized_dtype]
        )
    )
    assert relative_error.item() <= tolerance
