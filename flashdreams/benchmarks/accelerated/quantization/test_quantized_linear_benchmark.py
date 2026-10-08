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
from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    quantize,
)
from torch import Tensor, nn

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

_TORCHAO_EAGER_NUMERICS = {
    "emulate_precision_casts": True,
    "eager_numerics.division_rounding": True,
}
"""Explicit per-compile policy for reproducing eager FP8 quantization numerics."""

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


class _RowwiseFP8Linear(QuantizedNonPersistentLinear):
    """Expose the existing dynamic FP8 path with the standard Linear signature."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x, Granularity.SLICE, out_dtype=torch.bfloat16)


@pytest.mark.parametrize("backend", ("bf16", "flashdreams", "torchao"))
@pytest.mark.parametrize("execution", ("eager", "compile", "graph", "compile_graph"))
@torch.inference_mode()
def test_torchao_comparison(
    benchmark: BenchmarkFixture, backend: str, execution: str
) -> None:
    """Measure startup and steady output-projection costs in an isolated process."""
    import os
    import statistics
    import time
    from importlib.metadata import version

    from flashdreams.accelerated.quantization.linear import TorchaoNonPersistentLinear
    from flashdreams.infra.compile import compile_module
    from flashdreams.infra.cuda_graph import CUDAGraphWrapper

    def timed(call):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = call()
        torch.cuda.synchronize()
        return value, (time.perf_counter() - start) * 1000

    m, k, n = 4800, 2048, 2048
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    inputs = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()

    def prepare():
        if backend == "bf16":
            result = nn.Linear(k, n, bias=False, device="meta")
            result.weight = nn.Parameter(weight, requires_grad=False)
            return result
        if backend == "flashdreams":
            return _RowwiseFP8Linear(
                weight, None, WeightGranularity.PER_OUT_CHANNEL, torch.float8_e4m3fn
            )
        return TorchaoNonPersistentLinear(weight, None)

    layer, preparation_ms = timed(prepare)
    # Tensor-subclass logical dtype/numel describe BF16, not packed storage.
    storages = {weight.untyped_storage().data_ptr(): weight.untyped_storage().nbytes()}
    derived_bytes = 0
    for tensor in (*layer.parameters(), *layer.buffers()):
        parts = (tensor.qdata, tensor.scale) if hasattr(tensor, "qdata") else (tensor,)
        for part in parts:
            storage = part.untyped_storage()
            if storage.data_ptr() not in storages:
                derived_bytes += storage.nbytes()
                storages[storage.data_ptr()] = storage.nbytes()
    setup_peak = torch.cuda.max_memory_allocated()
    operation = layer
    compile_wrap_ms = 0.0
    compile_options = (
        _TORCHAO_EAGER_NUMERICS
        if backend == "torchao"
        and "compile" in execution
        and os.environ.get("FLASHDREAMS_TORCHAO_EAGER_NUMERICS") == "1"
        else None
    )
    if "compile" in execution:
        operation, compile_wrap_ms = timed(
            lambda: compile_module(layer, dynamic=False, options=compile_options)
        )
    graph = (
        CUDAGraphWrapper(operation, warmup_iters=2) if "graph" in execution else None
    )
    first_call = (
        (lambda: graph.drain(inputs))
        if graph is not None
        else (lambda: operation(inputs))
    )
    _, first_call_ms = timed(first_call)

    def warmup():
        for _ in range(_WARMUP_ROUNDS):
            first_call()
        if graph is not None:
            for _ in range(graph.warmup_iters):
                graph(inputs)

    _, warmup_ms = timed(warmup)
    capture_first_call_ms = 0.0
    if graph is not None:
        _, capture_first_call_ms = timed(lambda: graph(inputs))
        assert graph._graph is not None
    active = graph if graph is not None else operation
    startup_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    steady_base = torch.cuda.memory_allocated()

    def synchronized():
        result = active(inputs)
        torch.cuda.synchronize()
        return result

    benchmark.group = "torchao-output-projection"
    output = benchmark.pedantic(
        synchronized, iterations=1, rounds=_BENCHMARK_ROUNDS, warmup_rounds=0
    )
    steady_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    event_ms = []
    for _ in range(_BENCHMARK_ROUNDS):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        active(inputs)
        end.record()
        end.synchronize()
        event_ms.append(start.elapsed_time(end))
    reference = F.linear(inputs, weight)
    eager_output = layer(inputs)
    delta = output.float() - reference.float()
    ref_norm = reference.float().norm().item()
    relative_l2 = (
        delta.norm().item() / ref_norm
        if ref_norm
        else (0.0 if not delta.any() else float("inf"))
    )
    from torch._dynamo.utils import counters

    benchmark.extra_info.update(
        {
            "backend": backend,
            "execution": execution,
            "torchao": version("torchao") if backend == "torchao" else None,
            "m": m,
            "k": k,
            "n": n,
            "bias": False,
            "seed": _SEED,
            "source_dtype": "bfloat16",
            "activation_quantization_timed": backend != "bf16",
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "cache_state": os.environ.get(
                "FLASHDREAMS_BENCH_CACHE_STATE", "unspecified"
            ),
            "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "preparation_ms": preparation_ms,
            "compile_wrap_ms": compile_wrap_ms,
            "compile_options": compile_options,
            "first_call_ms": first_call_ms,
            "warmup_ms": warmup_ms,
            "capture_first_call_ms": capture_first_call_ms,
            "cuda_event_samples_ms": event_ms,
            "cuda_event_median_ms": statistics.median(event_ms),
            "cuda_event_p90_ms": sorted(event_ms)[44],
            "source_weight_bytes": weight.untyped_storage().nbytes(),
            "derived_weight_bytes": derived_bytes,
            "resident_weight_bytes": sum(storages.values()),
            "input_and_source_allocated_bytes": base_allocated,
            "preparation_peak_allocated_bytes": setup_peak,
            "startup_peak_allocated_bytes": startup_peak,
            "steady_base_allocated_bytes": steady_base,
            "steady_peak_allocated_bytes": steady_peak,
            "steady_peak_reserved_bytes": reserved_peak,
            "relative_l2": relative_l2,
            "eager_relative_l2": (
                (output.float() - eager_output.float()).norm()
                / eager_output.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
            ).item(),
            "eager_max_abs_error": (output.float() - eager_output.float())
            .abs()
            .max()
            .item(),
            "eager_mismatch_fraction": (
                ~torch.isclose(output, eager_output, rtol=0.02, atol=0.02)
            )
            .float()
            .mean()
            .item(),
            "mae": delta.abs().mean().item(),
            "max_abs_error": delta.abs().max().item(),
            "cosine_similarity": F.cosine_similarity(
                output.float().flatten(), reference.float().flatten(), dim=0
            ).item(),
            "dynamo_counters": {key: dict(value) for key, value in counters.items()},
            "requested_torchao_config": "PerRow/e4m3/TORCH/fast_accum=False/inductor_config=False/activation_lb=float32.tiny*448"
            if backend == "torchao"
            else None,
        }
    )

    # Record evidence before assertions, including numerical compatibility failures.
    assert output.dtype is torch.bfloat16 and output.shape == (m, n)
    assert torch.isfinite(output).all()
    assert relative_l2 <= (
        0.0 if backend == "bf16" else torch.finfo(torch.float8_e4m3fn).eps
    )
    torch.testing.assert_close(output, eager_output, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("seed", (0, 1, 42))
@pytest.mark.parametrize("use_bias", (False, True))
@torch.inference_mode()
def test_torchao_compile_lifecycle(seed: int, use_bias: bool) -> None:
    """Preserve random/zero numerics and output ownership through recompilation."""
    from torch._inductor import config

    from flashdreams.accelerated.quantization.linear import TorchaoNonPersistentLinear
    from flashdreams.infra.compile import compile_module
    from flashdreams.infra.cuda_graph import CUDAGraphWrapper

    before = (config.emulate_precision_casts, config.eager_numerics.division_rounding)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    source = torch.randn(
        128, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    bias = (
        torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=generator)
        if use_bias
        else None
    )
    layer = TorchaoNonPersistentLinear(source, bias)
    compiled = compile_module(layer, dynamic=False, options=_TORCHAO_EAGER_NUMERICS)
    x = torch.randn(
        2, 16, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    torch.testing.assert_close(compiled(x), layer(x), rtol=0.02, atol=0.02)
    reference = F.linear(x, source, bias)
    assert (
        compiled(x).float() - reference.float()
    ).norm() / reference.float().norm() < 0.125
    zero = torch.zeros_like(x)
    torch.testing.assert_close(
        compiled(zero), F.linear(zero, source, bias), rtol=0, atol=0
    )
    graph = CUDAGraphWrapper(compiled, warmup_iters=2)
    graph.drain(x)
    for _ in range(3):
        retained = graph(x)
    retained_copy = retained.clone()
    graph(x * 2)
    torch.testing.assert_close(retained, retained_copy, rtol=0, atol=0)
    torch.testing.assert_close(retained, layer(x), rtol=0.02, atol=0.02)
    changed_shape = x[:, :8].contiguous()
    graph.drain(changed_shape)
    for _ in range(3):
        output = graph(changed_shape)
    torch.testing.assert_close(output, layer(changed_shape), rtol=0.02, atol=0.02)

    graph.reset()
    replacement = TorchaoNonPersistentLinear(source * 0.5, bias)
    graph.fn = compile_module(
        replacement, dynamic=False, options=_TORCHAO_EAGER_NUMERICS
    )
    graph.drain(x)
    for _ in range(3):
        output = graph(x)
    torch.testing.assert_close(output, replacement(x), rtol=0.02, atol=0.02)
    torch.testing.assert_close(retained, retained_copy, rtol=0, atol=0)
    assert not torch.equal(output, retained)
    assert (
        config.emulate_precision_casts,
        config.eager_numerics.division_rounding,
    ) == before
