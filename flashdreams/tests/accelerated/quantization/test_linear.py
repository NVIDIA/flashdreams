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

"""CPU tests for quantized nonpersistent linear transformations."""

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.common.non_persistent_linear import (
    NonPersistentLinear,
)
from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    quantize,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("dtype", DTYPE_MAX)
@pytest.mark.parametrize("weight_granularity", WeightGranularity)
@pytest.mark.parametrize("input_granularity", Granularity)
@pytest.mark.parametrize("use_bias", (False, True), ids=("without-bias", "with-bias"))
def test_quantized_non_persistent_linear(
    dtype: torch.dtype,
    weight_granularity: WeightGranularity,
    input_granularity: Granularity,
    use_bias: bool,
) -> None:
    """Match quantized projection paths against full-precision linear inference."""
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn((16, 32), generator=generator)
    bias = torch.randn((16,), generator=generator) if use_bias else None
    inputs = torch.randn((2, 3, 32), generator=generator)
    linear = QuantizedNonPersistentLinear(weight, bias, weight_granularity, dtype)

    quantized, scale = quantize(inputs, dtype, input_granularity, axis=-1)
    dynamic_output = linear(inputs, input_granularity)
    prequantized_output = linear(quantized, scale)

    assert dynamic_output.shape == (2, 3, 16)
    assert dynamic_output.dtype is torch.float16
    torch.testing.assert_close(dynamic_output, prequantized_output, rtol=0, atol=0)

    full_precision_output = F.linear(inputs, weight, bias)
    tolerance = (
        torch.finfo(dtype).eps if dtype.is_floating_point else 2 / DTYPE_MAX[dtype]
    )
    torch.testing.assert_close(
        dynamic_output.float(),
        full_precision_output,
        rtol=tolerance,
        atol=tolerance * full_precision_output.abs().amax().item(),
    )
    relative_error = (
        dynamic_output.float() - full_precision_output
    ).norm() / full_precision_output.norm()
    assert relative_error.item() < tolerance

    expected_weight_dtype = torch.float8_e4m3fn if dtype is torch.float8_e5m2 else dtype
    assert linear.dtype is dtype
    assert linear.weight.dtype is expected_weight_dtype


@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float32))
def test_quantized_linear_output_dtype(out_dtype: torch.dtype) -> None:
    """Return projected activations in the requested data type."""
    linear = QuantizedNonPersistentLinear(
        torch.eye(16),
        torch.ones(16),
        WeightGranularity.PER_OUT_CHANNEL,
        torch.float8_e4m3fn,
    )

    output = linear(torch.ones((2, 16)), Granularity.SLICE, out_dtype=out_dtype)

    assert output.dtype is out_dtype
    torch.testing.assert_close(output, torch.full_like(output, 2))


def test_quantized_linear_buffers_are_nonpersistent() -> None:
    """Keep derived quantized tensors out of parameters and checkpoints."""
    linear = QuantizedNonPersistentLinear(
        torch.eye(16),
        torch.ones(16),
        WeightGranularity.TENSOR,
        torch.int8,
    )

    assert isinstance(linear, NonPersistentLinear)
    assert list(linear.parameters()) == []
    assert linear.state_dict() == {}
    assert set(dict(linear.named_buffers())) == {"weight", "bias", "weight_scale"}


@pytest.mark.parametrize(
    "dtype", (torch.float8_e4m3fn, torch.float8_e5m2), ids=("e4m3", "e5m2")
)
def test_quantized_linear_dtype_cast_preserves_quantization_buffers(
    dtype: torch.dtype,
) -> None:
    """Preserve FP8 weights and exact FP32 scales across module dtype casts."""
    weight = torch.arange(1, 129, dtype=torch.float32).reshape(8, 16) / 17
    linear = QuantizedNonPersistentLinear(
        weight,
        torch.ones(8),
        WeightGranularity.PER_OUT_CHANNEL,
        dtype,
    )
    quantized_weight = linear.weight.clone()
    weight_scale = linear.weight_scale.clone()
    assert not torch.equal(weight_scale, weight_scale.bfloat16().float())

    linear.to(dtype=torch.bfloat16)

    expected_weight_dtype = torch.float8_e4m3fn if dtype is torch.float8_e5m2 else dtype
    assert linear.dtype is dtype
    assert linear.weight.dtype is expected_weight_dtype
    assert linear.weight_scale.dtype is torch.float32
    assert linear.bias is not None and linear.bias.dtype is torch.bfloat16
    torch.testing.assert_close(
        linear.weight.float(), quantized_weight.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(linear.weight_scale, weight_scale, rtol=0, atol=0)


@pytest.mark.parametrize("granularity", Granularity)
def test_quantized_linear_empty_input(granularity: Granularity) -> None:
    """Preserve empty leading dimensions without reducing an empty tensor."""
    linear = QuantizedNonPersistentLinear(
        torch.eye(16),
        None,
        WeightGranularity.TENSOR,
        torch.float8_e4m3fn,
    )

    output = linear(torch.empty((0, 3, 16)), granularity)

    assert output.shape == (0, 3, 16)
    assert output.dtype is torch.float16


def test_quantized_linear_rejects_invalid_prequantized_input() -> None:
    """Reject prequantized activations whose dtype or scale is incompatible."""
    linear = QuantizedNonPersistentLinear(
        torch.eye(16),
        None,
        WeightGranularity.TENSOR,
        torch.float8_e4m3fn,
    )
    inputs = torch.ones((2, 3, 16))
    quantized, scale = quantize(inputs, torch.float8_e4m3fn, Granularity.SLICE, axis=-1)

    with pytest.raises(ValueError, match="input dtype"):
        linear(quantized.to(torch.float8_e5m2), scale)
    with pytest.raises(ValueError, match="scale shape"):
        linear(quantized, torch.ones((2, 1)))
    with pytest.raises(ValueError, match="FP32 scale"):
        linear(quantized, scale.to(torch.float16))
    with pytest.raises(ValueError, match="last dim"):
        linear(quantized[..., :-1], scale)
    with pytest.raises(ValueError, match="scale tensor or Granularity"):
        linear(inputs, "slice")  # type: ignore[call-overload]
