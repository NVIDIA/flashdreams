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

"""CUDA correctness tests for Torch and Triton tensor quantization."""

import pytest
import torch

from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    dequantize,
    quantize,
)

pytestmark = pytest.mark.ci_gpu

_IMPLEMENTATIONS = (
    pytest.param(False, id="torch"),
    pytest.param(True, id="triton"),
)

_GRANULARITY_AXES = (
    pytest.param(Granularity.TENSOR, -1, id="tensor"),
    pytest.param(Granularity.SLICE, 0, id="slice-axis-0"),
    pytest.param(Granularity.SLICE, 1, id="slice-axis-1"),
    pytest.param(Granularity.SLICE, -1, id="slice-axis-negative-1"),
)


@pytest.mark.parametrize("use_triton", _IMPLEMENTATIONS)
def test_quantize_preserves_reduced_dimensions(
    cuda_device: torch.device, use_triton: bool
) -> None:
    original = torch.linspace(-4.0, 4.0, 8**4, device=cuda_device).reshape(8, 8, 8, 8)

    tensor_quantized, tensor_scale = quantize(
        original, torch.float8_e4m3fn, Granularity.TENSOR, use_triton=use_triton
    )
    slice_quantized, slice_scale = quantize(
        original,
        torch.float8_e4m3fn,
        Granularity.SLICE,
        axis=2,
        use_triton=use_triton,
    )

    assert tensor_scale.shape == (1, 1, 1, 1)
    assert slice_scale.shape == (8, 8, 1, 8)
    assert tensor_quantized.dtype is torch.float8_e4m3fn
    assert slice_quantized.dtype is torch.float8_e4m3fn
    torch.testing.assert_close(
        tensor_scale,
        original.abs().amax().reshape(1, 1, 1, 1) / DTYPE_MAX[torch.float8_e4m3fn],
    )
    torch.testing.assert_close(
        slice_scale,
        original.abs().amax(dim=2, keepdim=True) / DTYPE_MAX[torch.float8_e4m3fn],
    )
    torch.testing.assert_close(
        dequantize(
            slice_quantized,
            slice_scale,
            dtype=torch.float32,
            use_triton=use_triton,
        ),
        original,
        rtol=0.06,
        atol=0.01,
    )


@pytest.mark.parametrize("use_triton", _IMPLEMENTATIONS)
def test_zero_groups_and_multiple_dequantization_scales(
    cuda_device: torch.device, use_triton: bool
) -> None:
    zeros = torch.zeros(2, 3, device=cuda_device)
    zero_quantized, zero_scale = quantize(
        zeros,
        torch.float8_e4m3fn,
        Granularity.SLICE,
        use_triton=use_triton,
    )

    assert zero_scale.shape == (2, 1)
    assert torch.isfinite(zero_scale).all()
    torch.testing.assert_close(
        dequantize(
            zero_quantized,
            zero_scale,
            dtype=torch.float32,
            use_triton=use_triton,
        ),
        zeros,
    )

    quantized = torch.tensor(
        [[1.0, -2.0], [3.0, -4.0]],
        device=cuda_device,
        dtype=torch.float8_e4m3fn,
    )
    dequantized = dequantize(
        quantized,
        torch.tensor([[0.5], [2.0]], device=cuda_device),
        torch.tensor([[2.0, 4.0]], device=cuda_device),
        torch.tensor(4.0, device=cuda_device),
        use_triton=use_triton,
    )
    assert dequantized.dtype is torch.float16
    torch.testing.assert_close(
        dequantized,
        torch.tensor(
            [[4.0, -16.0], [48.0, -128.0]],
            device=cuda_device,
            dtype=torch.float16,
        ),
    )
    torch.testing.assert_close(
        dequantize(
            quantized,
            dtype=torch.float32,
            use_triton=use_triton,
        ),
        quantized.float(),
    )


@pytest.mark.parametrize("use_triton", _IMPLEMENTATIONS)
def test_int8_quantization_rounds_to_nearest_integer(
    cuda_device: torch.device, use_triton: bool
) -> None:
    original = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], device=cuda_device)

    quantized, scale = quantize(
        original, torch.int8, Granularity.SLICE, use_triton=use_triton
    )

    assert DTYPE_MAX[torch.int8] == torch.iinfo(torch.int8).max
    assert quantized.dtype is torch.int8
    assert scale.shape == (1, 1)
    torch.testing.assert_close(
        quantized,
        torch.tensor(
            [[-127, -64, 0, 64, 127]],
            device=cuda_device,
            dtype=torch.int8,
        ),
    )
    torch.testing.assert_close(
        dequantize(
            quantized,
            scale,
            dtype=torch.float32,
            use_triton=use_triton,
        ),
        original,
        rtol=0.0,
        atol=scale.item() / 2,
    )


@pytest.mark.parametrize("dtype", DTYPE_MAX)
@pytest.mark.parametrize("granularity,axis", _GRANULARITY_AXES)
@pytest.mark.parametrize("use_triton", _IMPLEMENTATIONS)
def test_random_tensor_quantize_dequantize_round_trip(
    cuda_device: torch.device,
    dtype: torch.dtype,
    granularity: Granularity,
    axis: int,
    use_triton: bool,
) -> None:
    generator = torch.Generator(device=cuda_device).manual_seed(0)
    original = torch.randn((6, 5, 4), device=cuda_device, generator=generator).permute(
        2, 1, 0
    )

    quantized, scale = quantize(
        original,
        dtype,
        granularity,
        axis=axis,
        use_triton=use_triton,
    )
    restored = dequantize(
        quantized,
        scale,
        dtype=torch.float32,
        use_triton=use_triton,
    )

    reduction_axis: int | tuple[int, ...]
    reduction_axis = (
        tuple(range(original.ndim)) if granularity is Granularity.TENSOR else axis
    )
    expected_scale = (
        original.float().abs().amax(dim=reduction_axis, keepdim=True) / DTYPE_MAX[dtype]
    ).clamp_min(torch.finfo(torch.float32).tiny)
    torch.testing.assert_close(scale, expected_scale)

    if dtype.is_floating_point:
        rtol = torch.finfo(dtype).eps
        atol = scale.max().item()
    else:
        rtol = 0.0
        atol = scale.max().item() / 2
    torch.testing.assert_close(restored, original, rtol=rtol, atol=atol)


@pytest.mark.parametrize("use_triton", _IMPLEMENTATIONS)
def test_quantize_reduction_larger_than_one_tile(
    cuda_device: torch.device, use_triton: bool
) -> None:
    generator = torch.Generator(device=cuda_device).manual_seed(0)
    original = torch.randn(
        (2, 20_001),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    quantized, scale = quantize(
        original,
        torch.int8,
        Granularity.SLICE,
        axis=1,
        use_triton=use_triton,
    )
    expected_quantized, expected_scale = quantize(
        original, torch.int8, Granularity.SLICE, axis=1, use_triton=False
    )

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(quantized, expected_quantized, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPE_MAX)
@pytest.mark.parametrize("granularity", Granularity)
@pytest.mark.parametrize("use_triton", _IMPLEMENTATIONS)
def test_quantized_gemm(
    cuda_device: torch.device,
    dtype: torch.dtype,
    granularity: Granularity,
    use_triton: bool,
) -> None:
    generator = torch.Generator(device=cuda_device).manual_seed(1)
    left = torch.rand((32, 32), device=cuda_device, generator=generator)
    right = torch.rand((32, 32), device=cuda_device, generator=generator)

    left_quantized, left_scale = quantize(
        left, dtype, granularity, axis=1, use_triton=use_triton
    )
    if dtype is torch.int8:
        right_quantized, right_scale = quantize(
            right, dtype, granularity, axis=0, use_triton=use_triton
        )
        quantized_product = torch._int_mm(left_quantized, right_quantized)
        restored = dequantize(
            quantized_product,
            left_scale,
            right_scale,
            dtype=torch.float32,
            use_triton=use_triton,
        )
    else:
        right_dtype = torch.float8_e4m3fn if dtype is torch.float8_e5m2 else dtype
        right_quantized_t, right_scale_t = quantize(
            right.T.contiguous(),
            right_dtype,
            granularity,
            axis=1,
            use_triton=use_triton,
        )
        right_quantized = right_quantized_t.T
        right_scale = right_scale_t.T
        scaled_out_dtype = (
            torch.bfloat16 if granularity is Granularity.SLICE else torch.float32
        )
        restored = torch._scaled_mm(
            left_quantized,
            right_quantized,
            left_scale,
            right_scale,
            out_dtype=scaled_out_dtype,
        ).float()
    expected = left @ right

    if granularity is Granularity.SLICE:
        assert left_scale.shape == (32, 1)
        assert right_scale.shape == (1, 32)
    relative_error = (restored - expected).norm() / expected.norm()
    if dtype.is_floating_point:
        tolerance = torch.finfo(dtype).eps
    else:
        tolerance = 1.0 / DTYPE_MAX[dtype]
    assert relative_error.item() < tolerance
