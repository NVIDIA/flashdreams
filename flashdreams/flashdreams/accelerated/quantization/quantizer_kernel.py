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

"""Triton kernels for quantizing and dequantizing tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

_SUPPORTED_FORMATS = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
)

_ELEMENT_BLOCK_SIZE = 1024
"""Number of contiguous elements processed by elementwise programs."""

_MAX_REDUCTION_BLOCK_SIZE = 16384
"""Largest power-of-two reduction tile kept in one Triton program."""

_FLOAT32_TINY = tl.constexpr(torch.finfo(torch.float32).tiny)
"""Smallest positive normal FP32 value used for zero-valued scales."""


@triton.jit
def _quantize_values(
    values, scale, max_value: tl.constexpr, round_values: tl.constexpr
):
    scaled = tl.maximum(
        tl.minimum(libdevice.div_rn(values, scale), max_value), -max_value
    )
    if round_values:
        scaled = libdevice.rint(scaled)
    return scaled


@triton.jit
def _quantize_slices_kernel(
    original_ptr,
    quantized_ptr,
    scale_ptr,
    axis_size,
    inner_size,
    max_value: tl.constexpr,
    round_values: tl.constexpr,
    block_size: tl.constexpr,
):
    group_index = tl.program_id(0)
    inner_index = group_index % inner_size
    outer_index = group_index // inner_size
    group_start = outer_index * axis_size * inner_size + inner_index
    offsets = tl.arange(0, block_size)

    max_abs = 0.0
    for start in range(0, axis_size, block_size):
        axis_offsets = start + offsets
        values = tl.load(
            original_ptr + group_start + axis_offsets * inner_size,
            mask=axis_offsets < axis_size,
            other=0.0,
        ).to(tl.float32)
        max_abs = tl.maximum(max_abs, tl.max(tl.abs(values), axis=0))

    scale = tl.maximum(max_abs / max_value, _FLOAT32_TINY)
    tl.store(scale_ptr + group_index, scale)

    for start in range(0, axis_size, block_size):
        axis_offsets = start + offsets
        mask = axis_offsets < axis_size
        pointers = original_ptr + group_start + axis_offsets * inner_size
        values = tl.load(pointers, mask=mask, other=0.0).to(tl.float32)
        quantized = _quantize_values(values, scale, max_value, round_values)
        tl.store(
            quantized_ptr + group_start + axis_offsets * inner_size,
            quantized,
            mask,
        )


@triton.jit
def _partial_max_kernel(
    original_ptr, partial_max_ptr, element_count, block_size: tl.constexpr
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(
        original_ptr + offsets,
        mask=offsets < element_count,
        other=0.0,
    ).to(tl.float32)
    tl.store(partial_max_ptr + tl.program_id(0), tl.max(tl.abs(values), axis=0))


@triton.jit
def _tensor_scale_kernel(
    partial_max_ptr,
    scale_ptr,
    partial_count,
    max_value: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    max_abs = 0.0
    for start in range(0, partial_count, block_size):
        partial_offsets = start + offsets
        values = tl.load(
            partial_max_ptr + partial_offsets,
            mask=partial_offsets < partial_count,
            other=0.0,
        )
        max_abs = tl.maximum(max_abs, tl.max(values, axis=0))
    tl.store(scale_ptr, tl.maximum(max_abs / max_value, _FLOAT32_TINY))


@triton.jit
def _quantize_tensor_kernel(
    original_ptr,
    quantized_ptr,
    scale_ptr,
    element_count,
    max_value: tl.constexpr,
    round_values: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(original_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    quantized = _quantize_values(values, scale, max_value, round_values)
    tl.store(quantized_ptr + offsets, quantized, mask)


@triton.jit
def _cast_kernel(input_ptr, output_ptr, element_count, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    tl.store(output_ptr + offsets, tl.load(input_ptr + offsets, mask=mask), mask)


@triton.jit
def _multiply_broadcast_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    element_count,
    shape: tl.constexpr,
    scale_strides: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    remaining = offsets
    scale_offsets = tl.zeros([block_size], tl.int64)
    for dimension in tl.static_range(len(shape) - 1, -1, -1):
        coordinate = remaining % shape[dimension]
        remaining = remaining // shape[dimension]
        scale_offsets += coordinate * scale_strides[dimension]

    mask = offsets < element_count
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scales = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values * scales, mask)


def _reduction_block_size(element_count: int) -> int:
    """Return a bounded power-of-two reduction tile size."""
    return min(triton.next_power_of_2(element_count), _MAX_REDUCTION_BLOCK_SIZE)


def _num_warps(block_size: int) -> int:
    """Return enough warps for the selected reduction tile."""
    return 8 if block_size >= 2048 else 4


def _format_max(format: torch.dtype) -> float:
    """Return the largest finite value for a supported quantized format."""
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(f"unsupported quantization format: {format}")
    if format.is_floating_point:
        return torch.finfo(format).max
    return torch.iinfo(format).max


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize ``axis`` while retaining Torch's scalar-axis behavior."""
    if ndim == 0:
        if axis not in (-1, 0):
            raise IndexError(
                "Dimension out of range (expected to be in range of [-1, 0], "
                f"but got {axis})"
            )
        return 0
    if axis < -ndim or axis >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {axis})"
        )
    return axis % ndim


def quantize_triton(
    original: Tensor,
    format: torch.dtype,
    granularity: str,
    axis: int = -1,
) -> tuple[Tensor, Tensor]:
    """Quantize a CUDA tensor with tensorwise or axis-slice scales.

    Args:
        original: CUDA tensor to quantize.
        format: Quantized output data type.
        granularity: ``"tensor"`` or ``"slice"`` scale granularity.
        axis: Reduction dimension for slice granularity. Defaults to ``-1``.

    Returns:
        Quantized tensor and its FP32 scale tensor with reduced dimensions kept.

    Raises:
        ValueError: The input is not a CUDA tensor or an option is unsupported.
        IndexError: ``axis`` is invalid or selects an empty reduction dimension.
    """
    if not original.is_cuda:
        raise ValueError("Triton quantization requires a CUDA tensor")

    max_value = _format_max(format)
    original = original.detach().contiguous()
    quantized = torch.empty_like(original, dtype=format)

    if granularity == "tensor":
        if original.numel() == 0:
            raise IndexError("amax(): Expected reduction dim to have non-zero size")
        scale = torch.empty(
            (1,) * original.ndim,
            device=original.device,
            dtype=torch.float32,
        )
        partial_count = triton.cdiv(original.numel(), _ELEMENT_BLOCK_SIZE)
        partial_max = torch.empty(
            partial_count,
            device=original.device,
            dtype=torch.float32,
        )
        _partial_max_kernel[(partial_count,)](
            original,
            partial_max,
            original.numel(),
            block_size=_ELEMENT_BLOCK_SIZE,
        )
        reduction_block_size = _reduction_block_size(partial_count)
        _tensor_scale_kernel[(1,)](
            partial_max,
            scale,
            partial_count,
            max_value=max_value,
            block_size=reduction_block_size,
            num_warps=_num_warps(reduction_block_size),
        )
        _quantize_tensor_kernel[(partial_count,)](
            original,
            quantized,
            scale,
            original.numel(),
            max_value=max_value,
            round_values=format is torch.int8,
            block_size=_ELEMENT_BLOCK_SIZE,
        )
        return quantized, scale

    if granularity != "slice":
        raise ValueError(f"unsupported quantization granularity: {granularity}")

    normalized_axis = _normalize_axis(axis, original.ndim)
    axis_size = original.shape[normalized_axis] if original.ndim else 1
    if axis_size == 0:
        raise IndexError(
            f"amax(): Expected reduction dim {axis} to have non-zero size."
        )
    scale_shape = list(original.shape)
    if scale_shape:
        scale_shape[normalized_axis] = 1
    scale = torch.empty(scale_shape, device=original.device, dtype=torch.float32)
    if original.numel() == 0:
        return quantized, scale

    inner_size = original.stride(normalized_axis) if original.ndim else 1
    block_size = _reduction_block_size(axis_size)
    _quantize_slices_kernel[(scale.numel(),)](
        original,
        quantized,
        scale,
        axis_size,
        inner_size,
        max_value=max_value,
        round_values=format is torch.int8,
        block_size=block_size,
        num_warps=_num_warps(block_size),
    )
    return quantized, scale


def dequantize_triton(
    quantized: Tensor,
    *scales: Tensor,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize a CUDA tensor while applying broadcastable scales in order.

    Args:
        quantized: CUDA tensor to dequantize.
        scales: Scale tensors broadcastable to ``quantized``.
        dtype: Output data type. Defaults to ``torch.float16``.

    Returns:
        Dequantized tensor in ``dtype``.

    Raises:
        ValueError: A tensor is not on the same CUDA device as ``quantized``.
        RuntimeError: A scale cannot broadcast to ``quantized``.
    """
    if not quantized.is_cuda:
        raise ValueError("Triton dequantization requires a CUDA tensor")
    if any(scale.device != quantized.device for scale in scales):
        raise ValueError("dequantization scales must share the quantized tensor device")
    if quantized.numel() == 0:
        dequantized = quantized.to(scales[0].dtype) if scales else quantized
        for scale in scales:
            dequantized = dequantized * scale
        return dequantized.to(dtype)

    shape = tuple(quantized.shape)
    element_count = quantized.numel()
    grid = (triton.cdiv(element_count, _ELEMENT_BLOCK_SIZE),)
    current = quantized.contiguous()
    if not scales:
        output = torch.empty_like(current, dtype=dtype)
        _cast_kernel[grid](
            current,
            output,
            element_count,
            block_size=_ELEMENT_BLOCK_SIZE,
        )
        return output

    current_dtype = scales[0].dtype
    for index, scale in enumerate(scales):
        broadcast_scale = torch.broadcast_to(scale, shape)
        if index:
            current_dtype = torch.promote_types(current_dtype, scale.dtype)
        output_dtype = dtype if index == len(scales) - 1 else current_dtype
        output = torch.empty_like(current, dtype=output_dtype)
        _multiply_broadcast_kernel[grid](
            current,
            broadcast_scale,
            output,
            element_count,
            shape=shape,
            scale_strides=broadcast_scale.stride(),
            block_size=_ELEMENT_BLOCK_SIZE,
        )
        current = output
    return current
