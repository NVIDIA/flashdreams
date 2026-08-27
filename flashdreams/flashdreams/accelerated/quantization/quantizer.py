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

"""Quantization granularity and tensor conversion interfaces."""

from enum import Enum

import torch
from torch import Tensor

from flashdreams.accelerated.quantization.quantizer_kernel import (
    dequantize_triton,
    quantize_triton,
)

DTYPE_MAX: dict[torch.dtype, float] = {
    torch.float8_e4m3fn: torch.finfo(torch.float8_e4m3fn).max,
    torch.float8_e5m2: torch.finfo(torch.float8_e5m2).max,
    torch.int8: torch.iinfo(torch.int8).max,
}
"""Largest finite positive value for each supported quantized dtype."""


class Granularity(str, Enum):
    """Scale granularity for tensor quantization."""

    SLICE = "slice"
    """Use one scale per slice selected by ``axis`` in :func:`quantize`."""

    TENSOR = "tensor"
    """Use one scale for the complete tensor."""


def quantize(
    original: Tensor,
    format: torch.dtype,
    granularity: Granularity,
    axis: int = -1,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor]:
    """Quantize a tensor using the requested format and scale granularity.

    Args:
        original: Tensor to quantize.
        format: Data type of the quantized tensor.
        granularity: Scope over which scales are computed. ``SLICE`` reduces
            along ``axis``; ``TENSOR`` reduces over the complete tensor.
        axis: Dimension over which ``SLICE`` computes the maximum value. For a
            two-dimensional tensor, ``axis=1`` produces one scale per row and
            ``axis=0`` produces one scale per column. Defaults to ``-1``.
        use_triton: Use Triton for CUDA tensors. CPU tensors retain the Torch
            implementation. Defaults to ``True``.

    Returns:
        Quantized tensor and its FP32 scale tensor. The scale has the same
        number of dimensions as ``original`` and retains reduced dimensions
        with size one.

    Raises:
        ValueError: ``format`` or ``granularity`` is unsupported.
    """
    if format not in DTYPE_MAX:
        raise ValueError(f"unsupported quantization format: {format}")

    if granularity is Granularity.TENSOR:
        reduction_axis: int | tuple[int, ...] = tuple(range(original.ndim))
    elif granularity is Granularity.SLICE:
        reduction_axis = axis
    else:
        raise ValueError(f"unsupported quantization granularity: {granularity}")

    if use_triton and original.is_cuda:
        return quantize_triton(original, format, granularity, axis)

    original_float = original.detach().to(torch.float32)
    max_abs = original_float.abs().amax(dim=reduction_axis, keepdim=True)
    scale = (max_abs / DTYPE_MAX[format]).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (original_float / scale).clamp(-DTYPE_MAX[format], DTYPE_MAX[format])
    if not format.is_floating_point:
        quantized = quantized.round()
    return quantized.to(format), scale


def dequantize(
    quantized: Tensor,
    *scales: Tensor,
    dtype: torch.dtype = torch.float16,
    use_triton: bool = True,
) -> Tensor:
    """Dequantize a tensor using its scale tensors.

    Args:
        quantized: Tensor to dequantize.
        scales: Scale tensors used to dequantize ``quantized``.
        dtype: Data type of the dequantized tensor. Defaults to ``torch.float16``.
        use_triton: Use Triton for CUDA tensors. CPU tensors retain the Torch
            implementation. Defaults to ``True``.

    Returns:
        Dequantized tensor in ``dtype`` after applying every scale in order.
        Without scales, casts ``quantized`` directly to ``dtype``.
    """
    if use_triton and quantized.is_cuda:
        return dequantize_triton(quantized, *scales, dtype=dtype)
    if not scales:
        return quantized.to(dtype)

    dequantized = quantized.to(scales[0].dtype)
    for scale in scales:
        dequantized = dequantized * scale
    return dequantized.to(dtype)
