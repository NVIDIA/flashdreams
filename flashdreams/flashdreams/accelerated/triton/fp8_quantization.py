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

"""Row-scaled E4M3 FP8 activation quantization with Triton."""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

_FP8_MAX = 448.0
"""Finite E4M3 saturation bound used to normalize every activation row before
conversion to ``torch.float8_e4m3fn``."""


@triton.jit
def _quantize_fp8_rows_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    input_stride_row,
    input_stride_column: tl.constexpr,
    num_columns: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize one ``[C]`` activation row and store its FP32 scale.

    Each Triton program handles one row of an input matrix ``[R, C]``. Values
    are divided by ``max(max(abs(row)) / FP8_MAX, MIN_SCALE)`` before the output
    pointer converts them to E4M3. ``BLOCK_SIZE`` pads the reduction to a power
    of two; masked lanes load zero and never write, so they cannot change the
    scale or the logical output. The output is contiguous ``[R, C]`` even when
    input columns are strided, and ``scale_ptr`` stores one FP32
    dequantization scale per row as ``[R]``.

    Args:
        input_ptr: Base pointer for the source matrix ``[R, C]``.
        output_ptr: Base pointer for the contiguous E4M3 matrix ``[R, C]``.
        scale_ptr: Base pointer for the FP32 row scales ``[R]``.
        input_stride_row: Source row stride in elements.
        input_stride_column: Source column stride in elements.
        num_columns: Logical row width ``C``.
        FP8_MAX: Largest finite magnitude used for E4M3 saturation.
        MIN_SCALE: Positive scale floor used by zero-valued rows.
        BLOCK_SIZE: Power-of-two reduction width greater than or equal to ``C``.
    """
    # Select row ``r`` and a power-of-two block of candidate columns ``[C_pad]``.
    row = tl.program_id(0)
    column_offsets = tl.arange(0, BLOCK_SIZE)
    column_mask = column_offsets < num_columns

    # Load ``input[r, :]`` as FP32 and zero masked lanes so they do not affect
    # the row-wise absolute maximum. ``values`` has shape ``[C_pad]``.
    values = tl.load(
        input_ptr + row * input_stride_row + column_offsets * input_stride_column,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)

    # Reduce ``[C_pad] -> []`` to one positive scale for this row, then map the
    # row into the finite E4M3 interval. Explicit NaN propagation keeps invalid
    # source values observable instead of silently saturating them.
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / FP8_MAX, MIN_SCALE)
    quantized = tl.clamp(
        values / scale,
        -FP8_MAX,
        FP8_MAX,
        propagate_nan=tl.PropagateNan.ALL,
    )

    # Store the contiguous E4M3 row ``output[r, :]`` and scalar ``scales[r]``.
    # The output address uses the logical width rather than the input stride,
    # and the mask prevents padded reduction lanes from reaching memory.
    tl.store(
        output_ptr + row * num_columns + column_offsets,
        quantized,
        mask=column_mask,
    )
    tl.store(scale_ptr + row, scale)


def _quantize_fp8_rows(x: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize activation rows with one E4M3 scale per row.

    For row ``r``, the stored scale is
    ``max(max(abs(x[r])) / FP8_MAX, 1e-12)`` and the quantized row represents
    ``x[r] / scale[r]``. Quantization, scale reduction, and both stores execute
    in one Triton program per row. Arbitrary positive input strides are passed
    to the kernel, but fresh output storage always uses a contiguous ``[R, C]``
    layout. An empty row axis returns correctly shaped tensors without launching
    Triton.

    Args:
        x: CUDA activation matrix with shape ``[R, C]``. Rows may be strided,
            but ``C`` must be positive.

    Returns:
        Contiguous E4M3 activations with shape ``[R, C]`` and FP32 row scales
        with shape ``[R, 1]``.

    Raises:
        ValueError: ``x`` is not a two-dimensional tensor with a positive width.
    """
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must have shape [rows, columns] with columns > 0")
    num_rows, num_columns = x.shape
    output = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((num_rows, 1), device=x.device, dtype=torch.float32)
    if num_rows == 0:
        return output, scales

    # ponytail: one program owns each current <=2048-wide row; use a tiled
    # two-pass reduction if future projection widths become materially larger.
    # Four warps cover ordinary projection widths; the widest current reduction
    # uses eight. One stage is sufficient because a program performs one source
    # load followed by an in-register reduction and stores, with no tile loop.
    block_size = int(triton.next_power_of_2(num_columns))
    _quantize_fp8_rows_kernel[(num_rows,)](
        x,
        output,
        scales,
        x.stride(0),
        x.stride(1),
        num_columns,
        FP8_MAX=_FP8_MAX,
        MIN_SCALE=1e-12,
        BLOCK_SIZE=block_size,
        num_stages=1,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output, scales
