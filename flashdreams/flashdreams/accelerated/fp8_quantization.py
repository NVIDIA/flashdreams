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

"""Row-scaled E4M3 FP8 quantization and linear projection helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from flashdreams.accelerated.triton.fp8_quantization import (
    _FP8_MAX,
    _quantize_fp8_rows,
)


@torch.no_grad()
def quantize_fp8_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a linear weight with one FP32 scale per output row.

    A weight row maps every input feature into one output feature. For output
    row ``o``, this stores
    ``scale[o] = max(max(abs(weight[o])) / 448, 1e-12)`` and converts
    ``weight[o] / scale[o]`` to E4M3. Per-row scaling therefore preserves the
    linear layer's ``[O, I]`` layout while giving every output feature its own
    dequantization factor. The returned tensors are detached inference data;
    gradients continue to belong to the source parameter.

    Args:
        weight: Linear weight with shape ``[O, I]``.

    Returns:
        Contiguous E4M3 weight with shape ``[O, I]`` and its FP32 scales with
        shape ``[O]``.
    """
    # Compute one scale per output feature: ``[O, I] -> [O]``.
    weight_float = weight.detach().to(torch.float32)
    scale = (weight_float.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)

    # Broadcast ``[O] -> [O, 1]`` to normalize each row before E4M3 conversion.
    weight_fp8 = (
        (weight_float / scale[:, None])
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return weight_fp8, scale.contiguous()


def fp8_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None,
    out_dtype: torch.dtype,
) -> Tensor:
    """Apply a row-scaled FP8 GEMM and restore the requested activation dtype.

    Leading activation dimensions are flattened into ``R`` rows for the GEMM.
    Native-precision inputs are dynamically quantized with one scale per row;
    E4M3 inputs are interpreted as already quantized with unit row scales.
    Weight scales broadcast over rows, so each output feature is independently
    dequantized before the optional bias is applied.

    Args:
        x: Native-precision or E4M3 activations with shape ``[..., I]``.
        weight: E4M3 weight with shape ``[O, I]`` produced by
            :func:`quantize_fp8_weight`.
        weight_scale: Per-output-feature FP32 scales with shape ``[O]``.
        bias: Optional output bias with shape ``[O]``.
        out_dtype: Activation dtype returned to the caller.

    Returns:
        Projected activations with shape ``[..., O]``.
    """
    # Collapse arbitrary leading dimensions for GEMM:
    # ``[..., I] -> [R, I]``, where ``R = prod(x.shape[:-1])``.
    input_shape = x.shape
    x_2d = x.reshape(-1, input_shape[-1])

    # Supply activation scales as ``[R, 1]``. Native inputs are dynamically
    # quantized per row; pre-quantized E4M3 inputs use an identity scale.
    if x_2d.dtype == torch.float8_e4m3fn:
        x_fp8 = x_2d
        input_scale = torch.ones(
            (x_2d.shape[0], 1),
            device=x.device,
            dtype=torch.float32,
        )
    else:
        x_fp8, input_scale = _quantize_fp8_rows(x_2d)

    # Multiply ``[R, I] @ [I, O] -> [R, O]``. The row scales ``[R, 1]``
    # and transposed-weight scales ``[1, O]`` broadcast over that output.
    scaled_bias = bias.to(torch.bfloat16) if bias is not None else None
    output = torch._scaled_mm(
        x_fp8,
        weight.T,
        input_scale,
        weight_scale.reshape(1, -1).contiguous(),
        bias=scaled_bias,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    if out_dtype != torch.bfloat16:
        output = output.to(out_dtype)

    # Restore the original leading dimensions: ``[R, O] -> [..., O]``.
    return output.reshape(input_shape[:-1] + (weight.shape[0],))


__all__ = ["fp8_linear", "quantize_fp8_weight"]
