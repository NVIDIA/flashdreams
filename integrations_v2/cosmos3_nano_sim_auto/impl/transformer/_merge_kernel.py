# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton vector merge for the maskless fold's native FP32 weights.

Imported only by the optional CUDA inference path. Weight calculation stays in
:mod:`cosmos3_nano_sim_auto.impl.transformer.attention`, including its exp, reduction and empty-row policy.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice


@triton.jit
def _merge_weighted_kernel(
    OUTS,
    WEIGHTS,
    DENOMINATOR,
    RESULT,
    N: tl.constexpr,
    H: tl.constexpr,
    Q: tl.constexpr,
    D: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    WEIGHT_STRIDES: tl.constexpr,
    DENOM_STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    d = index % D
    q = (index // D) % Q
    h = (index // (D * Q)) % H
    b = index // (D * Q * H)
    merged = tl.full((BLOCK,), 0.0, tl.float32)
    for part in tl.static_range(PARTS):
        out = tl.load(
            OUTS[part]
            + b * OUT_STRIDES[part][0]
            + h * OUT_STRIDES[part][1]
            + q * OUT_STRIDES[part][2]
            + d * OUT_STRIDES[part][3],
            valid,
            other=0,
        ).to(tl.float32)
        weight = tl.load(
            WEIGHTS[part]
            + b * WEIGHT_STRIDES[part][0]
            + h * WEIGHT_STRIDES[part][1]
            + q * WEIGHT_STRIDES[part][2],
            valid,
            other=0,
        ).to(tl.float32)
        weighted = libdevice.mul_rn(out, weight)
        term = tl.where(weight > 0, weighted, 0.0)
        if part == 0:
            # Python sum starts at integer zero: its first tensor step is term+0.
            # Keep the operation explicit, including the sign of an input -0.
            merged = libdevice.add_rn(term, 0.0)
        else:
            merged = libdevice.add_rn(merged, term)
    divisor = tl.load(
        DENOMINATOR
        + b * DENOM_STRIDES[0]
        + h * DENOM_STRIDES[1]
        + q * DENOM_STRIDES[2],
        valid,
        other=1,
    ).to(tl.float32)
    quotient = libdevice.div_rn(merged, divisor)
    tl.store(RESULT + index, quotient, valid)


def merge_weighted_outputs(
    outputs: Sequence[Tensor],
    weights: Sequence[Tensor],
    divisor: Tensor,
    *,
    output_dtype: torch.dtype,
) -> Tensor:
    """Merge output vectors with already computed native weights and divisor.

    Args:
        outputs: CUDA ``[B, H, Q, D]`` float32, float16 or bfloat16 tensors.
        weights: One float32 ``[B, H, Q, 1]`` tensor per output.
        divisor: The safe float32 ``[B, H, Q, 1]`` denominator.
        output_dtype: The final model dtype, or float32 to keep the wider merge.

    Returns:
        A contiguous ``[B, H, Q, D]`` tensor. The caller selects this inference
        path only for supported inputs; the ordinary merge handles fallbacks.
    """
    # Triton launches on the current device; a caller can hold tensors from a
    # different CUDA device without having made that device current.
    with torch.cuda.device(outputs[0].device):
        return _launch_merge(outputs, weights, divisor, output_dtype=output_dtype)


def _launch_merge(
    outputs: Sequence[Tensor],
    weights: Sequence[Tensor],
    divisor: Tensor,
    *,
    output_dtype: torch.dtype,
) -> Tensor:
    shape = outputs[0].shape
    result = torch.empty(shape, device=outputs[0].device, dtype=output_dtype)
    if result.numel():
        _merge_weighted_kernel[(triton.cdiv(result.numel(), 256),)](
            tuple(outputs),
            tuple(weights),
            divisor,
            result,
            result.numel(),
            H=shape[1],
            Q=shape[2],
            D=shape[3],
            OUT_STRIDES=tuple(out.stride() for out in outputs),
            WEIGHT_STRIDES=tuple(weight.stride() for weight in weights),
            DENOM_STRIDES=divisor.stride(),
            PARTS=len(outputs),
            BLOCK=256,
            num_warps=4,
            # Preserve separate FP32 rounding, signed zero and subnormals.
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return result
