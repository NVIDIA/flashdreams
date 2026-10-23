# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-level CUDA merge checks against native ATen arithmetic."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.ci_gpu


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("interleaved", [False, True])
def test_fused_merge_preserves_bytes_and_empty_rows(dtype, count, interleaved):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("The fold merge kernel needs an NVIDIA CUDA GPU")
    from cosmos3_nano_sim_auto.impl.transformer._merge_kernel import (
        merge_weighted_outputs,
    )

    generator = torch.Generator(device="cuda").manual_seed(731)
    outputs, weights = [], []
    for part in range(count):
        out = torch.randn(
            (2, 3, 9, 17), device="cuda", dtype=dtype, generator=generator
        )
        if interleaved:
            out = out.transpose(1, 2).contiguous().transpose(1, 2)
        weight = torch.rand((2, 3, 9, 1), device="cuda", generator=generator)
        # Empty rows must suppress otherwise poisonous NaNs. The first sum
        # operation must also turn readable negative zero into positive zero.
        weight[..., 0, :] = 0
        out[..., 0, :] = float("nan")
        weight[..., 1, :] = 1
        out[..., 1, :] = -0.0
        weight[..., 2, :] = 0 if part else 1
        out[..., 2, :] = float("nan") if part else 3
        weight[..., 3, :] = 1e-40 if part == 0 else 0
        out[..., 3, :] = 1
        weight[..., 4, :] = 1
        out[..., 4, :] = float("inf") if part == 0 else float("-inf")
        weight[..., 5, :] = float("nan") if part == 0 else 1
        outputs.append(out)
        weights.append(weight)
    total = torch.stack(weights).sum(0)
    divisor = torch.where(total > 0, total, torch.ones_like(total))
    # Keep a subnormal numerator after division as a check against FTZ defaults.
    divisor[..., 3, :] = 1
    reference = (
        sum(
            torch.where(weight > 0, out * weight, torch.zeros_like(out))
            for out, weight in zip(outputs, weights, strict=True)
        )
        / divisor
    )
    for output_dtype in (torch.float32, dtype):
        actual = merge_weighted_outputs(
            outputs, weights, divisor, output_dtype=output_dtype
        )
        expected = reference.to(output_dtype)
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
        assert torch.all(actual[..., 0, :] == 0)
        assert not actual[..., 1, :].signbit().any()


def test_fused_merge_handles_an_empty_query_without_launching():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("The fold merge kernel needs an NVIDIA CUDA GPU")
    from cosmos3_nano_sim_auto.impl.transformer._merge_kernel import (
        merge_weighted_outputs,
    )

    out = torch.empty(1, 2, 0, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(1, 2, 0, 1, device="cuda")
    result = merge_weighted_outputs([out], [weight], weight, output_dtype=out.dtype)
    assert result.shape == out.shape
    assert result.dtype == out.dtype
    assert result.device == out.device
