# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity and numerical checks for in-tree Sage attention."""

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.multi_head_attention.triton.scaled_fp8_attention import (
    _quantize_value_channels,
    scaled_fp8_attention,
)

pytestmark = pytest.mark.ci_gpu


@pytest.mark.parametrize("length", [65, 6032, 27144, 32768])
@pytest.mark.parametrize("scale_max", [2.0, 448.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_channel_major_value_quantization(
    tma_device: torch.device, length: int, scale_max: float, dtype: torch.dtype
) -> None:
    """Match the tiled path exactly, including zero channels and outliers."""
    torch.manual_seed(1)
    value = torch.randn(2, length, 3, 128, device=tma_device, dtype=dtype)
    value[..., 0] = 0
    value[:, -1, :, 1] = 128
    packed = value.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    expected, expected_scale = _quantize_value_channels(value, scale_max, True)
    actual, actual_scale = _quantize_value_channels(packed, scale_max, True)
    assert actual.stride(1) == 1
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("use_tma,scale", [(True, None), (False, None), (True, -0.1)])
@pytest.mark.parametrize("length", [257, 32769])
def test_channel_major_attention(
    tma_device: torch.device, use_tma: bool, scale: float | None, length: int
) -> None:
    """Preserve the attention output and pointer-kernel fallback."""
    torch.manual_seed(1)
    query = torch.randn(1, 128, 3, 128, device=tma_device, dtype=torch.bfloat16)
    key = torch.randn(1, length, 3, 128, device=tma_device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Keep channel starts aligned even for a ragged visible cache prefix.
    packed = torch.empty(
        1, 3, 128, ((length + 7) // 8) * 8, device=tma_device, dtype=value.dtype
    )
    packed = packed[..., :length].permute(0, 3, 1, 2)
    packed.copy_(value)
    kwargs = dict(
        use_tma=use_tma, scale=scale, qk_quantization_blocks=(128, 64), use_fp16_pv=True
    )
    expected = scaled_fp8_attention(query, key, value, **kwargs)
    actual = scaled_fp8_attention(query, key, packed, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [257, 27144])
@pytest.mark.parametrize("use_tma", [True, False])
def test_precomputed_key_mean(
    tma_device: torch.device, length: int, use_tma: bool
) -> None:
    """Reuse a history sum without changing the global centering operation."""
    torch.manual_seed(4)
    q = torch.randn(1, 128, 3, 128, device=tma_device, dtype=torch.bfloat16)
    k = torch.randn(1, length, 3, 128, device=tma_device, dtype=q.dtype)
    v = torch.randn_like(k)
    split = max(0, length - 6032)
    mean = (
        (
            k[:, :split].sum(dim=1, keepdim=True, dtype=torch.float32)
            + k[:, split:].sum(dim=1, keepdim=True, dtype=torch.float32)
        )
        / length
    ).to(k.dtype)
    torch.testing.assert_close(mean, k.mean(dim=1, keepdim=True), rtol=0, atol=0)
    options = dict(qk_quantization_blocks=(128, 64), use_fp16_pv=True, use_tma=use_tma)
    expected = scaled_fp8_attention(q, k, v, **options)
    actual = scaled_fp8_attention(q, k, v, key_mean=mean, **options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for invalid in (mean[:, :, :, :1], mean.float(), mean.cpu()):
        with pytest.raises(ValueError, match="key_mean"):
            scaled_fp8_attention(q, k, v, key_mean=invalid, **options)


@pytest.mark.parametrize("length", [65, 257, 27144])
@pytest.mark.parametrize("blocks", [(128, 64), (16, 16)])
def test_sage_fp16_pv_matches_float32_attention(
    tma_device: torch.device, length: int, blocks: tuple[int, int]
) -> None:
    """Keep FP8 probability error bounded for scalar and lane-wise Q/K scales."""
    torch.manual_seed(7)
    q = torch.randn(1, 128, 2, 128, device=tma_device, dtype=torch.bfloat16)
    k = torch.randn(1, length, 2, 128, device=tma_device, dtype=q.dtype)
    v = torch.randn_like(k)
    actual = scaled_fp8_attention(
        q, k, v, qk_quantization_blocks=blocks, use_fp16_pv=True
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
    ).transpose(1, 2)
    assert actual.isfinite().all()
    assert (actual.float() - expected).norm() <= 0.06 * expected.norm()


def test_sage_fp16_pv_uniform_values(tma_device: torch.device) -> None:
    """Catch missing probability normalization and overflowing tile sums."""
    q = torch.ones(1, 128, 2, 128, device=tma_device, dtype=torch.bfloat16)
    k = torch.ones(1, 27144, 2, 128, device=tma_device, dtype=q.dtype)
    v = torch.full_like(k, 64)
    v[..., 0] = 0
    actual = scaled_fp8_attention(
        q, k, v, qk_quantization_blocks=(128, 64), use_fp16_pv=True
    )
    torch.testing.assert_close(actual, v[:, :128], rtol=0, atol=0)
