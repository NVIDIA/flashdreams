# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference tests for scale-aware low-precision attention."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.multi_head_attention.triton import (
    flash_attention_2,
    flash_attention_2_tma,
)
from flashdreams.accelerated.multi_head_attention.triton.scaled_fp8_attention import (
    scaled_fp8_attention,
)

pytestmark = pytest.mark.ci_gpu


def _assert_scaled_attention_matches_sdpa(
    device: torch.device, *, use_tma: bool
) -> None:
    """Compare scaled attention with BF16 SDPA under heterogeneous magnitudes."""
    generator = torch.Generator(device=device).manual_seed(123)
    query = torch.randn(
        1, 37, 2, 64, generator=generator, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        1, 53, 2, 64, generator=generator, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        key.shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    query *= torch.logspace(-1, 1, query.shape[1], device=device)[None, :, None, None]
    key *= torch.logspace(1, -1, key.shape[1], device=device)[None, :, None, None]
    value *= torch.logspace(-1, 1, value.shape[-1], device=device)[None, None, None, :]

    actual = scaled_fp8_attention(query, key, value, use_tma=use_tma)
    raw_attention = flash_attention_2_tma if use_tma else flash_attention_2
    raw = raw_attention(
        query.to(torch.float8_e4m3fn),
        key.to(torch.float8_e4m3fn),
        value.to(torch.float8_e4m3fn),
        output_dtype=query.dtype,
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)

    scaled_rmse = (actual - expected).float().square().mean().sqrt()
    raw_rmse = (raw - expected).float().square().mean().sqrt()
    assert scaled_rmse < 0.11
    assert scaled_rmse < raw_rmse * 0.6

    # Individually tiny softmax probabilities must survive the compensated
    # E4M3 cast when their aggregate contribution is large.
    key_length = 8192
    head_dim = 128
    query = torch.ones(1, 1, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.full(
        (1, key_length, 1, head_dim),
        -7.0 / math.sqrt(head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    key[:, 0] = 0
    value = torch.ones_like(key)
    value[:, 0] = 0

    actual = scaled_fp8_attention(
        query,
        key,
        value,
        smooth_key=False,
        use_tma=use_tma,
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)

    assert expected.float().mean() > 0.8
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_scaled_fp8_pointer_attention_matches_sdpa(cuda_device: torch.device) -> None:
    """Keep quantization scales effective in the portable pointer kernel."""
    _assert_scaled_attention_matches_sdpa(cuda_device, use_tma=False)


def test_scaled_fp8_tma_attention_matches_sdpa(tma_device: torch.device) -> None:
    """Keep quantization scales effective in the TMA kernel."""
    _assert_scaled_attention_matches_sdpa(tma_device, use_tma=True)
