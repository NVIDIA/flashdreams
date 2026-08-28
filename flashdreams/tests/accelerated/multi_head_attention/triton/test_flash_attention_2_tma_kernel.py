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

"""Reference tests for non-causal Triton TMA FlashAttention2 kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.multi_head_attention.triton import (
    flash_attention_2_tma,
    is_tma_flash_attention_supported,
)

pytestmark = pytest.mark.ci_gpu


@pytest.mark.parametrize(
    "misaligned_input",
    [
        pytest.param(0, id="query"),
        pytest.param(1, id="key"),
        pytest.param(2, id="value"),
    ],
)
def test_tma_support_rejects_misaligned_base_pointer(
    tma_device: torch.device,
    misaligned_input: int,
) -> None:
    """Reject each Q/K/V view whose base pointer is not 16-byte aligned.

    Args:
        tma_device: CUDA device satisfying the shared TMA capability gate.
        misaligned_input: Index of the input whose storage starts one element late.
    """
    tensors = [
        torch.empty((1, 4, 2, 64), device=tma_device, dtype=torch.bfloat16),
        torch.empty((1, 6, 2, 64), device=tma_device, dtype=torch.bfloat16),
        torch.empty((1, 6, 2, 64), device=tma_device, dtype=torch.bfloat16),
    ]
    assert is_tma_flash_attention_supported(*tensors)

    original = tensors[misaligned_input]
    storage = torch.empty(
        original.numel() + 1,
        device=tma_device,
        dtype=original.dtype,
    )
    tensors[misaligned_input] = storage[1:].view_as(original)
    misaligned = tensors[misaligned_input]
    bhld_strides = (
        misaligned.stride(0),
        misaligned.stride(2),
        misaligned.stride(1),
        misaligned.stride(3),
    )
    assert misaligned.data_ptr() % 16 != 0
    assert bhld_strides[-1] == 1
    assert all(
        stride * misaligned.element_size() % 16 == 0 for stride in bhld_strides[:-1]
    )
    assert not is_tma_flash_attention_supported(*tensors)
    with pytest.raises(RuntimeError, match="base pointers and strides"):
        flash_attention_2_tma(*tensors)


@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim"),
    [
        pytest.param(37, 53, 64, id="partial-tiles"),
        pytest.param(129, 128, 128, id="production-head-divisible-key"),
    ],
)
def test_tma_flash_attention_matches_sdpa(
    tma_device: torch.device,
    query_length: int,
    key_length: int,
    head_dim: int,
) -> None:
    """Match TMA FlashAttention2 with PyTorch non-causal SDPA.

    Exercise ragged sequence tiles and a production-sized head dimension while
    preserving the public token-major ``[B, L, H, D]`` layout.

    Args:
        tma_device: CUDA device satisfying the shared TMA capability gate.
        query_length: Number of query tokens.
        key_length: Number of key and value tokens.
        head_dim: Feature width of each attention head.
    """
    generator = torch.Generator(device=tma_device).manual_seed(123)
    # Generate token-major Q/K/V tensors; Triton consumes and returns this layout.
    query = torch.randn(
        1,
        query_length,
        2,
        head_dim,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        1,
        key_length,
        2,
        head_dim,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        key.shape,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )

    actual = flash_attention_2_tma(query, key, value)
    # PyTorch SDPA consumes head-major ``[B, H, L/S, D]`` views, so transpose
    # around the reference call without changing the public comparison layout.
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)

    # Allow BF16 and online-softmax reduction-order differences between kernels.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
