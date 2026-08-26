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

"""Reference tests for non-causal Triton FlashAttention2 kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.multi_head_attention.triton import flash_attention_2

pytestmark = pytest.mark.ci_gpu


@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim"),
    [
        pytest.param(37, 53, 64, id="partial-tiles"),
        pytest.param(129, 128, 128, id="production-head-divisible-key"),
    ],
)
def test_flash_attention_matches_sdpa(
    cuda_device: torch.device,
    query_length: int,
    key_length: int,
    head_dim: int,
) -> None:
    """Match pointer-based FlashAttention2 with PyTorch non-causal SDPA.

    Exercise ragged sequence tiles and a production-sized head dimension while
    preserving the public token-major ``[B, L, H, D]`` layout.

    Args:
        cuda_device: Active CUDA device.
        query_length: Number of query tokens.
        key_length: Number of key and value tokens.
        head_dim: Feature width of each attention head.
    """
    generator = torch.Generator(device=cuda_device).manual_seed(123)
    # Generate token-major Q/K/V tensors; Triton consumes and returns this layout.
    query = torch.randn(
        1,
        query_length,
        2,
        head_dim,
        generator=generator,
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        1,
        key_length,
        2,
        head_dim,
        generator=generator,
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        key.shape,
        generator=generator,
        device=cuda_device,
        dtype=torch.bfloat16,
    )

    actual = flash_attention_2(query, key, value)
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
