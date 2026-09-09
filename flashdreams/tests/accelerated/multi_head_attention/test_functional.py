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

"""CPU tests for functional dense and block-sparse attention dispatch."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from flashdreams.accelerated.multi_head_attention.functional import (
    backend_for,
    masked_attention,
)
from flashdreams.accelerated.multi_head_attention.reference import (
    reference_masked_attention,
)
from torch.nn.attention.flex_attention import create_block_mask

pytestmark = pytest.mark.ci_cpu


def _qkv(*, kv_heads: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(3)
    query = torch.randn(1, 4, 5, 8, generator=generator)
    key = torch.randn(1, kv_heads, 7, 8, generator=generator)
    value = torch.randn(1, kv_heads, 7, 6, generator=generator)
    return query, key, value


def _mask() -> torch.Tensor:
    rows = torch.arange(5)[:, None]
    columns = torch.arange(7)[None, :]
    return (columns <= rows + 1) | (columns == 6)


def test_dense_attention_matches_the_explicit_reference() -> None:
    """Keep dense dispatch pinned to the public correctness oracle."""
    query, key, value = _qkv()
    mask = _mask()

    actual = masked_attention(query, key, value, mask)
    expected = reference_masked_attention(query, key, value, mask)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("enable_gqa", [False, True])
def test_block_and_dense_masks_share_one_attention_contract(enable_gqa: bool) -> None:
    """Use the same visibility rule for dense and block-sparse backends."""
    query, key, value = _qkv(kv_heads=2 if enable_gqa else 4)
    mask = _mask()
    block_mask = create_block_mask(
        lambda _batch, _head, query_index, key_index: mask[query_index, key_index],
        B=None,
        H=None,
        Q_LEN=5,
        KV_LEN=7,
        device="cpu",
        BLOCK_SIZE=16,
    )

    dense = masked_attention(query, key, value, mask, enable_gqa=enable_gqa)
    blocks = masked_attention(
        query,
        key,
        value,
        block_mask,
        enable_gqa=enable_gqa,
    )

    torch.testing.assert_close(blocks, dense, atol=1e-5, rtol=1e-5)


def test_functional_attention_validates_exact_shapes_and_heads() -> None:
    """Reject broadcastable masks and implicit grouped-query layouts."""
    query, key, value = _qkv(kv_heads=2)

    with pytest.raises(ValueError, match="enable_gqa"):
        masked_attention(query, key, value, _mask())
    with pytest.raises(ValueError, match="two-dimensional"):
        masked_attention(query, query, query, _mask()[None])
    with pytest.raises(ValueError, match="does not match"):
        masked_attention(query, query, query, _mask()[:, :-1])


def test_cpu_queries_leave_backend_selection_to_pytorch() -> None:
    """Avoid selecting a CUDA-only kernel for CPU tensors."""
    assert isinstance(backend_for(torch.zeros(1)), nullcontext)
