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
from typing import Any

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

import flashdreams.accelerated.multi_head_attention.functional as functional
from flashdreams.accelerated.multi_head_attention.flex import FlexAttentionOptions
from flashdreams.accelerated.multi_head_attention.functional import (
    backend_for,
    masked_attention,
)
from flashdreams.accelerated.multi_head_attention.reference import (
    reference_masked_attention,
)

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


def test_reference_attention_zeros_fully_masked_queries() -> None:
    """Match SDPA when a query has no visible key instead of averaging values."""
    query, key, value = _qkv()
    mask = _mask()
    mask[2] = False

    actual = masked_attention(query, key, value, mask)
    expected = reference_masked_attention(query, key, value, mask)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(expected[:, :, 2], torch.zeros_like(expected[:, :, 2]))


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


def test_functional_attention_forwards_the_shared_flex_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward compilation and kernel tuning without altering head layout."""
    observed: dict[str, Any] = {}

    def compile_flex(*, dynamic: bool | None = None):
        observed["dynamic"] = dynamic

        def run(
            query: torch.Tensor,
            _key: torch.Tensor,
            value: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
            observed.update(kwargs)
            return torch.zeros(
                (*query.shape[:-1], value.shape[-1]),
                dtype=query.dtype,
                device=query.device,
            )

        return run

    monkeypatch.setattr(functional, "compiled_flex_attention", compile_flex)
    query, key, value = _qkv()
    options = FlexAttentionOptions(
        mask_block_m=16,
        mask_block_n=32,
        compile_dynamic=False,
        block_m=32,
        block_n=64,
        num_warps=4,
        prescale_qk=True,
        rows_guaranteed_safe=True,
    )
    mask = create_block_mask(
        lambda _batch, _head, query_index, key_index: key_index <= query_index,
        B=None,
        H=None,
        Q_LEN=5,
        KV_LEN=7,
        device="cpu",
        BLOCK_SIZE=options.mask_block_size,
    )

    output = masked_attention(query, key, value, mask, flex_options=options)

    assert output.shape == (1, 4, 5, 6)
    assert mask.BLOCK_SIZE == (16, 32)
    assert observed == {
        "dynamic": False,
        "block_mask": mask,
        "enable_gqa": False,
        "kernel_options": {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "num_warps": 4,
            "PRESCALE_QK": True,
            "ROWS_GUARANTEED_SAFE": True,
        },
    }


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
