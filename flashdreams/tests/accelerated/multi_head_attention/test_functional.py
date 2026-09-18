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
    DenseSDPABackend,
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


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_efficient_dense_gqa_matches_explicit_kv_expansion(kv_heads: int) -> None:
    """Match grouped K/V views to an explicitly expanded correctness oracle."""
    query, key, value = _qkv(kv_heads=kv_heads)
    mask = _mask()
    repeats = query.shape[1] // key.shape[1]

    actual = masked_attention(
        query,
        key,
        value,
        mask,
        enable_gqa=True,
        dense_backend=DenseSDPABackend.EFFICIENT,
    )
    expected = masked_attention(
        query,
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
        mask,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("query_batch", "key_batch"),
    [(1, 1), (2, 2), (1, 2), (2, 1)],
)
def test_efficient_dense_gqa_broadcasts_batches(
    query_batch: int,
    key_batch: int,
) -> None:
    """Match explicit K/V head expansion for every broadcast batch direction."""
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(query_batch, 4, 5, 8, generator=generator)
    key = torch.randn(key_batch, 2, 7, 8, generator=generator)
    value = torch.randn(key_batch, 2, 7, 6, generator=generator)
    mask = _mask()

    actual = masked_attention(
        query,
        key,
        value,
        mask,
        enable_gqa=True,
        dense_backend=DenseSDPABackend.EFFICIENT,
    )
    expected = masked_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        mask,
    )

    torch.testing.assert_close(actual, expected)


def test_efficient_dense_gqa_keeps_batch_and_head_masks() -> None:
    """Preserve a separately broadcast visibility mask for every batch and head."""
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(2, 4, 5, 8, generator=generator)
    key = torch.randn(2, 2, 7, 8, generator=generator)
    value = torch.randn(2, 2, 7, 6, generator=generator)
    mask = torch.rand(2, 4, 5, 7, generator=generator) > 0.3
    mask[..., 0] = True

    actual = masked_attention(
        query,
        key,
        value,
        mask,
        enable_gqa=True,
        dense_backend=DenseSDPABackend.EFFICIENT,
    )
    expected = masked_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        mask,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", tuple(DenseSDPABackend))
def test_dense_attention_accepts_additive_float_masks(
    backend: DenseSDPABackend,
) -> None:
    """Treat zero/-infinity masks as the boolean visibility rule they encode."""
    query, key, value = _qkv()
    boolean_mask = _mask()
    additive_mask = torch.zeros_like(boolean_mask, dtype=query.dtype).masked_fill(
        ~boolean_mask, -torch.inf
    )

    actual = masked_attention(
        query,
        key,
        value,
        additive_mask,
        dense_backend=backend,
    )
    expected = masked_attention(query, key, value, boolean_mask)

    torch.testing.assert_close(actual, expected)


def test_efficient_dense_gqa_shares_kv_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present grouped K/V to SDPA as zero-stride views instead of copies."""
    query, key, value = _qkv(kv_heads=2)
    observed: dict[str, Any] = {}

    def attention(
        grouped_query: torch.Tensor,
        grouped_key: torch.Tensor,
        grouped_value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        observed.update(
            {
                "query_shape": grouped_query.shape,
                "key_shape": grouped_key.shape,
                "key_group_stride": grouped_key.stride(1),
                "value_group_stride": grouped_value.stride(1),
                **kwargs,
            }
        )
        return torch.zeros(
            (*grouped_query.shape[:-1], grouped_value.shape[-1]),
            dtype=grouped_query.dtype,
        )

    monkeypatch.setattr(functional.F, "scaled_dot_product_attention", attention)

    mask = _mask()
    output = masked_attention(
        query,
        key,
        value,
        mask,
        enable_gqa=True,
        dense_backend=DenseSDPABackend.EFFICIENT,
    )

    assert output.shape == (1, 4, 5, 6)
    assert observed.pop("attn_mask") is mask
    assert observed == {
        "query_shape": torch.Size([2, 2, 5, 8]),
        "key_shape": torch.Size([2, 2, 7, 8]),
        "key_group_stride": 0,
        "value_group_stride": 0,
        "enable_gqa": False,
    }


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


def test_functional_attention_validates_shapes_and_heads() -> None:
    """Reject incompatible masks and implicit grouped-query layouts."""
    query, key, value = _qkv(kv_heads=2)

    with pytest.raises(ValueError, match="enable_gqa"):
        masked_attention(query, key, value, _mask())
    with pytest.raises(ValueError, match="broadcastable"):
        masked_attention(query, query, query, torch.ones(2, 5, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="does not match"):
        masked_attention(query, query, query, _mask()[:, :-1])
    with pytest.raises(TypeError, match="dense_backend"):
        masked_attention(
            query,
            key,
            value,
            _mask(),
            enable_gqa=True,
            dense_backend="efficient",  # ty: ignore[invalid-argument-type]
        )


def test_cpu_queries_leave_backend_selection_to_pytorch() -> None:
    """Avoid selecting a CUDA-only kernel for CPU tensors."""
    assert isinstance(backend_for(torch.zeros(1)), nullcontext)
