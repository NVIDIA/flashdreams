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

"""CPU tests for segmented K/V attention arenas."""

from __future__ import annotations

import math

import pytest
import torch

from flashdreams.accelerated.multi_head_attention.kv_arena import (
    KVSegmentLayout,
    allocate_kv_arena,
    stage_current_kv_,
)

pytestmark = pytest.mark.ci_cpu


def pair(tokens: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, 2, tokens, 4)
    key = torch.arange(start, start + math.prod(shape)).reshape(shape).float()
    return key, key + 1000


def test_staged_arena_is_bitwise_the_same_as_concatenation() -> None:
    static, current, memory = pair(3), pair(2, 100), pair(5, 200)
    arena, layout = allocate_kv_arena(static, memory, current_tokens=2, memory_tokens=5)

    actual = stage_current_kv_(arena, current, layout)
    expected = tuple(
        torch.cat((static[index], current[index], memory[index]), dim=2)
        for index in range(2)
    )

    assert all(tensor.is_contiguous() for tensor in actual)
    assert all(
        torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
    )


def test_repeated_staging_changes_only_the_current_segment() -> None:
    static, memory = pair(3), pair(5, 200)
    arena, layout = allocate_kv_arena(static, memory, current_tokens=2, memory_tokens=5)
    before_static = tuple(tensor[:, :, :3].clone() for tensor in arena)
    before_memory = tuple(tensor[:, :, 5:].clone() for tensor in arena)

    stage_current_kv_(arena, pair(2, 100), layout)
    stage_current_kv_(arena, pair(2, 300), layout)

    assert all(
        torch.equal(tensor[:, :, :3], expected)
        for tensor, expected in zip(arena, before_static, strict=True)
    )
    assert all(
        torch.equal(tensor[:, :, 5:], expected)
        for tensor, expected in zip(arena, before_memory, strict=True)
    )


def test_allocation_leaves_memory_reserve_zeroed() -> None:
    arena, layout = allocate_kv_arena(
        pair(1), pair(2, 100), current_tokens=3, memory_tokens=4
    )

    assert layout == KVSegmentLayout(1, 3, 4)
    assert all(torch.count_nonzero(tensor[:, :, -2:]) == 0 for tensor in arena)


def test_invalid_shapes_and_aliasing_are_refused() -> None:
    arena, layout = allocate_kv_arena(
        pair(1), pair(2, 100), current_tokens=2, memory_tokens=2
    )
    with pytest.raises(ValueError, match="must have shape"):
        stage_current_kv_(arena, pair(1), layout)
    current = (arena[0][:, :, 1:3], arena[1][:, :, 1:3])
    with pytest.raises(ValueError, match="must not alias"):
        stage_current_kv_(arena, current, layout)


def test_arena_requires_contiguous_distinct_storage() -> None:
    arena, layout = allocate_kv_arena(
        pair(1), pair(2, 100), current_tokens=2, memory_tokens=2
    )
    shape = arena[0].shape
    strided_key = arena[0].new_empty((*shape[:-1], shape[-1] * 2))[..., ::2]

    with pytest.raises(ValueError, match="must be contiguous"):
        stage_current_kv_((strided_key, arena[1]), pair(2), layout)
    with pytest.raises(ValueError, match="distinct storage"):
        stage_current_kv_((arena[0], arena[0]), pair(2), layout)


def test_layout_rejects_nonpositive_current_capacity() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        KVSegmentLayout(1, 0, 2)
