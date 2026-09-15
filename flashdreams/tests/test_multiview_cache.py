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

"""CPU tests for fixed-slot multi-view inference state."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.attention.kvcache import (
    FixedSlotKVCache,
    LayerKV,
    SlotRegion,
    TokenWindow,
)

pytestmark = pytest.mark.ci_cpu


def layer(tokens: int, value: float) -> LayerKV:
    """Build one cache layer with distinct key and value widths."""
    key = torch.full((1, 2, tokens, 3), value)
    val = torch.full((1, 2, tokens, 4), value + 0.5)
    return key, val


def cache() -> FixedSlotKVCache:
    """Build a cache with rotating control and history regions."""
    return FixedSlotKVCache(
        [layer(5, 1.0), layer(5, 2.0)],
        capacity=11,
        regions=[
            SlotRegion(name="control", start=0, slots=2, slot_tokens=2),
            SlotRegion(
                name="history",
                start=5,
                slots=2,
                slot_tokens=3,
                extends_length=True,
            ),
        ],
    )


def test_cache_can_use_caller_owned_storage() -> None:
    """Embed fixed-slot storage in an allocation owned by an adapter."""
    prefilled = [layer(5, 1.0), layer(5, 2.0)]
    storage = [layer(11, -9.0), layer(11, -9.0)]
    memory = FixedSlotKVCache(
        prefilled,
        capacity=11,
        regions=[
            SlotRegion(
                name="history",
                start=5,
                slots=2,
                slot_tokens=3,
                extends_length=True,
            )
        ],
        storage=storage,
    )

    assert [
        (key.data_ptr(), value.data_ptr())
        for key, value in zip(memory._k, memory._v, strict=True)
    ] == [(key.data_ptr(), value.data_ptr()) for key, value in storage]
    for (key, value), (prefill_key, prefill_value) in zip(
        memory.layers(), prefilled, strict=True
    ):
        assert torch.equal(key, prefill_key)
        assert torch.equal(value, prefill_value)
    for key, value in storage:
        assert torch.count_nonzero(key[:, :, 5:]) == 0
        assert torch.count_nonzero(value[:, :, 5:]) == 0


def test_invalid_caller_storage_does_not_mutate_any_layer() -> None:
    """Validate every supplied buffer before clearing or copying any of them."""
    prefilled = [layer(5, 1.0), layer(5, 2.0)]
    storage = [layer(11, -9.0), layer(11, -8.0)]
    storage[1] = (torch.zeros(2, 2, 11, 3), storage[1][1])
    before = [(key.clone(), value.clone()) for key, value in storage]

    with pytest.raises(ValueError, match="storage layer 1 key dimension 0"):
        FixedSlotKVCache(
            prefilled,
            capacity=11,
            regions=[],
            storage=storage,
        )

    for (key, value), (old_key, old_value) in zip(storage, before, strict=True):
        assert torch.equal(key, old_key)
        assert torch.equal(value, old_value)


def chunks(tokens: int, value: float) -> list[LayerKV]:
    """Build a two-layer cache write."""
    return [layer(tokens, value), layer(tokens, value + 1.0)]


def test_named_regions_rotate_independently_without_reallocation() -> None:
    memory = cache()
    pointers = [
        (key.data_ptr(), value.data_ptr())
        for key, value in zip(memory._k, memory._v, strict=True)
    ]

    assert memory.write("control", chunks(2, 10.0)) == 0
    assert memory.write("history", chunks(3, 20.0)) == 0
    assert memory.write("control", chunks(2, 30.0)) == 1
    assert memory.write("history", chunks(3, 40.0)) == 1
    assert memory.write("control", chunks(2, 50.0)) == 0
    assert memory.write("history", chunks(3, 60.0)) == 0

    key, value = memory.layers()[0]
    assert memory.length == memory.capacity == 11
    assert torch.equal(key[:, :, 0:2], chunks(2, 50.0)[0][0])
    assert torch.equal(key[:, :, 2:4], chunks(2, 30.0)[0][0])
    assert torch.equal(key[:, :, 5:8], chunks(3, 60.0)[0][0])
    assert torch.equal(key[:, :, 8:11], chunks(3, 40.0)[0][0])
    assert torch.equal(value[:, :, 5:8], chunks(3, 60.0)[0][1])
    assert [
        (stored_key.data_ptr(), stored_value.data_ptr())
        for stored_key, stored_value in zip(memory._k, memory._v, strict=True)
    ] == pointers


def test_short_write_clears_the_tail_of_a_reused_physical_slot() -> None:
    memory = cache()

    memory.write("history", chunks(3, 7.0))
    memory.write("history", chunks(3, 8.0))
    memory.write("history", chunks(1, 9.0))

    key, value = memory.layers()[0]
    assert memory.length == memory.capacity
    assert torch.equal(key[:, :, 5:6], chunks(1, 9.0)[0][0])
    assert torch.equal(value[:, :, 5:6], chunks(1, 9.0)[0][1])
    assert torch.count_nonzero(key[:, :, 6:8]) == 0
    assert torch.count_nonzero(value[:, :, 6:8]) == 0
    assert torch.equal(key[:, :, 8:11], chunks(3, 8.0)[0][0])
    assert torch.equal(value[:, :, 8:11], chunks(3, 8.0)[0][1])


def test_reset_restores_prefill_and_slot_positions_without_reallocation() -> None:
    memory = cache()
    pointers = [
        (key.data_ptr(), value.data_ptr())
        for key, value in zip(memory._k, memory._v, strict=True)
    ]
    memory.write("control", chunks(2, 10.0))
    memory.write("history", chunks(3, 20.0))

    replacement = [layer(5, 70.0), layer(5, 80.0)]
    memory.reset(replacement)

    assert memory.length == 5
    key, value = memory.layers()[0]
    assert torch.equal(key, replacement[0][0])
    assert torch.equal(value, replacement[0][1])
    assert memory.write("control", chunks(2, 90.0)) == 0
    assert memory.write("history", chunks(3, 100.0)) == 0
    assert [
        (stored_key.data_ptr(), stored_value.data_ptr())
        for stored_key, stored_value in zip(memory._k, memory._v, strict=True)
    ] == pointers


def test_reset_clears_stale_suffix_before_a_later_region_extends_length() -> None:
    memory = FixedSlotKVCache(
        [layer(1, 1.0)],
        capacity=5,
        regions=[
            SlotRegion(
                name="first", start=1, slots=1, slot_tokens=2, extends_length=True
            ),
            SlotRegion(
                name="second", start=3, slots=1, slot_tokens=2, extends_length=True
            ),
        ],
    )
    memory.write("first", [layer(2, 10.0)])
    memory.write("second", [layer(2, 20.0)])

    memory.reset([layer(1, 30.0)])
    memory.write("second", [layer(2, 40.0)])

    key, value = memory.layers()[0]
    assert memory.length == memory.capacity
    assert torch.count_nonzero(key[:, :, 1:3]) == 0
    assert torch.count_nonzero(value[:, :, 1:3]) == 0
    assert torch.equal(key[:, :, 3:5], layer(2, 40.0)[0])
    assert torch.equal(value[:, :, 3:5], layer(2, 40.0)[1])


def test_invalid_later_write_layer_does_not_modify_cache() -> None:
    memory = cache()
    before = [
        (key.clone(), value.clone())
        for key, value in zip(memory._k, memory._v, strict=True)
    ]
    malformed = chunks(2, 10.0)
    malformed[1] = (torch.zeros(2, 2, 2, 3), malformed[1][1])

    with pytest.raises(ValueError, match="layer 1 key dimension 0"):
        memory.write("control", malformed)

    assert memory.length == 5
    assert memory._next_slot["control"] == 0
    for (key, value), (old_key, old_value) in zip(
        zip(memory._k, memory._v, strict=True), before, strict=True
    ):
        assert torch.equal(key, old_key)
        assert torch.equal(value, old_value)


def test_invalid_later_reset_layer_does_not_modify_cache() -> None:
    memory = cache()
    memory.write("history", chunks(3, 10.0))
    before = [
        (key.clone(), value.clone())
        for key, value in zip(memory._k, memory._v, strict=True)
    ]
    malformed = [layer(5, 70.0), layer(5, 80.0)]
    malformed[1] = (torch.zeros(2, 2, 5, 3), malformed[1][1])

    with pytest.raises(ValueError, match="prefill layer 1.*dimension 0"):
        memory.reset(malformed)

    assert memory.length == 8
    assert memory._next_slot["history"] == 1
    for (key, value), (old_key, old_value) in zip(
        zip(memory._k, memory._v, strict=True), before, strict=True
    ):
        assert torch.equal(key, old_key)
        assert torch.equal(value, old_value)


def test_reset_rejects_an_incompatible_prefill_shape() -> None:
    """Validate replacement tensor geometry before writing cache storage."""
    memory = FixedSlotKVCache(
        [layer(5, 1.0)],
        capacity=5,
        regions=[],
    )

    with pytest.raises(ValueError, match="dimension 0"):
        memory.reset([(torch.zeros(2, 2, 5, 3), torch.zeros(2, 2, 5, 4))])


def test_cache_rejects_invalid_regions_and_writes() -> None:
    prefilled = [layer(5, 1.0)]

    with pytest.raises(ValueError, match="overlap"):
        FixedSlotKVCache(
            prefilled,
            capacity=10,
            regions=[
                SlotRegion(name="first", start=0, slots=2, slot_tokens=2),
                SlotRegion(name="second", start=3, slots=1, slot_tokens=2),
            ],
        )
    with pytest.raises(ValueError, match="past the cache capacity"):
        FixedSlotKVCache(
            prefilled,
            capacity=10,
            regions=[
                SlotRegion(
                    name="history",
                    start=5,
                    slots=2,
                    slot_tokens=3,
                    extends_length=True,
                )
            ],
        )

    memory = FixedSlotKVCache(
        prefilled,
        capacity=5,
        regions=[SlotRegion(name="history", start=5, slots=0, slot_tokens=0)],
    )
    with pytest.raises(ValueError, match="no reusable slots"):
        memory.write("history", [layer(1, 2.0)])
    with pytest.raises(ValueError, match="unknown slot region"):
        memory.write("missing", [layer(1, 2.0)])


def test_cache_rejects_chunks_larger_than_a_slot() -> None:
    memory = cache()

    with pytest.raises(ValueError, match="do not fit"):
        memory.write("history", chunks(4, 1.0))


def token_window() -> TokenWindow:
    """Build a small token window that wraps after two writes."""
    return TokenWindow(
        num_views=2,
        frames=4,
        spatial=2,
        width=3,
        device="cpu",
        dtype=torch.float32,
    )


def test_token_window_reads_across_a_wrap_and_rejects_stale_frames() -> None:
    window = token_window()
    first = torch.randn(2, 2, 2, 3)
    second = torch.randn(2, 2, 2, 3)
    third = torch.randn(2, 2, 2, 3)
    window.append(0, 2, first)
    window.append(2, 4, second)
    window.append(4, 6, third)

    assert torch.equal(window.read(4, 6), third)
    assert torch.equal(window.read(2, 4), second)
    assert torch.equal(window.read(2, 6), torch.cat([second, third], dim=1))
    with pytest.raises(ValueError, match="reaches back past"):
        window.read(0, 2)


def test_token_window_validates_order_size_and_shape() -> None:
    window = token_window()
    window.append(0, 2, torch.zeros(2, 2, 2, 3))

    with pytest.raises(ValueError, match="do not carry on"):
        window.append(3, 4, torch.zeros(2, 1, 2, 3))
    with pytest.raises(ValueError, match="do not fit"):
        token_window().append(0, 5, torch.zeros(2, 5, 2, 3))
    with pytest.raises(ValueError, match="expected"):
        token_window().append(0, 2, torch.zeros(1, 2, 2, 3))


def test_token_window_reset_preserves_storage() -> None:
    window = token_window()
    pointer = window._store.data_ptr()
    window.append(0, 2, torch.ones(2, 2, 2, 3))

    window.reset()

    assert window.start == window.end == 0
    assert window._store.data_ptr() == pointer
    replacement = torch.full((2, 2, 2, 3), 7.0)
    window.append(0, 2, replacement)
    assert torch.equal(window.read(0, 2), replacement)
