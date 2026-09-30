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

"""Stable contiguous K/V storage for attention over segmented inputs."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

KVPair = tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class KVSegmentLayout:
    """Token offsets for ``[static | current | memory]`` K/V storage."""

    static_tokens: int
    current_tokens: int
    memory_tokens: int

    def __post_init__(self) -> None:
        if self.static_tokens < 0 or self.memory_tokens < 0:
            raise ValueError("static and memory token capacities must be non-negative")
        if self.current_tokens <= 0:
            raise ValueError("current token capacity must be positive")

    @property
    def current_start(self) -> int:
        return self.static_tokens

    @property
    def memory_start(self) -> int:
        return self.static_tokens + self.current_tokens

    @property
    def total_tokens(self) -> int:
        return self.memory_start + self.memory_tokens


def _validate_pair(pair: KVPair, name: str) -> None:
    key, value = pair
    if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        raise ValueError(
            f"{name} K/V must have matching [B, H, T, D] shapes; "
            f"got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    if key.dtype is not value.dtype or key.device != value.device:
        raise ValueError(f"{name} K/V must have matching dtype and device")


def _validate_arena_storage(arena: KVPair) -> None:
    """Require the storage contract that makes an arena safe and useful."""
    _validate_pair(arena, "arena")
    key, value = arena
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("arena K/V must be contiguous")
    if key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
        raise ValueError("arena K/V must use distinct storage")


def validate_kv_arena_segments(
    arena: KVPair,
    static: KVPair,
    memory: KVPair,
    layout: KVSegmentLayout,
) -> None:
    """Verify persistent inputs are the arena segments named by ``layout``."""
    _validate_arena_storage(arena)
    _validate_pair(static, "static")
    _validate_pair(memory, "memory")

    for name, pair, start, tokens in (
        ("static", static, 0, layout.static_tokens),
        ("memory", memory, layout.memory_start, layout.memory_tokens),
    ):
        for arena_tensor, segment in zip(arena, pair, strict=True):
            expected_shape = (
                *arena_tensor.shape[:2],
                tokens,
                arena_tensor.shape[3],
            )
            expected_offset = (
                arena_tensor.storage_offset() + start * arena_tensor.stride(2)
            )
            if tuple(segment.shape) != expected_shape:
                raise ValueError(
                    f"{name} arena segment must have shape {expected_shape}, "
                    f"got {tuple(segment.shape)}"
                )
            if (
                segment.untyped_storage().data_ptr()
                != arena_tensor.untyped_storage().data_ptr()
                or segment.storage_offset() != expected_offset
                or segment.stride() != arena_tensor.stride()
            ):
                raise ValueError(
                    f"{name} K/V must alias its declared attention arena segment"
                )


def allocate_kv_arena(
    static: KVPair,
    memory: KVPair,
    *,
    current_tokens: int,
    memory_tokens: int,
) -> tuple[KVPair, KVSegmentLayout]:
    """Allocate ``[static | current | memory]`` and seed its persistent regions."""
    _validate_pair(static, "static")
    _validate_pair(memory, "memory")
    static_key, static_value = static
    memory_key, memory_value = memory
    if (
        static_key.shape[:2] + static_key.shape[3:]
        != memory_key.shape[:2] + memory_key.shape[3:]
        or static_key.dtype is not memory_key.dtype
        or static_key.device != memory_key.device
    ):
        raise ValueError(
            "static and memory K/V batch, heads, width, dtype, and device must match"
        )
    if memory_key.shape[2] > memory_tokens:
        raise ValueError(
            f"memory contains {memory_key.shape[2]} tokens past capacity {memory_tokens}"
        )

    layout = KVSegmentLayout(static_key.shape[2], current_tokens, memory_tokens)
    shape = (*static_key.shape[:2], layout.total_tokens, static_key.shape[3])
    arena_key = static_key.new_zeros(shape)
    arena_value = static_value.new_zeros(shape)
    arena_key[:, :, : layout.static_tokens].copy_(static_key)
    arena_value[:, :, : layout.static_tokens].copy_(static_value)
    memory_end = layout.memory_start + memory_key.shape[2]
    arena_key[:, :, layout.memory_start : memory_end].copy_(memory_key)
    arena_value[:, :, layout.memory_start : memory_end].copy_(memory_value)
    return (arena_key, arena_value), layout


def stage_current_kv_(
    arena: KVPair,
    current: KVPair,
    layout: KVSegmentLayout,
) -> KVPair:
    """Overwrite only the current segment and return the contiguous arena."""
    _validate_arena_storage(arena)
    _validate_pair(current, "current")
    arena_key, arena_value = arena
    current_key, current_value = current
    expected = (*arena_key.shape[:2], layout.current_tokens, arena_key.shape[3])
    if tuple(arena_key.shape) != (
        *expected[:2],
        layout.total_tokens,
        expected[3],
    ):
        raise ValueError(
            f"arena shape {tuple(arena_key.shape)} does not match layout total "
            f"{layout.total_tokens}"
        )
    if tuple(current_key.shape) != expected:
        raise ValueError(
            f"current K/V must have shape {expected}, got {tuple(current_key.shape)}"
        )
    if (
        current_key.dtype is not arena_key.dtype
        or current_key.device != arena_key.device
    ):
        raise ValueError("current and arena K/V must have matching dtype and device")
    arena_storage = {
        arena_key.untyped_storage().data_ptr(),
        arena_value.untyped_storage().data_ptr(),
    }
    if (
        current_key.untyped_storage().data_ptr() in arena_storage
        or current_value.untyped_storage().data_ptr() in arena_storage
    ):
        raise ValueError("current K/V must not alias the destination arena")

    current_end = layout.current_start + layout.current_tokens
    arena_key[:, :, layout.current_start : current_end].copy_(current_key)
    arena_value[:, :, layout.current_start : current_end].copy_(current_value)
    return arena
