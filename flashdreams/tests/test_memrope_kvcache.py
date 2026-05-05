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

"""Unit tests for MemRoPEKVCache."""

import pytest
import torch

from flashdreams.core.attention.memrope_kvcache import MemRoPEKVCache


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float32


def _scalar_chunk(
    start: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.arange(
        start, start + chunk_size, device=device, dtype=dtype
    ).reshape(1, chunk_size, 1, 1)


def _new_cache(device: torch.device, dtype: torch.dtype) -> MemRoPEKVCache:
    return MemRoPEKVCache(
        k_shape=(1, 12, 1, 1),
        v_shape=(1, 12, 1, 1),
        seq_dim=1,
        chunk_size=3,
        window_size=9,
        sink_size=3,
        frame_size=1,
        recent_size=4,
        ema_alpha_long=0.01,
        ema_alpha_short=0.1,
        device=device,
        dtype=dtype,
    )


def _new_no_memory_cache(device: torch.device, dtype: torch.dtype) -> MemRoPEKVCache:
    return MemRoPEKVCache(
        k_shape=(1, 12, 1, 1),
        v_shape=(1, 12, 1, 1),
        seq_dim=1,
        chunk_size=3,
        window_size=9,
        sink_size=3,
        frame_size=1,
        recent_size=6,
        memory_frames=0,
        ema_alpha_long=0.01,
        ema_alpha_short=0.1,
        device=device,
        dtype=dtype,
    )


def _append_chunk(
    cache: MemRoPEKVCache,
    chunk_idx: int,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache.before_update(chunk_idx)
    cache.update(values, values + 1000)
    k = cache.cached_k().clone()
    v = cache.cached_v().clone()
    cache.after_update(chunk_idx)
    return k, v


def test_memrope_kvcache_compresses_to_ema_layout(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    cache = _new_cache(device, dtype)
    for chunk_idx in range(4):
        values = _scalar_chunk(chunk_idx * 3, 3, device, dtype)
        _append_chunk(cache, chunk_idx, values)

    values = _scalar_chunk(12, 3, device, dtype)
    k, v = _append_chunk(cache, 4, values)

    expected = torch.tensor(
        [0, 1, 2, 6, 6, 8, 9, 10, 11, 12, 13, 14],
        device=device,
        dtype=dtype,
    ).reshape(1, 12, 1, 1)
    torch.testing.assert_close(k, expected)
    torch.testing.assert_close(v, expected + 1000)
    torch.testing.assert_close(
        cache.cached_frame_indices(),
        torch.arange(12, device=device, dtype=torch.long),
    )
    torch.testing.assert_close(
        cache.query_frame_indices(),
        torch.arange(9, 12, device=device, dtype=torch.long),
    )


def test_memrope_kvcache_query_indices_grow_while_filling(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    cache = _new_cache(device, dtype)
    values = _scalar_chunk(0, 3, device, dtype)
    _append_chunk(cache, 0, values)

    torch.testing.assert_close(
        cache.cached_frame_indices(),
        torch.arange(3, device=device, dtype=torch.long),
    )
    torch.testing.assert_close(
        cache.query_frame_indices(),
        torch.arange(3, device=device, dtype=torch.long),
    )


def test_memrope_kvcache_same_chunk_overwrite_preserves_memory(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    cache = _new_cache(device, dtype)
    for chunk_idx in range(5):
        values = _scalar_chunk(chunk_idx * 3, 3, device, dtype)
        _append_chunk(cache, chunk_idx, values)

    overwrite = _scalar_chunk(120, 3, device, dtype)
    k, _ = _append_chunk(cache, 4, overwrite)

    expected = torch.tensor(
        [0, 1, 2, 6, 6, 8, 9, 10, 11, 120, 121, 122],
        device=device,
        dtype=dtype,
    ).reshape(1, 12, 1, 1)
    torch.testing.assert_close(k, expected)


def test_memrope_kvcache_updates_initialized_ema(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    cache = _new_cache(device, dtype)
    for chunk_idx in range(5):
        values = _scalar_chunk(chunk_idx * 3, 3, device, dtype)
        _append_chunk(cache, chunk_idx, values)

    values = _scalar_chunk(15, 3, device, dtype)
    k, _ = _append_chunk(cache, 5, values)

    expected = torch.tensor(
        [0, 1, 2, 6.03, 6.3, 11, 12, 13, 14, 15, 16, 17],
        device=device,
        dtype=dtype,
    ).reshape(1, 12, 1, 1)
    torch.testing.assert_close(k, expected)


def test_memrope_kvcache_zero_memory_keeps_sink_and_recent(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    cache = _new_no_memory_cache(device, dtype)
    for chunk_idx in range(4):
        values = _scalar_chunk(chunk_idx * 3, 3, device, dtype)
        _append_chunk(cache, chunk_idx, values)

    values = _scalar_chunk(12, 3, device, dtype)
    k, _ = _append_chunk(cache, 4, values)

    expected = torch.tensor(
        [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        device=device,
        dtype=dtype,
    ).reshape(1, 12, 1, 1)
    torch.testing.assert_close(k, expected)
