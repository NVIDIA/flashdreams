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

"""CPU lifecycle tests for FP8 block K/V storage."""

from __future__ import annotations

import pytest
import torch

from flashdreams.core.attention import BlockKVCache

pytestmark = pytest.mark.ci_cpu


def test_fp8_block_kv_cache_converts_native_inputs() -> None:
    """Store native-precision inputs in fixed E4M3 cache tensors."""
    key = torch.tensor([[[[1.0, -2.0]], [[3.0, -4.0]]]])
    value = torch.tensor([[[[5.0, -6.0]], [[7.0, -8.0]]]])

    cache = BlockKVCache(
        k_shape=tuple(key.shape),
        v_shape=tuple(value.shape),
        seq_dim=1,
        chunk_size=key.shape[1],
        window_size=key.shape[1],
        device="cpu",
        dtype=torch.float8_e4m3fn,
    )
    cache.before_update(0)
    cache.update(key, value)

    assert cache._k.dtype is torch.float8_e4m3fn
    assert cache._v.dtype is torch.float8_e4m3fn
    torch.testing.assert_close(cache.cached_k().float(), key)
    torch.testing.assert_close(cache.cached_v().float(), value)
    cache.after_update(0)


def test_fp8_block_kv_cache_preserves_rolling_lifecycle() -> None:
    """Preserve filling, rolling, overwrite, and reset behavior in FP8."""
    cache = BlockKVCache(
        k_shape=(1, 4, 1, 1),
        v_shape=(1, 4, 1, 1),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        device="cpu",
        dtype=torch.float8_e4m3fn,
    )

    for chunk_idx in range(3):
        values = torch.tensor([[[[2.0 * chunk_idx]], [[2.0 * chunk_idx + 1.0]]]])
        cache.before_update(chunk_idx)
        cache.update(values, -values)
        cache.after_update(chunk_idx)

    cache.before_update(2)
    replacement = torch.tensor([[[[8.0]], [[9.0]]]])
    cache.update(replacement, -replacement)

    torch.testing.assert_close(
        cache.cached_k().float(),
        torch.tensor([[[[2.0]], [[3.0]], [[8.0]], [[9.0]]]]),
    )
    torch.testing.assert_close(
        cache.cached_v().float(),
        torch.tensor([[[[-2.0]], [[-3.0]], [[-8.0]], [[-9.0]]]]),
    )
    cache.after_update(2)

    key_pointer = cache._k.data_ptr()
    cache.reset()

    assert cache._k.data_ptr() == key_pointer
    assert cache.size == 0
