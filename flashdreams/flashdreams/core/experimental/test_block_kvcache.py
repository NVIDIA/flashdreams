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

"""Unit tests for BlockKVCache."""

import pytest
import torch

from flashdreams.core.experimental.kvcache import BlockKVCache

pytestmark = pytest.mark.ci_cpu


class _NaiveKVCache:
    """Baseline: full sequence via torch.cat; view = [sink | last window] or full. Shape [B, S, H, D]."""

    def __init__(self, sink_size: int, window_size: int) -> None:
        self.sink_size = sink_size
        self.window_size = window_size
        self._total_k: torch.Tensor | None = None
        self._total_v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self._total_k is None or self._total_v is None:
            self._total_k = k
            self._total_v = v
        else:
            self._total_k = torch.cat([self._total_k, k], dim=1)
            self._total_v = torch.cat([self._total_v, v], dim=1)

    def ovewrite_rightmost(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert self._total_k is not None
        assert self._total_v is not None
        length = k.shape[1]
        self._total_k[:, -length:] = k
        self._total_v[:, -length:] = v

    def cached_k(self) -> torch.Tensor:
        assert self._total_k is not None
        S = self._total_k.shape[1]
        if S >= self.sink_size + self.window_size:
            return torch.cat(
                [
                    self._total_k[:, : self.sink_size],
                    self._total_k[:, -self.window_size :],
                ],
                dim=1,
            )
        return self._total_k

    def cached_v(self) -> torch.Tensor:
        assert self._total_v is not None
        S = self._total_v.shape[1]
        if S >= self.sink_size + self.window_size:
            return torch.cat(
                [
                    self._total_v[:, : self.sink_size],
                    self._total_v[:, -self.window_size :],
                ],
                dim=1,
            )
        return self._total_v


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float32


@pytest.mark.parametrize("sink_size,window_size", [(0, 8), (0, 24), (3, 5), (3, 21)])
def test_block_kvcache_matches_baseline(
    device: torch.device,
    dtype: torch.dtype,
    sink_size: int,
    window_size: int,
) -> None:
    """Compare cache API with baseline."""
    batch, n_heads = 2, 4
    dim_k, dim_v = 8, 16
    chunk_size = 8
    buffer_size = sink_size + window_size

    k_shape = (batch, buffer_size, n_heads, dim_k)
    v_shape = (batch, buffer_size, n_heads, dim_v)

    cache = BlockKVCache(
        k_shape=k_shape,
        v_shape=v_shape,
        seq_dim=1,
        window_size=window_size,
        sink_size=sink_size,
        device=device,
        dtype=dtype,
    )

    naive = _NaiveKVCache(sink_size, window_size)
    num_chunks = 8

    for chunk_idx in range(num_chunks):
        new_k = torch.randn(
            batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
        )
        new_v = torch.randn(
            batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
        )

        naive.update(new_k, new_v)
        k_baseline = naive.cached_k()
        v_baseline = naive.cached_v()

        # test basic API
        cache.update(new_k, new_v, chunk_idx * chunk_size)
        k_api = cache.cached_k()
        v_api = cache.cached_v()
        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)

        # now test that passing in the same index again, should only update the cache at the same positions
        new_k = torch.randn(
            batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
        )
        new_v = torch.randn(
            batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
        )

        naive.ovewrite_rightmost(new_k, new_v)
        k_baseline = naive.cached_k()
        v_baseline = naive.cached_v()

        cache.update(new_k, new_v, chunk_idx * chunk_size)
        k_api = cache.cached_k()
        v_api = cache.cached_v()
        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)
