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

"""
Unit tests for BlockKVCache.
"""

import pytest
import torch

from flashdreams.core.attention.kvcache import BlockKVCache


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
        chunk_size=chunk_size,
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
        cache.before_update(chunk_idx)
        cache.update(new_k, new_v)
        k_api = cache.cached_k()
        v_api = cache.cached_v()
        cache.after_update(chunk_idx)
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

        cache.before_update(chunk_idx)
        cache.update(new_k, new_v)
        k_api = cache.cached_k()
        v_api = cache.cached_v()
        cache.after_update(chunk_idx)
        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for cudagraph test"
)
@pytest.mark.ci_gpu
@pytest.mark.parametrize("sink_size,window_size", [(0, 8), (0, 24), (3, 5), (3, 21)])
def test_block_kvcache_cudagraph_matches_baseline(
    dtype: torch.dtype,
    sink_size: int,
    window_size: int,
) -> None:
    """BlockKVCache with CUDA graph (steady-state path) should match baseline."""
    device = torch.device("cuda")
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
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        device=device,
        dtype=dtype,
    )

    naive = _NaiveKVCache(sink_size, window_size)
    num_chunks = 8

    # Static buffers for CUDA graph capture/replay (steady-state path).
    steady_k = torch.empty(
        batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
    )
    steady_v = torch.empty(
        batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
    )
    graph: torch.cuda.CUDAGraph | None = None
    warmup_iters = 3

    def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cache.update(k, v)
        k_output = cache.cached_k()
        v_output = cache.cached_v()
        return k_output, v_output

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

        cache.before_update(chunk_idx)
        if cache.is_steady_state():
            steady_k.copy_(new_k)
            steady_v.copy_(new_v)
            if graph is None:
                # Capture graph after warmup.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup_iters):
                        fn(steady_k, steady_v)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    k_api, v_api = fn(steady_k, steady_v)
            else:
                graph.replay()
        else:
            k_api, v_api = fn(new_k, new_v)
        cache.after_update(chunk_idx)

        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)

        # Overwrite same chunk (same as baseline test)
        new_k = torch.randn(
            batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
        )
        new_v = torch.randn(
            batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
        )

        naive.ovewrite_rightmost(new_k, new_v)
        k_baseline = naive.cached_k()
        v_baseline = naive.cached_v()

        cache.before_update(chunk_idx)
        if graph is not None:
            assert cache.is_steady_state()
            steady_k.copy_(new_k)
            steady_v.copy_(new_v)
            graph.replay()
        else:
            k_api, v_api = fn(new_k, new_v)
        cache.after_update(chunk_idx)

        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)

    # make sure the graph is captured.
    assert graph is not None
