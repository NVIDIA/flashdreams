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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
)

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch._dynamo.callback import CallbackArgs


class _NaiveKVCache:
    """Naive [sink | rolling window] cache for test parity. Shape [B, S, H, D]."""

    def __init__(
        self,
        *,
        window_size: int,
        chunk_size: int,
        sink_size: int = 0,
    ) -> None:
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.sink_size = sink_size
        self.total_size = self.sink_size + self.window_size
        self._cache_k: torch.Tensor | None = None
        self._cache_v: torch.Tensor | None = None
        self._prev_chunk_idx = -1

    def update(self, chunk_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        assert chunk_idx in (self._prev_chunk_idx, self._prev_chunk_idx + 1)
        if self._cache_k is None or self._cache_v is None:
            assert chunk_idx == 0
            self._cache_k = k.clone()
            self._cache_v = v.clone()
            self._prev_chunk_idx = 0
            return

        length = k.shape[1]
        if chunk_idx == self._prev_chunk_idx:
            overlaps_sink = self.sink_size > 0 and chunk_idx * length < self.sink_size
            if self._cache_k.shape[1] == self.total_size and not overlaps_sink:
                if length <= self.window_size:
                    self._cache_k[:, -length:] = k
                    self._cache_v[:, -length:] = v
                else:
                    self._cache_k = torch.cat(
                        [
                            self._cache_k[:, : self.sink_size],
                            k[:, -self.window_size :],
                        ],
                        dim=1,
                    )
                    self._cache_v = torch.cat(
                        [
                            self._cache_v[:, : self.sink_size],
                            v[:, -self.window_size :],
                        ],
                        dim=1,
                    )
            else:
                self._cache_k[:, -length:] = k
                self._cache_v[:, -length:] = v
            return

        sink_k = self._cache_k[:, : self.sink_size]
        sink_v = self._cache_v[:, : self.sink_size]
        window_k = torch.cat([self._cache_k[:, self.sink_size :], k], dim=1)[
            :, -self.window_size :
        ]
        window_v = torch.cat([self._cache_v[:, self.sink_size :], v], dim=1)[
            :, -self.window_size :
        ]
        self._cache_k = torch.cat([sink_k, window_k], dim=1)
        self._cache_v = torch.cat([sink_v, window_v], dim=1)
        self._prev_chunk_idx += 1

    def cached_k(self) -> torch.Tensor:
        assert self._cache_k is not None
        return self._cache_k

    def cached_v(self) -> torch.Tensor:
        assert self._cache_v is not None
        return self._cache_v


class _FakeProcessGroup:
    def __init__(self, world_size: int, rank: int) -> None:
        self._world_size = world_size
        self._rank = rank

    def size(self) -> int:
        return self._world_size

    def rank(self) -> int:
        return self._rank


class _FakeDeviceMesh:
    def __init__(self, world_size: int) -> None:
        self._world_size = world_size

    def size(self) -> int:
        return self._world_size


@pytest.mark.ci_cpu
def test_reset_preserves_storage_and_restores_empty_bookkeeping() -> None:
    cache = BlockKVCache(
        k_shape=(1, 4, 1, 2),
        v_shape=(1, 4, 1, 2),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    k_ptr = cache._k.data_ptr()
    v_ptr = cache._v.data_ptr()
    cache.before_update(0)
    cache.update(torch.ones((1, 2, 1, 2)), torch.ones((1, 2, 1, 2)))
    cache.after_update(0)

    cache.reset()

    assert cache._k.data_ptr() == k_ptr
    assert cache._v.data_ptr() == v_ptr
    assert cache._prev_chunk_idx == -1
    assert cache._curr_chunk_idx is None
    assert cache._n_cached == 0

    cache.before_update(0)
    assert cache.size == 2


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float32


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    "sink_size,window_size", [(0, 8), (0, 24), (3, 5), (3, 21), (1, 16)]
)
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
    buffer_size = window_size + sink_size

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

    naive = _NaiveKVCache(
        window_size=window_size,
        chunk_size=chunk_size,
        sink_size=sink_size,
    )
    num_chunks = 8

    for chunk_idx in range(num_chunks):
        new_k = torch.randn(
            batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
        )
        new_v = torch.randn(
            batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
        )

        naive.update(chunk_idx, new_k, new_v)
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

        naive.update(chunk_idx, new_k, new_v)
        k_baseline = naive.cached_k()
        v_baseline = naive.cached_v()

        cache.before_update(chunk_idx)
        cache.update(new_k, new_v)
        k_api = cache.cached_k()
        v_api = cache.cached_v()
        cache.after_update(chunk_idx)
        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)


@pytest.mark.ci_cpu
def test_block_kvcache_size_and_write_end_track_current_update() -> None:
    """Attention can slice RoPE using the cache target write region."""
    cache = BlockKVCache(
        k_shape=(1, 6, 1, 1),
        v_shape=(1, 6, 1, 1),
        seq_dim=1,
        chunk_size=2,
        window_size=6,
        device="cpu",
        dtype=torch.float32,
    )
    assert cache.size == 0

    for chunk_idx, expected_end in [(0, 2), (1, 4), (2, 6), (3, 6)]:
        cache.before_update(chunk_idx)
        assert cache.write_end == expected_end
        assert cache.size == expected_end
        values = torch.full((1, 2, 1, 1), float(chunk_idx))
        cache.update(values, values)
        cache.after_update(chunk_idx)
        assert cache.size == expected_end

    cache.before_update(3)
    assert cache.write_end == 6
    assert cache.size == 6
    values = torch.full((1, 2, 1, 1), 30.0)
    cache.update(values, values)
    cache.after_update(3)
    assert cache.size == 6


@pytest.mark.ci_cpu
def test_standard_rope_indexing_changes_with_ar_index() -> None:
    """Standard RoPE follows unbounded AR time positions."""
    rope = RotaryPositionEmbedding3D(
        head_dim=12,
        len_t=3,
        len_h=2,
        len_w=2,
        interleaved=True,
        device=torch.device("cpu"),
    )
    rope_freqs_0 = rope.shift_t(0)
    rope_freqs_1 = rope.shift_t(1)
    assert rope_freqs_0.shape == rope_freqs_1.shape == (12, 1, 1, 12)
    assert not torch.equal(rope_freqs_0, rope_freqs_1)


@pytest.mark.ci_cpu
def test_kvcache_relative_rope_cp_freqs_match_cache_chunks() -> None:
    """CP cache-relative freqs must follow the chunk-sharded cache layout."""
    full_rope = KVCacheRelativeRotaryPositionEmbedding3D(
        head_dim=12,
        len_t=3,
        len_h=2,
        len_w=2,
        sink_size_t=3,
        window_size_t=3,
        interleaved=True,
        device=torch.device("cpu"),
    )
    freqs_full = full_rope.shift_t(0)

    chunk_tokens = 3 * 2 * 2
    world_size = 2
    for rank in range(world_size):
        cp_rope = KVCacheRelativeRotaryPositionEmbedding3D(
            head_dim=12,
            len_t=3,
            len_h=2,
            len_w=2,
            sink_size_t=3,
            window_size_t=3,
            interleaved=True,
            device=torch.device("cpu"),
        )
        cp_rope_any = cast(Any, cp_rope)
        cp_rope_any.cp_group = _FakeProcessGroup(world_size=world_size, rank=rank)
        cp_rope_any.device_mesh = _FakeDeviceMesh(world_size=world_size)

        freqs_rank = cp_rope.shift_t(0)
        expected = torch.cat(
            [
                freqs_full[0:chunk_tokens].chunk(world_size, dim=0)[rank],
                freqs_full[chunk_tokens : 2 * chunk_tokens].chunk(world_size, dim=0)[
                    rank
                ],
            ],
            dim=0,
        )
        torch.testing.assert_close(freqs_rank, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for cudagraph test"
)
@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "sink_size,window_size", [(0, 8), (0, 24), (3, 5), (3, 21), (1, 16)]
)
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
    buffer_size = window_size + sink_size

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

    naive = _NaiveKVCache(
        window_size=window_size,
        chunk_size=chunk_size,
        sink_size=sink_size,
    )
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

        naive.update(chunk_idx, new_k, new_v)
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

        naive.update(chunk_idx, new_k, new_v)
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for cudagraph test"
)
@pytest.mark.ci_gpu
def test_block_kvcache_compile_cudagraph_matches_baseline(
    dtype: torch.dtype,
) -> None:
    """BlockKVCache with torch.compile and CUDA graph should match baseline.

    To print out the torch.compile events run with

    uv run --group test pytest \
        flashdreams/tests/test_kvcache.py::test_block_kvcache_compile_cudagraph_matches_baseline \
        -vv -s -o log_cli=true --log-cli-level=INFO
    """
    import torch._dynamo as dynamo

    device = torch.device("cuda")
    batch, n_heads = 2, 4
    dim_k, dim_v = 8, 16
    chunk_size = 8
    sink_size, window_size = 0, 24
    buffer_size = window_size + sink_size

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
    naive = _NaiveKVCache(
        window_size=window_size,
        chunk_size=chunk_size,
        sink_size=sink_size,
    )

    steady_k = torch.empty(
        batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
    )
    steady_v = torch.empty(
        batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
    )
    graph: torch.cuda.CUDAGraph | None = None
    graph_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
    warmup_iters = 3
    num_chunks = 8

    compile_events: list[tuple[int | None, str, str, str]] = []
    cudagraph_capture_events: list[tuple[int | None, str]] = []
    current_chunk_idx: int | None = None
    current_phase = ""

    def on_compile_start(args: CallbackArgs) -> None:
        compile_events.append(
            (
                current_chunk_idx,
                current_phase,
                str(args.callback_trigger),
                args.compile_id,
            )
        )
        _LOGGER.info(
            "torch.compile triggered at chunk_idx=%s phase=%s trigger=%s compile_id=%s",
            current_chunk_idx,
            current_phase,
            args.callback_trigger,
            args.compile_id,
        )

    def cache_step(
        active_cache: BlockKVCache, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_cache.update(k, v)
        return active_cache.cached_k(), active_cache.cached_v()

    compiled_cache_step = torch.compile(cache_step, mode="max-autotune-no-cudagraphs")

    def run_cache_step(
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal graph, graph_outputs

        if cache.is_steady_state():
            steady_k.copy_(new_k)
            steady_v.copy_(new_v)
            if graph is None:
                _LOGGER.info(
                    "warming compiled steady-state path before CUDA graph capture at chunk_idx=%s phase=%s",
                    current_chunk_idx,
                    current_phase,
                )
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup_iters):
                        compiled_cache_step(cache, steady_k, steady_v)
                torch.cuda.current_stream().wait_stream(s)
                compile_event_count_before_capture = len(compile_events)
                cudagraph_capture_events.append((current_chunk_idx, current_phase))
                _LOGGER.info(
                    "capturing CUDA graph at chunk_idx=%s phase=%s",
                    current_chunk_idx,
                    current_phase,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_outputs = compiled_cache_step(cache, steady_k, steady_v)
                assert len(compile_events) == compile_event_count_before_capture
            else:
                graph.replay()
            assert graph_outputs is not None
            return graph_outputs

        return compiled_cache_step(cache, new_k, new_v)

    dynamo.on_compile_start(on_compile_start)
    try:
        for chunk_idx in range(num_chunks):
            _LOGGER.info("Running chunk %s", chunk_idx)
            current_chunk_idx = chunk_idx
            cache.before_update(chunk_idx)
            try:
                current_phase = "append"
                new_k = torch.randn(
                    batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
                )
                new_v = torch.randn(
                    batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
                )

                naive.update(chunk_idx, new_k, new_v)
                k_baseline = naive.cached_k()
                v_baseline = naive.cached_v()

                k_api, v_api = run_cache_step(new_k, new_v)
                torch.testing.assert_close(k_api, k_baseline)
                torch.testing.assert_close(v_api, v_baseline)

                current_phase = "overwrite"
                new_k = torch.randn(
                    batch, chunk_size, n_heads, dim_k, device=device, dtype=dtype
                )
                new_v = torch.randn(
                    batch, chunk_size, n_heads, dim_v, device=device, dtype=dtype
                )

                naive.update(chunk_idx, new_k, new_v)
                k_baseline = naive.cached_k()
                v_baseline = naive.cached_v()

                k_api, v_api = run_cache_step(new_k, new_v)
                torch.testing.assert_close(k_api, k_baseline)
                torch.testing.assert_close(v_api, v_baseline)
            finally:
                cache.after_update(chunk_idx)
                current_chunk_idx = None
                current_phase = ""
    finally:
        dynamo.callback_handler.remove_start_callback(on_compile_start)

    assert graph is not None
    assert cudagraph_capture_events == [(3, "append")]
    assert any(chunk_idx == 0 for chunk_idx, *_ in compile_events)
    _LOGGER.info(
        "torch.compile events: %s; CUDA graph capture events: %s",
        compile_events,
        cudagraph_capture_events,
    )
