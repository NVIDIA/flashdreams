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
Unit tests for RollingBlockKVCache.
"""

from typing import Any, cast

import pytest
import torch

from flashdreams.core.attention.kvcache import (
    RollingBlockKVCache,
)
from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
)


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
            # Reuse (same chunk_idx): overwrite the rightmost ``length``
            # tokens in place. ``chunk_size <= window_size`` is enforced by
            # ``RollingBlockKVCache.__post_init__``, so the chunk always fits the
            # window and there is no partial-keep (``length > window_size``) case.
            self._cache_k[:, -length:] = k
            self._cache_v[:, -length:] = v
            return

        if self._cache_k.shape[1] == self.total_size:
            tail_k = self._cache_k[:, self.sink_size + self.chunk_size :]
            tail_v = self._cache_v[:, self.sink_size + self.chunk_size :]
            window_k = torch.cat([tail_k, k], dim=1)[:, -self.window_size :]
            window_v = torch.cat([tail_v, v], dim=1)[:, -self.window_size :]
            self._cache_k = torch.cat(
                [self._cache_k[:, : self.sink_size], window_k], dim=1
            )
            self._cache_v = torch.cat(
                [self._cache_v[:, : self.sink_size], window_v], dim=1
            )
        else:
            self._cache_k = torch.cat([self._cache_k, k], dim=1)
            self._cache_v = torch.cat([self._cache_v, v], dim=1)
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


def _new_cache(
    *,
    chunk_size: int,
    window_size: int,
    sink_size: int,
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RollingBlockKVCache:
    """Build a self-contained :class:`RollingBlockKVCache` from sizes (``seq_dim=1``)."""
    return RollingBlockKVCache(
        k_shape=k_shape,
        v_shape=v_shape,
        seq_dim=1,
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        device=device,
        dtype=dtype,
    )


def _cursor_cache(
    *, chunk_size: int = 2, window_size: int = 6, sink_size: int = 0
) -> RollingBlockKVCache:
    """A minimal-buffer cache for driving the inlined cursor in isolation."""
    total = sink_size + window_size
    return _new_cache(
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        k_shape=(1, total, 1, 1),
        v_shape=(1, total, 1, 1),
    )


def _cascade_before(caches: list[RollingBlockKVCache], chunk_idx: int) -> None:
    """Owner-style step (mirrors ``*TransformerCache.start``): advance + roll
    every cache's own cursor for ``chunk_idx``. The caches share geometry +
    chunk sequence, so they march in lock-step."""
    for c in caches:
        c.before_update(chunk_idx)


def _cascade_after(caches: list[RollingBlockKVCache], chunk_idx: int) -> None:
    """Owner-style finalize: ``after_update`` every cache's own cursor."""
    for c in caches:
        c.after_update(chunk_idx)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float32


@pytest.mark.ci_cpu
@pytest.mark.parametrize("sink_size,window_size", [(0, 8), (0, 24), (3, 21)])
def test_block_kvcache_branchless_api_matches_baseline(
    device: torch.device,
    dtype: torch.dtype,
    sink_size: int,
    window_size: int,
) -> None:
    """``update_at`` / ``cached_{k,v}_at`` must be numerically equivalent to
    the naive baseline for the same chunk sequence.

    This pins the contract the torch.compile path relies on: all
    data-dependent control flow lives in
    :meth:`RollingBlockKVCache.before_update`, which precomputes
    ``write_start`` / ``valid_len`` in Python before the compiled forward,
    and ``update_at`` / ``cached_{k,v}_at`` are the only traced cache ops.

    Parametrization deliberately excludes ``chunk_size > window_size``
    (e.g. ``sink_size=3, window_size=5``); the precondition is asserted
    once in :meth:`RollingBlockKVCache.__post_init__`. Real configs
    (single-view ``window_size_t=6, len_t=2``) satisfy ``chunk_size <= window_size``.
    """
    batch, n_heads = 2, 4
    dim_k, dim_v = 8, 16
    chunk_size = 8
    buffer_size = window_size + sink_size

    k_shape = (batch, buffer_size, n_heads, dim_k)
    v_shape = (batch, buffer_size, n_heads, dim_v)

    cache = _new_cache(
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        k_shape=k_shape,
        v_shape=v_shape,
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

        cache.before_update(chunk_idx)
        cache.update_at(new_k, new_v, cache.write_start)
        k_api = cache.cached_k_at(cache.valid_len)
        v_api = cache.cached_v_at(cache.valid_len)
        cache.after_update(chunk_idx)
        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)

        # Overwrite same chunk (filling-same-chunk / steady-state-replay):
        # write_start should rewind onto the just-written slot, not append.
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
        cache.update_at(new_k, new_v, cache.write_start)
        k_api = cache.cached_k_at(cache.valid_len)
        v_api = cache.cached_v_at(cache.valid_len)
        cache.after_update(chunk_idx)
        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)


@pytest.mark.ci_cpu
def test_cache_size_tracks_current_update() -> None:
    """``cache.size()`` follows the branchless ``valid_len`` precompute
    across the fill -> steady -> reuse sequence. Pure introspection test."""
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)
    assert cache.size() == 0

    for chunk_idx, expected_end in [(0, 2), (1, 4), (2, 6), (3, 6)]:
        cache.before_update(chunk_idx)
        assert cache.size() == expected_end
        values = torch.full((1, 2, 1, 1), float(chunk_idx))
        cache.update_at(values, values, cache.write_start)
        cache.after_update(chunk_idx)
        assert cache.size() == expected_end

    # Multi-step reuse path: re-running chunk 3 keeps size at the
    # steady-state total_size.
    cache.before_update(3)
    assert cache.size() == 6
    values = torch.full((1, 2, 1, 1), 30.0)
    cache.update_at(values, values, cache.write_start)
    cache.after_update(3)
    assert cache.size() == 6


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
@pytest.mark.parametrize("sink_size,window_size", [(0, 8), (0, 24), (3, 21)])
def test_block_kvcache_cudagraph_matches_baseline(
    dtype: torch.dtype,
    sink_size: int,
    window_size: int,
) -> None:
    """Branchless ``update_at`` / ``cached_*_at`` inside a CUDA graph
    (steady-state path) matches the naive baseline.

    Static ``write_start`` / ``valid_len`` (both ``int`` precomputed by
    :meth:`RollingBlockKVCache.before_update`) flow into the captured
    region as Python ints, matching the production ``torch.compile``
    branchless-int contract.
    """
    device = torch.device("cuda")
    batch, n_heads = 2, 4
    dim_k, dim_v = 8, 16
    chunk_size = 8
    buffer_size = window_size + sink_size

    k_shape = (batch, buffer_size, n_heads, dim_k)
    v_shape = (batch, buffer_size, n_heads, dim_v)

    cache = _new_cache(
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        k_shape=k_shape,
        v_shape=v_shape,
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

    def fn(
        k: torch.Tensor, v: torch.Tensor, write_start: int, valid_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache.update_at(k, v, write_start)
        return cache.cached_k_at(valid_len), cache.cached_v_at(valid_len)

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
        rng = cache.range
        if cache.is_steady_state():
            steady_k.copy_(new_k)
            steady_v.copy_(new_v)
            if graph is None:
                # Capture graph after warmup.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup_iters):
                        fn(steady_k, steady_v, rng.write_start, rng.valid_len)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    k_api, v_api = fn(
                        steady_k, steady_v, rng.write_start, rng.valid_len
                    )
            else:
                graph.replay()
        else:
            k_api, v_api = fn(new_k, new_v, rng.write_start, rng.valid_len)
        cache.after_update(chunk_idx)

        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)

        # Overwrite same chunk: write_start rewinds onto the just-written slot.
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
        rng = cache.range
        if graph is not None:
            assert cache.is_steady_state()
            steady_k.copy_(new_k)
            steady_v.copy_(new_v)
            graph.replay()
        else:
            k_api, v_api = fn(new_k, new_v, rng.write_start, rng.valid_len)
        cache.after_update(chunk_idx)

        torch.testing.assert_close(k_api, k_baseline)
        torch.testing.assert_close(v_api, v_baseline)

    # make sure the graph is captured.
    assert graph is not None


# ---- inlined rolling-cursor + lock-step tests ----


@pytest.mark.ci_cpu
def test_cursor_advances_through_filling_and_steady() -> None:
    """The inlined cursor on a RollingBlockKVCache reproduces the state machine."""
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)

    assert cache._n_cached == 0
    assert cache._curr_chunk_idx is None
    assert cache._prev_chunk_idx == -1

    for chunk_idx, expected_size in [(0, 2), (1, 4), (2, 6), (3, 6), (4, 6)]:
        cache.before_update(chunk_idx)
        assert cache._curr_chunk_idx == chunk_idx
        assert cache.size() == expected_size
        assert cache.valid_len == expected_size
        cache.after_update(chunk_idx)
        assert cache._curr_chunk_idx is None
        assert cache._prev_chunk_idx == chunk_idx


@pytest.mark.ci_cpu
def test_before_update_is_idempotent() -> None:
    """Re-entrant before_update(chunk_idx) calls within an AR step are no-ops."""
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)

    # AR 0: first call advances, repeated calls do nothing.
    cache.before_update(0)
    n0 = cache._n_cached
    needs_roll0 = cache._needs_buffer_roll
    for _ in range(10):
        cache.before_update(0)
        assert cache._n_cached == n0
        assert cache._needs_buffer_roll == needs_roll0
        assert cache._curr_chunk_idx == 0
    cache.after_update(0)

    # Fill to steady state, then verify the steady-state transition sets the
    # roll flag and idempotent re-entries observe it.
    for chunk_idx in [1, 2]:
        cache.before_update(chunk_idx)
        cache.after_update(chunk_idx)
    cache.before_update(3)
    assert cache._needs_buffer_roll is True
    for _ in range(5):
        cache.before_update(3)
        assert cache._needs_buffer_roll is True


@pytest.mark.ci_cpu
def test_after_update_is_idempotent() -> None:
    """Re-entrant after_update(chunk_idx) calls are no-ops."""
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)

    cache.before_update(0)
    cache.after_update(0)
    # Re-entry within the same AR step sees _curr_chunk_idx already None.
    for _ in range(10):
        cache.after_update(0)
        assert cache._curr_chunk_idx is None
        assert cache._prev_chunk_idx == 0
        assert cache._n_cached == 2


@pytest.mark.ci_cpu
def test_write_start_matches_branchy_logic() -> None:
    """write_start covers steady, filling-advance, filling-same."""
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)

    # Filling, advancing.
    cache.before_update(0)
    assert cache.write_start == 0
    cache.after_update(0)
    cache.before_update(1)
    assert cache.write_start == 2
    cache.after_update(1)
    cache.before_update(2)
    assert cache.write_start == 4
    cache.after_update(2)

    # Steady state: rightmost slot, total_size - chunk_size = 6 - 2 = 4.
    cache.before_update(3)
    assert cache.is_steady_state()
    assert cache.write_start == 4
    cache.after_update(3)

    # Filling-same in steady state: chunk_idx repeats, _curr == _prev.
    cache.before_update(3)
    assert cache.write_start == 4
    cache.after_update(3)


@pytest.mark.ci_cpu
def test_precomputes_write_start_and_valid_len_in_before_update() -> None:
    """``write_start`` / ``valid_len`` are populated by ``before_update`` itself
    (no separate compute step), and 10 idempotent re-entries within the same AR
    step don't drift either field."""
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)

    # Zero-init before any before_update.
    assert cache.write_start == 0
    assert cache.valid_len == 0

    # AR 0: first call populates both fields; 10 idempotent re-entries leave
    # them untouched.
    cache.before_update(0)
    assert cache.write_start == 0
    assert cache.valid_len == 2
    for _ in range(10):
        cache.before_update(0)
        assert cache.write_start == 0
        assert cache.valid_len == 2
    cache.after_update(0)

    # AR 1, AR 2: filling continues, both fields advance lock-step with cursor.
    for chunk_idx, expected_ws, expected_vl in [(1, 2, 4), (2, 4, 6)]:
        cache.before_update(chunk_idx)
        assert cache.write_start == expected_ws
        assert cache.valid_len == expected_vl
        cache.after_update(chunk_idx)

    # AR 3 hits steady state: rightmost slot at total_size - chunk_size = 4,
    # full valid_len = total_size = 6.
    cache.before_update(3)
    assert cache.is_steady_state()
    assert cache.write_start == 4
    assert cache.valid_len == 6
    for _ in range(10):
        cache.before_update(3)
        assert cache.write_start == 4
        assert cache.valid_len == 6
    cache.after_update(3)


@pytest.mark.ci_cpu
def test_precomputes_for_multi_step_reuse_path() -> None:
    """An advance -> reuse -> advance sequence (multi-step scheduler reusing
    the same chunk) must re-precompute write_start / valid_len for the reuse
    step, not leave stale values from the prior advance.

    Specifically guards against an early-return regression in
    ``RollingBlockKVCache.before_update``: the reuse path (`chunk_idx ==
    _prev_chunk_idx`) must reach the precompute block at the bottom so
    ``write_start`` rewinds onto the just-written slot
    (``_n_cached - chunk_size``) instead of holding the previous-step's
    append offset.
    """
    cache = _cursor_cache(chunk_size=2, window_size=6, sink_size=0)

    # Advance: AR 0 appends at 0, valid_len = 2.
    cache.before_update(0)
    assert cache.write_start == 0
    assert cache.valid_len == 2
    cache.after_update(0)

    # Reuse: scheduler re-runs chunk 0 on the same slot. The reuse-slot
    # semantics rewind write_start to ``_n_cached - chunk_size = 2 - 2 = 0``
    # and keep valid_len at the same 2 (no new tokens become visible).
    cache.before_update(0)
    assert cache.write_start == 0
    assert cache.valid_len == 2
    cache.after_update(0)

    # Advance to AR 1: now appending fresh tokens at ``_n_cached = 2``,
    # valid_len grows to 4. Verifies the reuse cycle didn't poison _n_cached.
    cache.before_update(1)
    assert cache.write_start == 2
    assert cache.valid_len == 4
    cache.after_update(1)

    # Fill to steady state, then exercise the steady-state reuse path:
    # write_start must rewind to total_size - chunk_size on reuse, not
    # stay at the previous append offset.
    cache.before_update(2)
    assert cache.write_start == 4
    assert cache.valid_len == 6
    cache.after_update(2)
    cache.before_update(3)
    assert cache.is_steady_state()
    assert cache.write_start == 4
    assert cache.valid_len == 6
    cache.after_update(3)
    # Steady-state reuse of chunk 3 must keep write_start at the rightmost
    # slot (still 4) and valid_len at the full total_size (still 6).
    cache.before_update(3)
    assert cache.write_start == 4
    assert cache.valid_len == 6
    cache.after_update(3)


@pytest.mark.ci_cpu
def test_lockstep_caches_advance_equally() -> None:
    """N independent RollingBlockKVCaches sharing geometry + chunk sequence
    advance in lock-step: the per-cache before/after_update cascade gives every
    cache identical branchless params and advances each cursor exactly once."""
    num_caches = 4
    caches = [
        _new_cache(
            chunk_size=2,
            window_size=6,
            sink_size=0,
            k_shape=(1, 6, 1, 4),
            v_shape=(1, 6, 1, 4),
        )
        for _ in range(num_caches)
    ]

    for chunk_idx in [0, 1, 2, 3, 4]:
        # Per-cache cascade: each cache advances + rolls its own cursor/buffer.
        _cascade_before(caches, chunk_idx)
        # Lock-step geometry => each cache sees the same window-relative state.
        write_starts = [c.write_start for c in caches]
        valid_lens = [c.valid_len for c in caches]
        assert all(ws == write_starts[0] for ws in write_starts)
        assert all(vl == valid_lens[0] for vl in valid_lens)

        # Each cache writes (per-buffer) at the same write_start.
        write_start = write_starts[0]
        for idx, c in enumerate(caches):
            payload = torch.full((1, 2, 1, 4), float(10 * idx + chunk_idx))
            c.update_at(payload, payload, write_start)

        _cascade_after(caches, chunk_idx)

        # Each cache advanced exactly once per AR step (``_n_cached`` increments
        # by chunk_size, not chunk_size * num_caches).
        assert caches[0]._n_cached == min((chunk_idx + 1) * 2, caches[0].total_size)


@pytest.mark.ci_cpu
def test_lockstep_caches_cond_uncond_pattern() -> None:
    """Mimic the cond/uncond CFG pattern: two independent groups of caches with
    physically-distinct K/V tensors advance in lock-step. The per-cache cascade
    across both branches advances every cursor once and rolls every buffer.

    Models call cache.start() -> network_cache(_uncond).before_update(idx) on
    each branch, run predict_flow, then cache.finalize() ->
    network_cache(_uncond).after_update(idx).
    """

    def _make_group() -> list[RollingBlockKVCache]:
        return [
            _new_cache(
                chunk_size=2,
                window_size=6,
                sink_size=0,
                k_shape=(1, 6, 1, 4),
                v_shape=(1, 6, 1, 4),
            )
            for _ in range(3)
        ]

    cond_caches = _make_group()
    uncond_caches = _make_group()
    all_caches = cond_caches + uncond_caches

    # Distinct buffer identities.
    assert cond_caches[0]._k.data_ptr() != uncond_caches[0]._k.data_ptr()

    for chunk_idx in [0, 1, 2, 3, 4]:
        # Per-cache cascade across both branches advances each cursor once and
        # rolls every member buffer (cond + uncond).
        _cascade_before(all_caches, chunk_idx)
        # Every cache sees identical branchless params.
        write_starts = [bc.write_start for bc in all_caches]
        assert all(ws == write_starts[0] for ws in write_starts)
        valid_lens = [bc.valid_len for bc in all_caches]
        assert all(vl == valid_lens[0] for vl in valid_lens)

        # Writes.
        write_start = write_starts[0]
        for branch_idx, branch in enumerate([cond_caches, uncond_caches]):
            for bc_idx, bc in enumerate(branch):
                payload = torch.full(
                    (1, 2, 1, 4), float(100 * branch_idx + 10 * bc_idx + chunk_idx)
                )
                bc.update_at(payload, payload, write_start)

        # Finalize across BOTH branches; each cursor advances once.
        _cascade_after(all_caches, chunk_idx)
        # _n_cached advanced by chunk_size, not by chunk_size * 2 branches.
        assert all_caches[0]._n_cached == min(
            (chunk_idx + 1) * 2, all_caches[0].total_size
        )


@pytest.mark.ci_cpu
def test_lockstep_buffer_roll_propagates_to_every_cache() -> None:
    """When the lock-step caches enter steady-state, each member's own
    before_update rolls its private buffer left (not just one cache)."""
    caches = [
        _new_cache(
            chunk_size=2,
            window_size=6,
            sink_size=0,
            k_shape=(1, 6, 1, 4),
            v_shape=(1, 6, 1, 4),
        )
        for _ in range(3)
    ]

    # Fill the cache to steady-state. Each AR step writes a per-cache
    # distinct value so we can verify the roll moved each cache's buffer.
    for chunk_idx in [0, 1, 2]:
        _cascade_before(caches, chunk_idx)
        write_start = caches[0].write_start
        for cache_idx, c in enumerate(caches):
            payload = torch.full(
                (1, 2, 1, 4), float(100 * cache_idx + 10 * chunk_idx + 1)
            )
            c.update_at(payload, payload, write_start)
        _cascade_after(caches, chunk_idx)

    # Cache state before steady-state roll: each cache's buffer = [chunk0, chunk1, chunk2].
    pre_roll_snapshots = [c._k.clone() for c in caches]
    for cache_idx, snap in enumerate(pre_roll_snapshots):
        # Each cache's chunk 0 entries should be 100*cache_idx + 1 (the value we wrote for chunk 0).
        expected_chunk0 = float(100 * cache_idx + 1)
        assert snap[0, 0, 0, 0].item() == expected_chunk0

    # Now AR step 3 enters steady-state: each cache's before_update rolls its
    # own buffer left by chunk_size=2, so what was at position [2:6] now lives
    # at [0:4]. The new position 0 holds the chunk written at chunk_idx=1,
    # i.e. value 100*cache_idx + 11.
    _cascade_before(caches, 3)

    for cache_idx, c in enumerate(caches):
        expected_after_roll = float(100 * cache_idx + 11)
        assert c._k[0, 0, 0, 0].item() == expected_after_roll, (
            f"cache {cache_idx} buffer was not rolled correctly"
        )


@pytest.mark.ci_cpu
def test_rolling_cache_exposes_geometry_fields() -> None:
    """A RollingBlockKVCache exposes its inlined geometry (chunk/window/sink)
    and derives ``total_size = sink_size + window_size``."""
    cache = _new_cache(
        chunk_size=2,
        window_size=6,
        sink_size=2,
        k_shape=(1, 8, 1, 4),
        v_shape=(1, 8, 1, 4),
    )
    assert (cache.chunk_size, cache.window_size, cache.sink_size, cache.seq_dim) == (
        2,
        6,
        2,
        1,
    )
    assert cache.total_size == 8


@pytest.mark.ci_cpu
def test_rolling_cache_k_shape_must_match_total_size() -> None:
    """Passing ``k_shape[seq_dim]`` that disagrees with ``sink_size + window_size``
    raises in ``__post_init__``."""
    # total_size == 8; k_shape[seq_dim=1] == 6 disagrees.
    with pytest.raises(AssertionError):
        _new_cache(
            chunk_size=4,
            window_size=8,
            sink_size=0,
            k_shape=(1, 6, 1, 4),
            v_shape=(1, 6, 1, 4),
        )


@pytest.mark.ci_cpu
def test_rolling_cache_constructor_requires_size_fields() -> None:
    """Omitting a required size field (``chunk_size``) raises ``TypeError`` --
    the inlined geometry fields have no defaults.
    """
    # Cast through ``Any`` so static type checkers (mypy, pyright, ty)
    # don't flag the deliberately-missing required kwarg.
    ctor = cast(Any, RollingBlockKVCache)
    with pytest.raises(TypeError, match="chunk_size"):
        ctor(
            k_shape=(1, 6, 1, 4),
            v_shape=(1, 6, 1, 4),
            seq_dim=1,
            window_size=6,
            sink_size=0,
            device="cpu",
            dtype=torch.float32,
        )
