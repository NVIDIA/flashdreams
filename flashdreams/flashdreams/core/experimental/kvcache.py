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

"""Experimental block KV cache (alternative implementation)."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class BlockKVCache:
    """
    KV cache for causal attention with a fixed-size local window and CUDA-graph support.

    Keys and values can have arbitrary shape ``[..., total_size, ...]``; the sequence
    (rolling) dimension is given by ``seq_dim`` (dimension index, can be negative).
    Layout along that dimension: [sink tokens | local window tokens]. Sink tokens are
    never evicted; the local window rolls left as new chunks are added if full. Chunks are
    non-overlapping: each update adds one chunk of tokens at the next logical position
    in the full sequence.

    Phases:
        - Filling: cache not yet full; tokens are written contiguously;
          ``cached_k()`` / ``cached_v()`` return only the valid prefix.
        - Filling-then-steady: cache not yet full but will be full after the update;
          might first trigger a left-roll of the local window. Then, it will be in
          steady-state thus overwrite the rightmost positions.
          ``cached_k()`` / ``cached_v()`` return the full buffer.
        - Steady-state: cache full; each new chunk triggers a left-roll of the
          local window and overwrites the rightmost positions;
          ``cached_k()`` / ``cached_v()`` return the full buffer.

    Note: only supports continous chunks or overwriting the same cache positions.
    The argument ``chunk_start`` and ``chunk_end`` are the start and end indices of
    the new chunk in the full sequence (not the indices into the cache). The new chunk
    should either be contiguous with the previous chunk (i.e. next chunk_start ==
    previous chunk_end), in which case the chunk is appended, or, in steady-state,
    written after the roll. If next chunk_start equals the previous chunk_start, the
    same cache positions are overwritten.

    Per-step usage:
        1. before_update(chunk_start, chunk_end) — prepare (roll local window if steady-state).
        2. update(k, v) — write the new chunk's keys/values into the cache.
        3. cached_k() / cached_v() — get cached keys/values for attention.
        4. after_update(chunk_start, chunk_end) — update internal bookkeeping.
    """

    k_shape: tuple[int, ...]
    """Shape of the keys. Must be the same as the values shape except for the last dimension."""

    v_shape: tuple[int, ...]
    """Shape of the values. Must be the same as the keys shape except for the last dimension."""

    seq_dim: int
    """Sequence dimension that will be rolled. Can be negative."""

    window_size: int
    """Size of the local attention window (excluding sink tokens)."""

    sink_size: int = 0
    """Number of sink tokens at the start of the cache that are never evicted. Defaults to 0."""

    device: torch.device | str = torch.device("cuda")
    """Device to store the cache on."""

    dtype: torch.dtype = torch.float16
    """Data type to store the cache in."""

    _prev_chunk_start: int = -1
    """Start index of the last written chunk; -1 when empty."""

    _prev_chunk_end: int = 0
    """End index of the last written chunk; 0 when empty."""

    _n_cached: int = 0
    """Number of valid tokens currently in the cache."""

    _k: Tensor = field(init=False)
    """Cached keys. shape ``[..., total_size, ..., Dk]``, where the ``total_size`` is the length of the cache buffer at ``seq_dim`` dimension."""

    _v: Tensor = field(init=False)
    """Cached values. shape ``[..., total_size, ..., Dv]``, where the ``total_size`` is the length of the cache buffer at ``seq_dim`` dimension."""

    def __post_init__(self) -> None:
        # k and v should have the same shape except for the last dimension
        assert self.k_shape[:-1] == self.v_shape[:-1], (
            "k and v must have the same shape except for the last dimension"
        )

        # update seq_dim to be positive
        tensor_dim = len(self.k_shape)
        assert -tensor_dim <= self.seq_dim < tensor_dim, (
            f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {self.seq_dim}"
        )
        self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim

        # check non-negative sink size
        assert self.sink_size >= 0, "sink_size must be non-negative"

        # buffer length at seq_dim must equal sink_size + window_size
        expected_length = self.sink_size + self.window_size
        assert self.k_shape[self.seq_dim] == expected_length, (
            f"k_shape[seq_dim] ({self.k_shape[self.seq_dim]}) must equal sink_size + window_size ({expected_length})"
        )

        # initialize k and v
        self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
        self._v = torch.empty(self.v_shape, device=self.device, dtype=self.dtype)

    def _seq_slice(self, start: int | None, end: int | None) -> tuple[slice | int, ...]:
        """Return an index tuple that slices the sequence dimension to [start:end]; other dims are :."""
        idx: list[slice | int] = [slice(None)] * len(self.k_shape)
        idx[self.seq_dim] = slice(start, end)
        return tuple(idx)

    def _roll_local_window_left(self, evict_size: int) -> None:
        """Shift the local window left by evict_size tokens.

        Before the roll:
        | -- sink tokens -- | -- (evicted tokens) -- retained window tokens -- |

        After the roll:
        | -- sink tokens -- | -- retained window tokens -- |

        This function will write the "retained window tokens" to the left, if there are any.

        Args:
            evict_size: Number of tokens to evict from the left of the local window.
        """
        print(f"rolling local window left with evict_size {evict_size}")
        if evict_size <= 0:
            # No tokens to evict.
            return

        if self._n_cached < self.sink_size:
            # If the sink is not yet full, do not evict any tokens.
            return

        if self._n_cached < self.sink_size + evict_size:
            # If the sink is full but the local window size is less than the evict size,
            # then evict all the tokens in the local window. Nothing to roll.
            self._n_cached = self.sink_size
        else:
            # If the sink is full and the local window size is greater than the evict size,
            # then evict the leftmost evict_size tokens, and roll the rest to left.

            # range of source tokens to roll
            src_start = self.sink_size + evict_size
            src_end = self._n_cached
            # range of destination tokens to write
            dst_start = self.sink_size
            dst_end = self._n_cached - evict_size

            # write the tokens to new positions.
            dst_slice = self._seq_slice(dst_start, dst_end)
            src_slice = self._seq_slice(src_start, src_end)
            self._k[dst_slice] = self._k[src_slice].clone()
            self._v[dst_slice] = self._v[src_slice].clone()
            self._n_cached -= evict_size

    def _append_to_end(self, k: Tensor, v: Tensor) -> None:
        """Append the new chunk to the end of the cache.

        Before the write:
        | -- sink tokens -- | -- local window tokens -- |

        After the write:
        | -- sink tokens -- | -- local window tokens -- | -- new chunk -- |
        """
        total_size = self._k.shape[self.seq_dim]
        chunk_size = k.shape[self.seq_dim]

        write_start = self._n_cached
        write_end = min(write_start + chunk_size, total_size)
        sl_write = self._seq_slice(write_start, write_end)

        read_end = chunk_size
        read_start = chunk_size - (write_end - write_start)
        sl_read = self._seq_slice(read_start, read_end)
        self._k[sl_write] = k[sl_read]
        self._v[sl_write] = v[sl_read]
        self._n_cached += read_end - read_start

    def _write_to_rightmost(self, k: Tensor, v: Tensor, chunk_start: int) -> None:
        """Write the new chunk to the rightmost positions of the valid tokens.

        Before the write:
        | -- sink tokens -- | -- local window tokens -- (old tokens) -- |

        After the write:
        | -- sink tokens -- | -- local window tokens -- (new tokens) -- |
        """
        print(f"writing to rightmost: {k.shape}")
        chunk_size = k.shape[self.seq_dim]

        sink_capacity = max(self.sink_size - self._n_cached, 0)
        if sink_capacity > 0:
            # If the sink is not yet full, partial of the new tokens need
            # to be written to the sink.
            if chunk_size < sink_capacity:
                # The new tokens fit into the sink. So we should actually
                # append the new tokens to the end of the cache.
                self._append_to_end(k, v)
            else:
                # The new tokens only partially fit into the sink.
                # 1. write to sink
                write_start = self._n_cached
                write_end = write_start + sink_capacity
                read_start = 0
                read_end = read_start + sink_capacity
                sl_read = self._seq_slice(read_start, read_end)
                sl_write = self._seq_slice(write_start, write_end)
                self._k[sl_write] = k[sl_read]
                self._v[sl_write] = v[sl_read]
                self._n_cached += sink_capacity
                # 2. the rest of the tokens should be written to the
                # rightmost positions of the valid tokens. The leftover
                # slice starts ``sink_capacity`` tokens later in the
                # original sequence.
                sl_read = self._seq_slice(read_end, chunk_size)
                self._write_to_rightmost(
                    k[sl_read], v[sl_read], chunk_start + sink_capacity
                )

        else:
            # If the sink is full.
            write_end = self._n_cached
            write_start = write_end - chunk_size
            if write_start > self.sink_size:
                # The input token does not overlap with the sink tokens. Simply write to the
                # rightmost positions of the valid tokens.
                sl = self._seq_slice(write_start, write_end)
                self._k[sl] = k
                self._v[sl] = v
            else:
                # The input token overlaps with the sink tokens, so we only keep partial of it.
                # Here we can safely assume the sink tokens have already been filled up because
                # we have processed that case in the above if statement.
                write_start = self.sink_size
                read_end = chunk_size
                read_start = read_end - (write_end - write_start)
                sl_read = self._seq_slice(read_start, read_end)
                sl_write = self._seq_slice(write_start, write_end)
                self._k[sl_write] = k[sl_read]
                self._v[sl_write] = v[sl_read]

                # We might also need to update the sink tokens.
                if chunk_start < self.sink_size:
                    sl_read = sl_write = self._seq_slice(chunk_start, self.sink_size)
                    self._k[sl_write] = k[sl_read]
                    self._v[sl_write] = v[sl_read]

    def is_steady_state(self, chunk_start: int) -> bool:
        """If this update will write to the same positions as the rest of the updates."""
        is_overlapping_with_sink = chunk_start < self.sink_size
        if is_overlapping_with_sink:
            # Exclude sink update from steady state.
            return False

        # TODO: fill in the rest of the logic.
        return False

    def update(self, k: Tensor, v: Tensor, chunk_start: int) -> None:
        """Write the new chunk's keys and values into the cache."""
        if self._prev_chunk_start == chunk_start:
            self._write_to_rightmost(k, v, chunk_start)
        else:
            assert self._prev_chunk_end == chunk_start
            chunk_size = k.shape[self.seq_dim]
            chunk_end = chunk_start + chunk_size

            total_size = self._k.shape[self.seq_dim]
            evict_size = self._n_cached + chunk_size - total_size
            self._roll_local_window_left(evict_size)
            self._append_to_end(k, v)

            self._prev_chunk_start = chunk_start
            self._prev_chunk_end = chunk_end

    def cached_v(self) -> Tensor:
        total_size = self._k.shape[self.seq_dim]
        if self._n_cached == total_size:
            return self._v
        else:
            return self._v[self._seq_slice(0, self._n_cached)]

    def cached_k(self) -> Tensor:
        total_size = self._k.shape[self.seq_dim]
        if self._n_cached == total_size:
            return self._k
        else:
            return self._k[self._seq_slice(0, self._n_cached)]
