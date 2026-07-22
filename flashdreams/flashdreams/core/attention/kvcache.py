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

"""Block KV cache for causal attention with a fixed-size local window."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from typing_extensions import Self


@dataclass
class BlockKVCache:
    """
    KV cache for causal attention with a fixed-size local window, CUDA-graph compatible.

    Keys and values can have arbitrary shape ``[..., total_size, ...]``; the sequence
    (rolling) dimension is given by ``seq_dim`` (dimension index, can be negative).
    Layout along that dimension: [sink tokens | local window tokens]. Sink tokens are
    never evicted; the local window rolls left as new chunks are added if full. Chunks are
    non-overlapping: each update adds one chunk of ``chunk_size`` tokens at the
    next logical position in the full sequence.

    Phases:
        - Filling: cache not yet full; tokens are written contiguously;
          ``cached_k()`` / ``cached_v()`` return only the valid prefix.
        - Steady-state: if adding a chunk would exceed the fixed cache size, the
          local window rolls left by the overflow amount and the new chunk
          overwrites the rightmost positions; ``cached_k()`` / ``cached_v()``
          return the full buffer.

    The argument ``chunk_idx`` (0, 1, 2, ...) is the index of the new chunk in the full
    sequence (not an index into the cache). If ``chunk_idx`` is greater than
    the previous one, the chunk is appended (or, in steady-state, written after
    the roll). If ``chunk_idx`` equals the previous one, the same cache positions
    are overwritten.

    Per-step usage:
        1. before_update(chunk_idx) — prepare (roll local window if steady-state).
        2. update(k, v) — write the new chunk's keys/values into the cache.
        3. cached_k() / cached_v() — get cached keys/values for attention.
        4. after_update(chunk_idx) — update internal bookkeeping.
    """

    k_shape: tuple[int, ...]
    """Shape of the keys. Must be the same as the values shape except for the last dimension."""

    v_shape: tuple[int, ...]
    """Shape of the values. Must be the same as the keys shape except for the last dimension."""

    seq_dim: int
    """Sequence dimension that will be rolled. Can be negative."""

    chunk_size: int
    """Number of tokens processed each time."""

    window_size: int
    """Size of the local attention window (excluding sink tokens)."""

    sink_size: int = 0
    """Number of sink tokens at the start of the cache that are never evicted. Defaults to 0."""

    device: torch.device | str = torch.device("cuda")
    """Device to store the cache on."""

    dtype: torch.dtype = torch.float16
    """Data type to store the cache in."""

    _prev_chunk_idx: int = -1
    """Chunk index of the last written chunk; -1 when empty."""

    _curr_chunk_idx: int | None = None
    """The index of the current chunk that is being processed. None when empty."""

    _n_cached: int = 0
    """Number of valid tokens currently in the cache."""

    _k: Tensor = field(init=False)
    """Cached keys. shape ``[..., total_size, ..., Dk]``, where the ``total_size`` is the length of the cache buffer at ``seq_dim`` dimension."""

    _v: Tensor = field(init=False)
    """Cached values. shape ``[..., total_size, ..., Dv]``, where the ``total_size`` is the length of the cache buffer at ``seq_dim`` dimension."""

    @property
    def size(self) -> int:
        """Number of valid cached tokens visible to attention."""
        if self._curr_chunk_idx is None:
            return self._n_cached
        return self._visible_end()

    @property
    def write_end(self) -> int:
        """Right edge of the current chunk in the physical cache layout."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before write_end"
        )
        return self.size

    @classmethod
    def from_tensor(cls, k: Tensor, v: Tensor, seq_dim: int) -> Self:
        """Build a single-chunk cache pre-filled with the given key and value tensors."""
        cache = cls(
            k_shape=k.shape,
            v_shape=v.shape,
            seq_dim=seq_dim,
            chunk_size=k.shape[seq_dim],
            window_size=k.shape[seq_dim],
            device=k.device,
            dtype=k.dtype,
        )
        cache.before_update(0)
        cache.update(k, v)
        cache.after_update(0)
        cache._curr_chunk_idx = 0
        return cache

    def __post_init__(self) -> None:
        assert self.k_shape[:-1] == self.v_shape[:-1], (
            "k and v must have the same shape except for the last dimension"
        )

        tensor_dim = len(self.k_shape)
        assert -tensor_dim <= self.seq_dim < tensor_dim, (
            f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {self.seq_dim}"
        )
        # Normalize seq_dim to a non-negative index so downstream
        # indexing math doesn't have to special-case negatives.
        self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim

        assert self.sink_size >= 0, "sink_size must be non-negative"

        expected_length = self.sink_size + self.window_size
        assert self.k_shape[self.seq_dim] == expected_length, (
            f"k_shape[seq_dim] ({self.k_shape[self.seq_dim]}) must equal sink_size + window_size ({expected_length})"
        )

        self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
        self._v = torch.empty(self.v_shape, device=self.device, dtype=self.dtype)

    def _seq_slice(self, start: int | None, end: int | None) -> tuple[slice | int, ...]:
        """Return an index tuple selecting ``[start:end]`` on ``seq_dim`` and all elements elsewhere."""
        idx: list[slice | int] = [slice(None)] * len(self.k_shape)
        idx[self.seq_dim] = slice(start, end)
        return tuple(idx)

    def _roll_local_window_left(self, shift_size: int) -> None:
        """Shift valid local-window tokens left by ``shift_size`` tokens."""
        total_size = self._k.shape[self.seq_dim]
        assert 0 < shift_size <= self.chunk_size, (
            f"shift_size ({shift_size}) must be in (0, {self.chunk_size}]"
        )
        valid_end = min(self._n_cached, total_size)
        valid_local_size = max(0, valid_end - self.sink_size)
        tokens_to_keep = max(0, valid_local_size - shift_size)

        if tokens_to_keep > 0:
            src_start = self.sink_size + shift_size
            src_end = src_start + tokens_to_keep
            dst_start = self.sink_size
            dst_end = self.sink_size + tokens_to_keep

            dst_slice = self._seq_slice(dst_start, dst_end)
            src_slice = self._seq_slice(src_start, src_end)
            self._k[dst_slice] = self._k[src_slice].clone()
            self._v[dst_slice] = self._v[src_slice].clone()
        # before_update() only rolls when the next contiguous chunk would overflow
        # this fixed buffer; update() must immediately write that chunk into the
        # newly freed right edge.
        self._n_cached = total_size

    def _current_chunk_overlaps_sink(self) -> bool:
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before checking sink overlap"
        )
        return (
            self.sink_size > 0
            and self._curr_chunk_idx * self.chunk_size < self.sink_size
        )

    def _current_write_bounds(self) -> tuple[int, int]:
        """Return the physical cache range written by the current update."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before computing write bounds"
        )
        total_size = self._k.shape[self.seq_dim]
        assert self.chunk_size <= total_size, (
            f"chunk_size ({self.chunk_size}) must be <= cache size ({total_size})"
        )

        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            write_start = torch.sym_min(self._n_cached, total_size - self.chunk_size)
            write_end = write_start + self.chunk_size
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            write_end = torch.sym_min(self._n_cached, total_size)
            write_start = torch.sym_max(write_end - self.chunk_size, 0)
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )
        return write_start, write_end

    def _write_current_chunk(self, k: Tensor, v: Tensor) -> None:
        """Write the current chunk through a filling/steady compatible path."""
        write_start, write_end = self._current_write_bounds()
        read_start = 0
        read_end = write_end - write_start

        if (
            self.sink_size > 0
            and not self._current_chunk_overlaps_sink()
            and write_start < self.sink_size
        ):
            write_start = self.sink_size
            keep_size = write_end - write_start
            read_end = self.chunk_size
            read_start = read_end - keep_size

        sl_read = self._seq_slice(read_start, read_end)
        sl_write = self._seq_slice(write_start, write_end)
        self._k[sl_write] = k[sl_read]
        self._v[sl_write] = v[sl_read]

    def _visible_end(self) -> int:
        """Right edge of cached tokens visible to attention during this update."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before computing visible cache size"
        )
        total_size = self._k.shape[self.seq_dim]
        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            return torch.sym_min(self._n_cached + self.chunk_size, total_size)
        if self._curr_chunk_idx == self._prev_chunk_idx:
            return torch.sym_min(self._n_cached, total_size)
        raise ValueError(
            f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
        )

    def is_steady_state(self) -> bool:
        """Return True if the cache is full (steady-state phase)."""
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before is_steady_state()"
        )
        total_size = self._k.shape[self.seq_dim]
        is_full = total_size == self._n_cached
        is_overlapping_with_sink = (
            self.sink_size > 0
            and self._curr_chunk_idx * self.chunk_size
            < self.sink_size  # start < sink_size
        )
        return is_full and not is_overlapping_with_sink

    def before_update(self, chunk_idx: int) -> None:
        """
        Prepare the cache before writing new tokens.

        If ``chunk_idx`` equals the previous chunk index, this is a no-op. Otherwise,
        we expect the ``chunk_idx`` to be +1 from the previous chunk index. In this case,
        we will roll the local window left if the cache is in steady-state, or no op
        if the cache is in filling phase.

        Args:
            chunk_idx: Chunk index of the new chunk in the full sequence.
        """
        assert self._curr_chunk_idx is None, (
            "Must call after_update() before before_update()"
        )
        self._curr_chunk_idx = chunk_idx

        if chunk_idx == self._prev_chunk_idx:
            return

        assert chunk_idx == self._prev_chunk_idx + 1, (
            "Expected the new chunk_idx to be +1 from the previous chunk_idx, "
            f"got {chunk_idx} != {self._prev_chunk_idx} + 1"
        )
        total_size = self._k.shape[self.seq_dim]
        if not self._current_chunk_overlaps_sink():
            overflow = self._n_cached + self.chunk_size - total_size
            if overflow > 0:
                self._roll_local_window_left(overflow)

    def update(self, k: Tensor, v: Tensor) -> None:
        """
        Write the new chunk's keys and values into the cache.

        Must be called after ``before_update()`` and before ``after_update()``.

        Args:
            k: Keys; shape must match cached keys except at seq_dim, where length must be chunk_size.
            v: Values; shape must match cached values except at seq_dim, where length must be chunk_size.
        """
        assert self._curr_chunk_idx is not None, (
            "Must call before_update() before update()"
        )

        chunk_size_k = k.shape[self.seq_dim]
        chunk_size_v = v.shape[self.seq_dim]
        assert chunk_size_k == self.chunk_size, (
            f"Expected input k to have chunk_size ({chunk_size_k}) at seq_dim ({self.seq_dim}), "
            f"got {chunk_size_k} != {self.chunk_size}"
        )
        assert chunk_size_v == self.chunk_size, (
            f"Expected input v to have chunk_size ({chunk_size_v}) at seq_dim ({self.seq_dim}), "
            f"got {chunk_size_v} != {self.chunk_size}"
        )
        self._write_current_chunk(k, v)

    def after_update(self, chunk_idx: int) -> None:
        """
        Finalize bookkeeping after writing new tokens.

        Updates ``_prev_chunk_idx`` and, in filling phase, ``_n_cached``.

        Args:
            chunk_idx: The index of the new chunk in the full sequence.
        """
        assert chunk_idx == self._curr_chunk_idx, (
            f"Expected chunk_idx to be {self._curr_chunk_idx}, got {chunk_idx}"
        )

        if self._curr_chunk_idx == self._prev_chunk_idx + 1:
            if self.is_steady_state():
                pass
            else:
                total_size = self._k.shape[self.seq_dim]
                self._n_cached = min(self._n_cached + self.chunk_size, total_size)
            self._prev_chunk_idx += 1
        elif self._curr_chunk_idx == self._prev_chunk_idx:
            pass
        else:
            raise ValueError(
                f"{self._curr_chunk_idx=} should be either {self._prev_chunk_idx + 1} or {self._prev_chunk_idx}."
            )

        self._curr_chunk_idx = None

    def cached_k(self) -> Tensor:
        """
        Return cached keys for attention (valid prefix in filling phase, full buffer in steady-state).
        """
        return self._k[self._seq_slice(0, self._visible_end())]

    def cached_v(self) -> Tensor:
        """
        Return cached values for attention (valid prefix in filling phase, full buffer in steady-state).
        """
        return self._v[self._seq_slice(0, self._visible_end())]

    def reset(self) -> None:
        """Reset bookkeeping while preserving the allocated tensor storage."""
        self._prev_chunk_idx = -1
        self._curr_chunk_idx = None
        self._n_cached = 0
