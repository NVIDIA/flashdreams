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

"""K/V caches for causal attention and fixed-slot multi-view memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import Tensor
from typing_extensions import Self

LayerKV = tuple[Tensor, Tensor]
"""Key/value tensors for one attention layer."""


@dataclass(frozen=True)
class SlotRegion:
    """Describe one independently rotating region of a fixed-slot cache."""

    name: str
    """Unique name used to select the region when writing."""

    start: int
    """First token of the region in the physical cache."""

    slots: int
    """Number of circular slots in the region."""

    slot_tokens: int
    """Maximum number of tokens held by each slot."""

    extends_length: bool = False
    """Whether writes extend the visible cache prefix."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a slot region needs a non-empty name.")
        if self.start < 0:
            raise ValueError(f"region {self.name!r} starts before zero.")
        if self.slots < 0:
            raise ValueError(f"region {self.name!r} cannot have {self.slots} slots.")
        if self.slots and self.slot_tokens <= 0:
            raise ValueError(
                f"region {self.name!r} has {self.slots} slots but "
                f"{self.slot_tokens} tokens per slot."
            )
        if not self.slots and self.slot_tokens < 0:
            raise ValueError(
                f"region {self.name!r} cannot have {self.slot_tokens} tokens per slot."
            )

    @property
    def end(self) -> int:
        """Return the exclusive physical end of the region."""
        return self.start + self.slots * self.slot_tokens


class FixedSlotKVCache:
    """Store per-layer keys and values in named circular slot regions.

    A fixed prefill occupies the initial visible prefix. Regions inside that
    prefix can rotate controls or other replaceable context, while regions
    after it can extend the visible prefix with generated history. All writes
    preserve the allocated tensor addresses for CUDA-graph compatibility.
    """

    def __init__(
        self,
        prefilled: Sequence[LayerKV],
        *,
        capacity: int,
        regions: Sequence[SlotRegion],
        seq_dim: int = 2,
    ) -> None:
        """Allocate storage and copy the prefilled key/value tensors.

        Args:
            prefilled: Per-layer key/value tensors with a shared sequence length.
            capacity: Total tokens reserved along ``seq_dim``.
            regions: Independently rotating physical slot regions.
            seq_dim: Sequence dimension in every key/value tensor.

        Raises:
            ValueError: Inputs or slot regions cannot describe a valid cache.
        """
        if not prefilled:
            raise ValueError("a fixed-slot cache needs at least one layer.")
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}.")

        first_k, first_v = prefilled[0]
        if first_k.ndim != first_v.ndim:
            raise ValueError("key and value tensors must have the same rank.")
        tensor_dim = first_k.ndim
        if not -tensor_dim <= seq_dim < tensor_dim:
            raise ValueError(
                f"seq_dim must be in [-{tensor_dim}, {tensor_dim}), got {seq_dim}."
            )
        self._seq_dim = seq_dim % tensor_dim
        self._initial_length = int(first_k.shape[self._seq_dim])
        if int(first_v.shape[self._seq_dim]) != self._initial_length:
            raise ValueError("prefilled keys and values need the same sequence length.")
        if capacity < self._initial_length:
            raise ValueError(
                f"the prefill wrote {self._initial_length} tokens, past a "
                f"capacity of {capacity}."
            )

        self._capacity = capacity
        self._regions = self._validate_regions(regions)
        self._next_slot = {name: 0 for name in self._regions}
        self._length = self._initial_length
        self._k: list[Tensor] = []
        self._v: list[Tensor] = []

        for layer, (key, value) in enumerate(prefilled):
            self._validate_prefill_layer(key, value, layer)
            key_shape = list(key.shape)
            value_shape = list(value.shape)
            key_shape[self._seq_dim] = capacity
            value_shape[self._seq_dim] = capacity
            key_buffer = key.new_zeros(key_shape)
            value_buffer = value.new_zeros(value_shape)
            prefix = self._seq_slice(0, self._initial_length, tensor_dim)
            with torch.no_grad():
                key_buffer[prefix] = key
                value_buffer[prefix] = value
            self._k.append(key_buffer)
            self._v.append(value_buffer)

    @property
    def capacity(self) -> int:
        """Return the allocated token capacity."""
        return self._capacity

    @property
    def length(self) -> int:
        """Return the visible token prefix length."""
        return self._length

    @property
    def seq_dim(self) -> int:
        """Return the normalized sequence dimension."""
        return self._seq_dim

    def layers(self) -> list[LayerKV]:
        """Return per-layer views over the visible token prefix."""
        return [
            (
                key[self._seq_slice(0, self._length, key.ndim)],
                value[self._seq_slice(0, self._length, value.ndim)],
            )
            for key, value in zip(self._k, self._v, strict=True)
        ]

    def write(self, region_name: str, chunk: Sequence[LayerKV]) -> int:
        """Write a chunk into the next slot of a named region.

        Args:
            region_name: Region to advance.
            chunk: Per-layer key/value tensors to write.

        Returns:
            Physical slot index within the selected region.

        Raises:
            ValueError: The region is missing, has no slots, or cannot hold the
                supplied tensors.
        """
        try:
            region = self._regions[region_name]
        except KeyError as error:
            raise ValueError(f"unknown slot region {region_name!r}.") from error
        if not region.slots:
            raise ValueError(f"slot region {region_name!r} has no reusable slots.")
        if len(chunk) != len(self._k):
            raise ValueError(
                f"{len(chunk)} layers of chunk for {len(self._k)} layers of memory."
            )

        count = int(chunk[0][0].shape[self._seq_dim])
        if count > region.slot_tokens:
            raise ValueError(
                f"{count} tokens do not fit the {region.slot_tokens}-token slots "
                f"in region {region_name!r}."
            )
        if count < 1:
            raise ValueError("a cache write needs at least one token.")

        slot = self._next_slot[region_name]
        start = region.start + slot * region.slot_tokens
        end = start + count
        destination = self._seq_slice(start, end, self._k[0].ndim)
        tail = self._seq_slice(end, start + region.slot_tokens, self._k[0].ndim)
        for layer, ((key, value), key_buffer, value_buffer) in enumerate(
            zip(chunk, self._k, self._v, strict=True)
        ):
            self._validate_chunk_layer(
                key, key_buffer, count=count, layer=layer, kind="key"
            )
            self._validate_chunk_layer(
                value, value_buffer, count=count, layer=layer, kind="value"
            )

        with torch.no_grad():
            for (key, value), key_buffer, value_buffer in zip(
                chunk, self._k, self._v, strict=True
            ):
                key_buffer[destination] = key
                value_buffer[destination] = value
                key_buffer[tail].zero_()
                value_buffer[tail].zero_()

        if region.extends_length:
            self._length = max(
                self._length, region.start + (slot + 1) * region.slot_tokens
            )
        self._next_slot[region_name] = (slot + 1) % region.slots
        return slot

    def reset(self, prefilled: Sequence[LayerKV]) -> None:
        """Restore a fresh prefill without reallocating cache storage.

        Args:
            prefilled: Replacement prefix with the original layer shapes and
                sequence length.
        """
        if len(prefilled) != len(self._k):
            raise ValueError(
                f"{len(prefilled)} prefill layers for {len(self._k)} cache layers."
            )
        prefix = self._seq_slice(0, self._initial_length, self._k[0].ndim)
        suffix = self._seq_slice(self._initial_length, self._capacity, self._k[0].ndim)
        for layer, ((key, value), key_buffer, value_buffer) in enumerate(
            zip(prefilled, self._k, self._v, strict=True)
        ):
            self._validate_prefill_layer(key, value, layer)
            self._validate_chunk_layer(
                key,
                key_buffer,
                count=self._initial_length,
                layer=layer,
                kind="key",
            )
            self._validate_chunk_layer(
                value,
                value_buffer,
                count=self._initial_length,
                layer=layer,
                kind="value",
            )

        with torch.no_grad():
            for (key, value), key_buffer, value_buffer in zip(
                prefilled, self._k, self._v, strict=True
            ):
                key_buffer[prefix] = key
                value_buffer[prefix] = value
                key_buffer[suffix].zero_()
                value_buffer[suffix].zero_()
        self._length = self._initial_length
        for name in self._next_slot:
            self._next_slot[name] = 0

    def _validate_regions(self, regions: Sequence[SlotRegion]) -> dict[str, SlotRegion]:
        by_name: dict[str, SlotRegion] = {}
        occupied: list[SlotRegion] = []
        for region in regions:
            if region.name in by_name:
                raise ValueError(f"duplicate slot region name {region.name!r}.")
            if region.end > self._capacity:
                raise ValueError(
                    f"region {region.name!r} ends at {region.end}, past the "
                    f"cache capacity of {self._capacity}."
                )
            if not region.extends_length and region.end > self._initial_length:
                raise ValueError(
                    f"non-extending region {region.name!r} ends at {region.end}, "
                    f"past the prefilled prefix of {self._initial_length}."
                )
            by_name[region.name] = region
            if region.slots:
                occupied.append(region)

        ordered = sorted(occupied, key=lambda region: region.start)
        for previous, current in pairwise(ordered):
            if current.start < previous.end:
                raise ValueError(
                    f"slot regions {previous.name!r} and {current.name!r} overlap."
                )
        return by_name

    def _validate_prefill_layer(self, key: Tensor, value: Tensor, layer: int) -> None:
        expected_rank = self._k[0].ndim if self._k else key.ndim
        if key.ndim != expected_rank or value.ndim != expected_rank:
            raise ValueError(f"prefill layer {layer} has inconsistent tensor ranks.")
        if int(key.shape[self._seq_dim]) != self._initial_length:
            raise ValueError(
                f"prefill layer {layer} has {key.shape[self._seq_dim]} key tokens; "
                f"expected {self._initial_length}."
            )
        if int(value.shape[self._seq_dim]) != self._initial_length:
            raise ValueError(
                f"prefill layer {layer} has {value.shape[self._seq_dim]} value tokens; "
                f"expected {self._initial_length}."
            )
        for dim in range(key.ndim - 1):
            if dim != self._seq_dim and key.shape[dim] != value.shape[dim]:
                raise ValueError(
                    f"prefill layer {layer} key/value shapes disagree at dimension {dim}."
                )

    def _validate_chunk_layer(
        self,
        source: Tensor,
        destination: Tensor,
        *,
        count: int,
        layer: int,
        kind: str,
    ) -> None:
        if source.ndim != destination.ndim:
            raise ValueError(
                f"chunk layer {layer} {kind} rank does not match its cache."
            )
        for dim, (source_size, destination_size) in enumerate(
            zip(source.shape, destination.shape, strict=True)
        ):
            expected = count if dim == self._seq_dim else destination_size
            if source_size != expected:
                raise ValueError(
                    f"chunk layer {layer} {kind} dimension {dim} is {source_size}; "
                    f"expected {expected}."
                )

    def _seq_slice(self, start: int, end: int, ndim: int) -> tuple[slice, ...]:
        index = [slice(None)] * ndim
        index[self._seq_dim] = slice(start, end)
        return tuple(index)


class TokenWindow:
    """Store the most recent multi-view token frames in a circular buffer."""

    def __init__(
        self,
        *,
        num_views: int,
        frames: int,
        spatial: int,
        width: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        """Allocate a bounded ``[V, T, S, D]`` token window."""
        if min(num_views, frames, spatial, width) < 1:
            raise ValueError(
                "num_views, frames, spatial, and width must all be positive."
            )
        self._frames = frames
        self._store = torch.zeros(
            num_views, frames, spatial, width, device=device, dtype=dtype
        )
        self._end = 0

    @property
    def capacity(self) -> int:
        """Return the number of frames held at once."""
        return self._frames

    @property
    def start(self) -> int:
        """Return the oldest absolute frame still readable."""
        return max(0, self._end - self._frames)

    @property
    def end(self) -> int:
        """Return one past the newest absolute frame written."""
        return self._end

    def append(self, start: int, end: int, tokens: Tensor) -> None:
        """Append the next consecutive frame range.

        Args:
            start: First absolute frame in the range.
            end: Exclusive absolute end of the range.
            tokens: Tokens shaped ``[V, end - start, S, D]``.

        Raises:
            ValueError: The range is non-consecutive, empty, too large, or has
                an incompatible tensor shape.
        """
        if start != self._end:
            raise ValueError(
                f"frames [{start}, {end}) do not carry on from {self._end}; "
                "a window is written in order."
            )
        count = end - start
        if count < 1:
            raise ValueError(f"[{start}, {end}) is not a frame range.")
        if count > self._frames:
            raise ValueError(
                f"{count} frames at once do not fit a {self._frames}-frame window."
            )
        expected = (
            self._store.shape[0],
            count,
            self._store.shape[2],
            self._store.shape[3],
        )
        if tuple(tokens.shape) != expected:
            raise ValueError(
                f"tokens have shape {tuple(tokens.shape)}; expected {expected}."
            )
        with torch.no_grad():
            self._store[:, self._ring(start, end)] = tokens
        self._end = end

    def read(self, start: int, end: int) -> Tensor:
        """Return tokens for an absolute frame range.

        Raises:
            ValueError: The range is empty, unwritten, or no longer retained.
        """
        if start >= end:
            raise ValueError(f"[{start}, {end}) is not a frame range.")
        if end > self._end:
            raise ValueError(
                f"frames up to {self._end} are written; [{start}, {end}) "
                "reaches past that."
            )
        if start < self.start:
            raise ValueError(
                f"[{start}, {end}) reaches back past frame {self.start}, which "
                f"is as far as a {self._frames}-frame window goes."
            )
        return self._store[:, self._ring(start, end)]

    def reset(self) -> None:
        """Reset the visible frame range without reallocating storage."""
        self._end = 0

    def _ring(self, start: int, end: int) -> Tensor:
        frames = torch.arange(start, end, device=self._store.device)
        return frames % self._frames


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

    def clone_kv(self) -> tuple[Tensor, Tensor]:
        """Return clones of the full physical K/V buffers."""
        return self._k.clone(), self._v.clone()

    def overwrite_kv_(self, k: Tensor, v: Tensor) -> None:
        """Overwrite the full K/V buffers without changing their addresses.

        Args:
            k: Replacement keys with the exact cache shape.
            v: Replacement values with the exact cache shape.
        """
        if k.shape != self._k.shape or v.shape != self._v.shape:
            raise ValueError(
                "overwrite_kv_ shape mismatch: "
                f"got k {tuple(k.shape)} / v {tuple(v.shape)}, "
                f"cache holds k {tuple(self._k.shape)} / v {tuple(self._v.shape)}"
            )
        self._k.copy_(k)
        self._v.copy_(v)


__all__ = [
    "BlockKVCache",
    "FixedSlotKVCache",
    "LayerKV",
    "SlotRegion",
    "TokenWindow",
]
