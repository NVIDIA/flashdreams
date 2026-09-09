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

"""Fixed-slot K/V storage and bounded token windows for multi-view inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

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
        with torch.no_grad():
            for layer, ((key, value), key_buffer, value_buffer) in enumerate(
                zip(chunk, self._k, self._v, strict=True)
            ):
                self._validate_chunk_layer(
                    key, key_buffer, count=count, layer=layer, kind="key"
                )
                self._validate_chunk_layer(
                    value, value_buffer, count=count, layer=layer, kind="value"
                )
                key_buffer[destination] = key
                value_buffer[destination] = value

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
        with torch.no_grad():
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
                key_buffer[prefix] = key
                value_buffer[prefix] = value
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
