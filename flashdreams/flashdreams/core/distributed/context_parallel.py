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

"""Tensor and object splitting/gathering primitives for context parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import (
    ProcessGroup,
    all_gather,
    all_gather_object,
    get_world_size,
)

from flashdreams.core.distributed.parallel import ParallelContext, balanced_ranges


def split_inputs_cp(
    x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None
) -> Tensor:
    """Slice a tensor along ``seq_dim`` to this rank's CP shard.

    Args:
        x: Input tensor.
        seq_dim: Dimension to split along (negative indexing supported).
        cp_group: CP process group; ``None`` returns ``x`` unchanged.

    Returns:
        Contiguous slice of length ``x.shape[seq_dim] // cp_size``.

    Raises:
        AssertionError: ``seq_dim`` is not divisible by the CP size.
    """
    if cp_group is None:
        return x

    cp_size = cp_group.size()
    if seq_dim < 0:
        seq_dim = x.ndim + seq_dim  # bring it to positive dimension

    assert x.shape[seq_dim] % cp_size == 0, (
        f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    )
    x = x.view(
        *x.shape[:seq_dim],
        cp_size,
        x.shape[seq_dim] // cp_size,
        *x.shape[(seq_dim + 1) :],
    )
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x.contiguous()


def cat_outputs_cp(
    x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None
) -> Tensor:
    """Gather and concatenate per-rank tensors along ``seq_dim``.

    Args:
        x: This rank's local tensor.
        seq_dim: Concatenation dimension.
        cp_group: CP process group; ``None`` returns ``x`` unchanged.

    Returns:
        Tensor with the gathered shards concatenated along ``seq_dim``.

    Raises:
        RuntimeError: ``all_gather`` failed.
    """
    if cp_group is None:
        return x

    x = x.contiguous()
    world_size = get_world_size(cp_group)
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather tensors") from e

    return torch.cat(gathered_tensors, dim=seq_dim)


T = TypeVar("T")


def split_inputs_cp_object_list(
    object_list: list[T], cp_group: ProcessGroup | None = None
) -> list[T]:
    """Slice a list to this rank's CP shard.

    Args:
        object_list: List to split.
        cp_group: CP process group; ``None`` returns ``object_list`` unchanged.

    Returns:
        This rank's contiguous slice of length ``len(object_list) // cp_size``.

    Raises:
        AssertionError: ``len(object_list)`` is not divisible by the CP size.
    """
    if cp_group is None:
        return object_list

    cp_size = cp_group.size()
    n_objects = len(object_list)
    assert n_objects % cp_size == 0, f"{n_objects} cannot divide cp_size {cp_size}"

    n_objects_per_rank = n_objects // cp_size
    rank = cp_group.rank()
    start_idx = rank * n_objects_per_rank
    end_idx = start_idx + n_objects_per_rank
    return object_list[start_idx:end_idx]


def cat_outputs_cp_object_list(
    object_list: list[T], cp_group: ProcessGroup | None = None
) -> list[T]:
    """Gather per-rank lists and flatten into a single list.

    Args:
        object_list: This rank's local list.
        cp_group: CP process group; ``None`` returns ``object_list`` unchanged.

    Returns:
        Flattened concatenation of every rank's list.
    """
    if cp_group is None:
        return object_list

    world_size = get_world_size(cp_group)
    gathered_object_list: list[list[T]] = [[] for _ in range(world_size)]

    try:
        all_gather_object(gathered_object_list, object_list, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather objects") from e

    # all_gather_object treats each list as a single object -> flatten.
    return [item for sublist in gathered_object_list for item in sublist]


@dataclass(frozen=True)
class TokenShard:
    """Contiguous token ranges in process-group rank order."""

    ranges: tuple[tuple[int, int], ...]
    rank: int

    @property
    def total(self) -> int:
        """Tokens in the whole sequence."""
        return self.ranges[-1][1]

    @property
    def local(self) -> tuple[int, int]:
        """This rank's ``[start, end)``."""
        return self.ranges[self.rank]

    @property
    def sizes(self) -> list[int]:
        """Tokens per rank, in rank order -- the shape a gather has to expect."""
        return [end - start for start, end in self.ranges]

    @property
    def local_size(self) -> int:
        """Tokens this rank owns."""
        start, end = self.local
        return end - start

    @property
    def is_sharded(self) -> bool:
        """Whether this actually splits anything."""
        return len(self.ranges) > 1

    def select(self, x: Tensor, dim: int) -> Tensor:
        """This rank's slice of ``x`` along its token axis.

        A ``narrow``, so the result is a view and the tensor it came from is not
        copied. Callers that keep it past a write to the source should clone.
        """
        start, end = self.local
        return x.narrow(dim, start, end - start)


def build_shard(total: int, ctx: ParallelContext) -> TokenShard:
    """Divide ``total`` tokens over ``ctx``'s context axis, near-equally.

    The remainder goes to the low ranks a token at a time, so the widest range
    is never more than one token past the narrowest and no rank waits on a
    materially longer one.

    Raises:
        ValueError: there are fewer tokens than ranks, which would leave a rank
            with nothing to denoise and a gather with an empty buffer.
    """
    if total < 1:
        raise ValueError(f"a shard needs at least one token, got {total}.")
    size = ctx.cp_size
    if size > total:
        raise ValueError(
            f"cp_size={size} is past the {total} tokens in this pass; a rank with no "
            "token has nothing to denoise."
        )
    return TokenShard(ranges=tuple(balanced_ranges(total, size)), rank=ctx.cp_rank)


def all_gather_sequence(
    local: Tensor, shard: TokenShard, group: dist.ProcessGroup | None, *, dim: int
) -> Tensor:
    """Gather uneven token shards in rank order, padding only for transport.

    Singleton shards return the input unchanged. ``group`` must cover exactly
    the ranks described by ``shard``; ``dim`` supports negative indexing.
    """
    if not -local.ndim <= dim < local.ndim:
        raise IndexError(
            f"token dimension {dim} is invalid for {local.ndim} dimensions"
        )
    dim %= local.ndim
    sizes = shard.sizes
    if local.shape[dim] != sizes[shard.rank]:
        raise ValueError(
            f"rank {shard.rank} offered {local.shape[dim]} tokens for its "
            f"{sizes[shard.rank]}-token range."
        )
    if not shard.is_sharded:
        return local
    if group is None:
        raise ValueError("a sharded sequence requires its context process group")
    if dist.get_world_size(group) != len(sizes) or dist.get_rank(group) != shard.rank:
        raise ValueError("context process group does not match the token shard")
    widest = max(sizes)
    padded = local
    if local.shape[dim] < widest:
        pad_shape = (
            local.shape[:dim] + (widest - local.shape[dim],) + local.shape[dim + 1 :]
        )
        padded = torch.cat([local, local.new_zeros(pad_shape)], dim=dim)
    padded = padded.contiguous()

    buffers = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(buffers, padded, group=group)
    return torch.cat(
        [
            buffer.narrow(dim, 0, size)
            for buffer, size in zip(buffers, sizes, strict=True)
        ],
        dim=dim,
    )


def local_query_range(total: int, ctx: ParallelContext) -> tuple[int, int] | None:
    """Return this rank's query range, or ``None`` for the full sequence."""
    if not ctx.context_parallel:
        return None
    return build_shard(total, ctx).local


def gather_tokens(
    local: Tensor, shard: TokenShard, ctx: ParallelContext, *, dim: int
) -> Tensor:
    """:func:`all_gather_sequence` over the context group, or a pass-through.

    The form every caller outside this module wants: it takes a context rather
    than a bare group, and it returns ``local`` untouched when the axis is inert,
    so a call site never has to ask whether it is running distributed.
    """
    if len(shard.ranges) != ctx.cp_size or shard.rank != ctx.cp_rank:
        raise ValueError("token shard does not match the context axis")
    return all_gather_sequence(local, shard, ctx.cp_group, dim=dim)
