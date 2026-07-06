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

from typing import TypeVar

import torch
from torch import Tensor
from torch.distributed import (
    ProcessGroup,
    all_gather,
    all_gather_object,
    get_world_size,
)


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
