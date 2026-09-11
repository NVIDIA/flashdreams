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

"""Tensor × context process meshes and balanced contiguous rank ranges."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def balanced_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``total`` into contiguous ranges, giving low ranks the remainder."""
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}.")
    if not 1 <= parts <= total:
        raise ValueError(f"parts must lie in [1, {total}], got {parts}.")
    base, extra = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        end = start + base + (1 if index < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _env_int(name: str, default: int) -> int:
    """Read one of ``torchrun``'s integers, tolerating an unset variable."""
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class ParallelContext:
    """Rank coordinates and process groups; singleton axes have no group."""

    tp_group: dist.ProcessGroup | None
    tp_rank: int
    tp_size: int
    cp_group: dist.ProcessGroup | None
    cp_rank: int
    cp_size: int
    device: torch.device

    @property
    def world_size(self) -> int:
        """Ranks in the whole rollout, being the product of the two axes."""
        return self.tp_size * self.cp_size

    @property
    def rank(self) -> int:
        """This process's world rank, recovered from its two indexes."""
        return self.cp_rank * self.tp_size + self.tp_rank

    @property
    def is_main(self) -> bool:
        """Whether this rank reports, writes video and owns stdout."""
        return self.rank == 0

    @property
    def tensor_parallel(self) -> bool:
        """Whether the model is split across ranks."""
        return self.tp_size > 1

    @property
    def context_parallel(self) -> bool:
        """Whether the chunk is split across ranks."""
        return self.cp_size > 1

    @property
    def label(self) -> str:
        """``tp<n>cp<n>``, for a filename or a chart legend."""
        return f"tp{self.tp_size}cp{self.cp_size}"

    @classmethod
    def single(cls, device: torch.device | str = "cpu") -> ParallelContext:
        """The inert context: one rank, no groups, no collectives."""
        return cls(
            tp_group=None,
            tp_rank=0,
            tp_size=1,
            cp_group=None,
            cp_rank=0,
            cp_size=1,
            device=torch.device(device),
        )

    def barrier(self) -> None:
        """Wait for every rank, or return at once when there is only this one."""
        if self.world_size > 1:
            dist.barrier()


def plan_mesh(
    world_size: int,
    tensor_parallel: int | None = None,
    *,
    head_groups: int,
) -> tuple[int, int]:
    """Choose tensor/context sizes, defaulting to evenly divided head groups.

    An explicit tensor size may leave uneven head shards but must divide the
    world and cannot exceed the model's key/value head count.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}.")
    if head_groups < 1:
        raise ValueError(f"head_groups must be >= 1, got {head_groups}.")
    if tensor_parallel is None:
        tensor_parallel = math.gcd(world_size, head_groups)
    if tensor_parallel < 1:
        raise ValueError(f"tensor_parallel must be >= 1, got {tensor_parallel}.")
    if tensor_parallel > head_groups:
        raise ValueError(
            f"tensor_parallel={tensor_parallel} is past the {head_groups} "
            "key/value head groups this checkpoint has to hand out."
        )
    if world_size % tensor_parallel:
        raise ValueError(
            f"tensor_parallel={tensor_parallel} does not divide world_size={world_size}; "
            "the mesh has to be rectangular."
        )
    return tensor_parallel, world_size // tensor_parallel


def init_parallel(
    tensor_parallel: int | None = None, *, head_groups: int
) -> ParallelContext:
    """Initialize a launcher-provided world or reuse the current process group.

    Pass the model's key/value head count as ``head_groups``. Without a
    launcher or initialized process group, return a single-process context.
    """
    world_size = (
        dist.get_world_size() if dist.is_initialized() else _env_int("WORLD_SIZE", 1)
    )
    local_rank = _env_int("LOCAL_RANK", 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tp_size, cp_size = plan_mesh(world_size, tensor_parallel, head_groups=head_groups)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size == 1:
        return ParallelContext.single(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return build_context(tp_size, cp_size, device)


def build_context(tp_size: int, cp_size: int, device: torch.device) -> ParallelContext:
    """Cut the initialized world into the mesh's row and column groups.

    Every rank walks the same two loops and creates every group in the same
    order, which :func:`torch.distributed.new_group` requires -- it is a
    collective, and a rank that skipped the groups it is not in would deadlock
    against the ranks that did not.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if tp_size < 1 or cp_size < 1 or tp_size * cp_size != world_size:
        raise ValueError(
            f"a {tp_size}x{cp_size} mesh does not cover {world_size} ranks."
        )
    tp_rank, cp_rank = rank % tp_size, rank // tp_size

    tp_group = None
    for index in range(cp_size):
        members = list(range(index * tp_size, (index + 1) * tp_size))
        group = dist.new_group(members)
        if index == cp_rank:
            tp_group = group

    cp_group = None
    for index in range(tp_size):
        members = list(range(index, world_size, tp_size))
        group = dist.new_group(members)
        if index == tp_rank:
            cp_group = group

    return ParallelContext(
        tp_group=tp_group if tp_size > 1 else None,
        tp_rank=tp_rank,
        tp_size=tp_size,
        cp_group=cp_group if cp_size > 1 else None,
        cp_rank=cp_rank,
        cp_size=cp_size,
        device=device,
    )
