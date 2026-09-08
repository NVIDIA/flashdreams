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

"""Hierarchical V → T → HW context-parallel process groups."""

from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed import ProcessGroup


@dataclass
class HierarchicalCPGroups:
    """Per-axis CP process groups (V, T, HW) and their unions.

    Use ``*_size`` to compute per-rank tensor shapes without re-deriving
    the V → T → HW power-of-2 split.
    """

    rank: int

    HW_ranks: tuple[int, ...] = field(default_factory=tuple)
    HW_group: ProcessGroup | None = None
    T_ranks: tuple[int, ...] = field(default_factory=tuple)
    T_group: ProcessGroup | None = None
    THW_ranks: tuple[int, ...] = field(default_factory=tuple)
    THW_group: ProcessGroup | None = None
    V_ranks: tuple[int, ...] = field(default_factory=tuple)
    V_group: ProcessGroup | None = None
    VHW_ranks: tuple[int, ...] = field(default_factory=tuple)
    VHW_group: ProcessGroup | None = None

    @property
    def HW_size(self) -> int:
        return 1 if self.HW_group is None else self.HW_group.size()

    @property
    def T_size(self) -> int:
        return 1 if self.T_group is None else self.T_group.size()

    @property
    def THW_size(self) -> int:
        return 1 if self.THW_group is None else self.THW_group.size()

    @property
    def V_size(self) -> int:
        return 1 if self.V_group is None else self.V_group.size()


def create_hierarchical_cp_groups(
    world_size: int, rank: int, V: int, T: int, single_group_as_none: bool = False
) -> HierarchicalCPGroups:
    """Create hierarchical CP groups by splitting V, then T, then HW.

    Sizes are clamped to powers of 2 along each axis. ``HW`` absorbs the
    remainder so that ``V * T * HW == world_size``.

    Size derivation (priority V > T > HW):
        cp_size_V   = V if power-of-2 else 1
        cp_size_T   = T if power-of-2 else 1 (capped by remaining ranks)
        cp_size_HW  = world_size // (cp_size_V * cp_size_T)
        cp_size_THW = cp_size_T * cp_size_HW  (V-major union)
        cp_size_VHW = cp_size_V * cp_size_HW  (T-major union)

    Worked example (V=1, T=4, world_size=8 -> cp_size_HW=2):
        HW  (same V,T):  [0,1] [2,3] [4,5] [6,7]
        T   (same V,HW): [0,2,4,6] [1,3,5,7]
        THW (same V):    [0,1,2,3,4,5,6,7]
        V   (singleton since cp_size_V=1)
        VHW (same T):    [0,1] [2,3] [4,5] [6,7]

    Args:
        world_size: Total GPU count.
        rank: This rank's index in ``[0, world_size)``.
        V: Number of views/videos to split across.
        T: Number of temporal chunks to split across.
        single_group_as_none: Drop singleton groups so callers can short-
            circuit with ``if group is not None``.

    Returns:
        Populated ``HierarchicalCPGroups`` for ``rank``.
    """

    def is_power_of_2(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    dist_initialized = True if dist.is_initialized() else False
    groups = HierarchicalCPGroups(rank=rank)

    # Only split if the size is a power of 2.
    cp_size_V = min(V, world_size) if is_power_of_2(V) else 1
    cp_size_T = min(T, world_size // cp_size_V) if is_power_of_2(T) else 1
    cp_size_HW = world_size // (cp_size_V * cp_size_T)
    cp_size_THW = cp_size_T * cp_size_HW

    # Rank layout: rank = V_idx * cp_size_THW + T_idx * cp_size_HW + HW_idx
    # Decode current rank's indices
    v_idx = rank // cp_size_THW
    t_idx = (rank % cp_size_THW) // cp_size_HW
    hw_idx = rank % cp_size_HW

    # Create HW groups: ranks with same V_idx and T_idx
    for v in range(cp_size_V):
        for t in range(cp_size_T):
            ranks = tuple(
                [v * cp_size_THW + t * cp_size_HW + hw for hw in range(cp_size_HW)]
            )
            group = dist.new_group(ranks) if dist_initialized else None
            if v_idx == v and t_idx == t:
                groups.HW_ranks = ranks
                groups.HW_group = group
                if single_group_as_none and len(ranks) == 1:
                    groups.HW_group = None

    # Create T groups: ranks with same V_idx and HW_idx
    for v in range(cp_size_V):
        for hw in range(cp_size_HW):
            ranks = tuple(
                [v * cp_size_THW + t * cp_size_HW + hw for t in range(cp_size_T)]
            )
            group = dist.new_group(ranks) if dist_initialized else None
            if v_idx == v and hw_idx == hw:
                groups.T_ranks = ranks
                groups.T_group = group
                if single_group_as_none and len(ranks) == 1:
                    groups.T_group = None

    # Create THW groups: ranks with same V_idx
    for v in range(cp_size_V):
        ranks = tuple(
            [
                v * cp_size_THW + t * cp_size_HW + hw
                for t in range(cp_size_T)
                for hw in range(cp_size_HW)
            ]
        )
        group = dist.new_group(ranks) if dist_initialized else None
        if v_idx == v:
            groups.THW_ranks = ranks
            groups.THW_group = group
            if single_group_as_none and len(ranks) == 1:
                groups.THW_group = None

    # Create V groups: ranks with same T_idx and HW_idx
    for t in range(cp_size_T):
        for hw in range(cp_size_HW):
            ranks = tuple(
                [v * cp_size_THW + t * cp_size_HW + hw for v in range(cp_size_V)]
            )
            group = dist.new_group(ranks) if dist_initialized else None
            if t_idx == t and hw_idx == hw:
                groups.V_ranks = ranks
                groups.V_group = group
                if single_group_as_none and len(ranks) == 1:
                    groups.V_group = None

    # Create VHW groups: ranks with same T_idx
    for t in range(cp_size_T):
        ranks = tuple(
            [
                v * cp_size_THW + t * cp_size_HW + hw
                for v in range(cp_size_V)
                for hw in range(cp_size_HW)
            ]
        )
        group = dist.new_group(ranks) if dist_initialized else None
        if t_idx == t:
            groups.VHW_ranks = ranks
            groups.VHW_group = group
            if single_group_as_none and len(ranks) == 1:
                groups.VHW_group = None

    return groups
