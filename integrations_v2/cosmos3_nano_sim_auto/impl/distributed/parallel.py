# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiview checkpoint and environment defaults for the core process mesh."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
from cosmos_interactive._spec import SPEC

from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.core.distributed.parallel import init_parallel as init_core_parallel


def owned_views(parallel: ParallelContext | None, num_views: int) -> range:
    """Cameras this rank encodes: round-robin, camera ``i`` to rank ``i % world_size``.

    The one statement of the rule. Callers that read, encode, or broadcast
    per-camera data go through it so the ranks cannot disagree about who owns what.
    """
    if parallel is None:
        return range(num_views)
    return range(parallel.rank, num_views, parallel.world_size)


def gather_views(
    local: Sequence[torch.Tensor], *, num_views: int, parallel: ParallelContext
) -> list[torch.Tensor] | None:
    """This rank's cameras, in :func:`owned_views` order, sent to rank 0.

    Rank 0 gets every camera back in rig order and the others ``None``. Rank 0
    owns camera 0 on any mesh, so its own first tensor says what every camera's
    looks like and no shape is exchanged. A rank owning nothing sends nothing.
    """
    owned = owned_views(parallel, num_views)
    if not parallel.is_main:
        for tensor in local:
            dist.send(tensor.contiguous(), dst=0)
        return None
    views: list[torch.Tensor | None] = [None] * num_views
    for index, view in enumerate(owned):
        views[view] = local[index]
    for view in range(num_views):
        if views[view] is None:
            received = torch.empty_like(local[0])
            dist.recv(received, src=view % parallel.world_size)
            views[view] = received
    return [view for view in views if view is not None]


def init_parallel(tensor_parallel: int | None = None) -> ParallelContext:
    """Initialize the core mesh using Multiview's checkpoint and launch options.

    Unset, the tensor size is ``gcd(world_size, 2)``: on 8 H100s at four views
    the context axis is faster than the tensor axis, and a tensor size of two
    keeps nearly all of that speed at half the per-rank weight memory.
    """
    if tensor_parallel is None:
        raw = os.environ.get("COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tensor_parallel = int(raw) if raw else math.gcd(world_size, 2)
    return init_core_parallel(tensor_parallel, head_groups=SPEC.llm.num_key_value_heads)
