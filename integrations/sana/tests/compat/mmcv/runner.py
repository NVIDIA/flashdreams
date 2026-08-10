# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small ``mmcv.runner`` compatibility layer for SANA inference imports."""

from __future__ import annotations

import torch.distributed as dist


def get_dist_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class DefaultOptimizerConstructor:
    pass


OPTIMIZERS = {}
OPTIMIZER_BUILDERS = {}


def build_optimizer(*_args, **_kwargs):
    raise NotImplementedError(
        "Optimizer construction is not available in the parity harness."
    )


__all__ = [
    "DefaultOptimizerConstructor",
    "OPTIMIZER_BUILDERS",
    "OPTIMIZERS",
    "build_optimizer",
    "get_dist_info",
]
