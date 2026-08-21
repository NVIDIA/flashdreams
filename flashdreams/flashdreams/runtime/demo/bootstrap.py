# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process bootstrap for demo applications."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedDemoContext:
    """CUDA/distributed launch context for a demo process."""

    device: torch.device
    world_rank: int
    world_size: int


def configure_logging(*, world_rank: int | None = None) -> None:
    from flashdreams.core.distributed import configure_loguru_for_distributed

    configure_loguru_for_distributed(world_rank=world_rank)
    for logger_name in ("aioice", "aioice.ice", "aiortc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _distributed_init() -> None:
    from flashdreams.core.distributed import init as distributed_init

    distributed_init()


def initialize_cuda_distributed(
    *,
    default_device: str | torch.device = "cuda:0",
    distributed_init_fn: Callable[[], object] | None = None,
    configure_logging_fn: Callable[..., None] = configure_logging,
    torch_module: Any = torch,
    dist_module: Any = dist,
) -> DistributedDemoContext:
    """Initialize CUDA and optional torch.distributed for demo serving."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for inference in the demo server.")

    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if has_rank != has_world_size:
        raise RuntimeError(
            "Distributed launch expects both RANK and WORLD_SIZE to be set."
        )

    distributed_launch = has_rank and has_world_size
    if distributed_launch:
        if distributed_init_fn is None:
            distributed_init_fn = _distributed_init
        distributed_init_fn()
        world_rank = dist_module.get_rank()
        world_size = dist_module.get_world_size()
    else:
        world_rank = 0
        world_size = 1

    device_count = torch_module.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA device count must be >= 1 for inference.")
    if distributed_launch:
        local_rank = world_rank % device_count
        torch_device = torch_module.device(f"cuda:{local_rank}")
    else:
        torch_device = torch_module.device(default_device)
        if torch_device.type != "cuda":
            raise RuntimeError(
                f"CUDA device is required for inference, got {torch_device}."
            )
        if torch_device.index is None:
            torch_device = torch_module.device("cuda:0")
    torch_module.cuda.set_device(torch_device)
    configure_logging_fn(world_rank=world_rank)
    return DistributedDemoContext(
        device=torch_device,
        world_rank=world_rank,
        world_size=world_size,
    )


def cleanup_cuda_distributed(
    *,
    world_rank: int | None = None,
    synchronize_distributed: bool = True,
    torch_module: Any = torch,
    dist_module: Any = dist,
) -> None:
    """Release process-level CUDA and distributed state owned by demo serving."""
    gc.collect()
    cuda = torch_module.cuda
    if cuda.is_available():
        cuda.empty_cache()
        cuda.synchronize()

    is_available = getattr(dist_module, "is_available", None)
    if callable(is_available) and not is_available():
        return
    if not dist_module.is_initialized():
        return
    if synchronize_distributed:
        dist_module.barrier()
    if world_rank is None:
        logging.getLogger(__name__).info("Destroying process group.")
    else:
        logging.getLogger(__name__).info(
            "[Rank %s] Destroying process group.",
            world_rank,
        )
    dist_module.destroy_process_group()


__all__ = [
    "DistributedDemoContext",
    "cleanup_cuda_distributed",
    "configure_logging",
    "initialize_cuda_distributed",
]
