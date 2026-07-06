# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process bootstrap for the single-session WebRTC demo servers."""

from __future__ import annotations

import gc
import logging
from typing import Protocol

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger

from flashdreams.core.distributed import configure_loguru_for_distributed


class WebRTCServerLifecycle(Protocol):
    """Rank-coordination surface the serve loop needs from a session manager."""

    def send_exit_signal(self) -> None: ...
    def wait_for_termination(self) -> None: ...


def configure_logging(*, world_rank: int | None = None) -> None:
    configure_loguru_for_distributed(world_rank=world_rank)
    for logger_name in ("aioice", "aioice.ice", "aiortc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def run_webrtc_server(
    *,
    world_rank: int,
    session_manager: WebRTCServerLifecycle,
    app: web.Application | None,
    host: str,
    port: int,
) -> None:
    """Serve on rank 0, idle on worker ranks, then tear the runtime down."""
    if world_rank == 0:
        if app is None:
            raise ValueError("Rank 0 requires an aiohttp app to serve.")
        try:
            web.run_app(app, host=host, port=port)
        finally:
            session_manager.send_exit_signal()
    else:
        try:
            session_manager.wait_for_termination()
        except KeyboardInterrupt:
            logger.warning("Worker rank interrupted, shutting down.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()
        logger.info("[Rank {}] Destroying process group", world_rank)
        dist.destroy_process_group()
