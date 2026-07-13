# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process bootstrap for the single-session WebRTC demo servers."""

from __future__ import annotations

import gc
import logging
import multiprocessing
import time
from typing import Protocol

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger

from flashdreams.core.distributed import configure_loguru_for_distributed

_CHILD_PROCESS_TERMINATION_TIMEOUT_S = 5.0


class WebRTCServerLifecycle(Protocol):
    """Rank-coordination surface the serve loop needs from a session manager."""

    def send_exit_signal(self) -> None: ...
    def wait_for_termination(self) -> None: ...


def configure_logging(*, world_rank: int | None = None) -> None:
    configure_loguru_for_distributed(world_rank=world_rank)
    for logger_name in ("aioice", "aioice.ice", "aiortc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _terminate_child_processes() -> None:
    """Terminate and reap subprocesses still owned by this server rank."""
    children = multiprocessing.active_children()
    if not children:
        return

    child_summary = ", ".join(f"{child.name} (pid={child.pid})" for child in children)
    logger.warning("Terminating child processes during shutdown: {}", child_summary)
    for child in children:
        try:
            child.terminate()
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _CHILD_PROCESS_TERMINATION_TIMEOUT_S
    for child in children:
        child.join(timeout=max(0.0, deadline - time.monotonic()))

    survivors = [child for child in children if child.is_alive()]
    for child in survivors:
        logger.warning(
            "Child process {} (pid={}) did not terminate; killing it.",
            child.name,
            child.pid,
        )
        child.kill()
    for child in survivors:
        child.join()


def run_webrtc_server(
    *,
    world_rank: int,
    session_manager: WebRTCServerLifecycle,
    app: web.Application | None,
    host: str,
    port: int,
) -> None:
    """Serve on rank 0, idle on worker ranks, then tear the runtime down."""
    if world_rank == 0 and app is None:
        raise ValueError("Rank 0 requires an aiohttp app to serve.")

    try:
        if world_rank == 0:
            assert app is not None
            web.run_app(app, host=host, port=port)
        else:
            try:
                session_manager.wait_for_termination()
            except KeyboardInterrupt:
                logger.warning("Worker rank interrupted, shutting down.")
    finally:
        try:
            _terminate_child_processes()
        finally:
            if world_rank == 0:
                session_manager.send_exit_signal()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()
        logger.info("[Rank {}] Destroying process group", world_rank)
        dist.destroy_process_group()
