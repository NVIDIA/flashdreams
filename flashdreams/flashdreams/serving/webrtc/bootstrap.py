# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process bootstrap for the single-session WebRTC demo servers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger

from flashdreams.runtime.demo.bootstrap import (
    DistributedDemoContext as WebRTCDistributedContext,
)
from flashdreams.runtime.demo.bootstrap import (
    cleanup_cuda_distributed,
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.serving.webrtc.runtime import WebRTCServerLifecycle


def run_webrtc_server(
    *,
    world_rank: int,
    session_manager: WebRTCServerLifecycle,
    app: web.Application | None,
    host: str,
    port: int,
) -> None:
    """Serve on rank 0, idle on worker ranks, then tear the runtime down."""
    primary_error: BaseException | None = None
    completed = False
    try:
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
        completed = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _cleanup_webrtc_process(
            session_manager=session_manager,
            world_rank=world_rank,
            synchronize_distributed=completed,
            primary_error=primary_error,
        )


def _cleanup_webrtc_process(
    *,
    session_manager: WebRTCServerLifecycle,
    world_rank: int,
    synchronize_distributed: bool,
    primary_error: BaseException | None,
) -> None:
    errors: list[BaseException] = []
    if primary_error is not None:
        _record_webrtc_process_cleanup_error(
            errors,
            _shutdown_webrtc_session_manager,
            session_manager,
        )
    _record_webrtc_process_cleanup_error(
        errors,
        cleanup_cuda_distributed,
        world_rank=world_rank,
        synchronize_distributed=synchronize_distributed,
        torch_module=torch,
        dist_module=dist,
    )
    if primary_error is not None:
        _record_webrtc_process_cleanup_notes(primary_error, errors)
        return
    _raise_first_webrtc_process_cleanup_error(errors)


def _shutdown_webrtc_session_manager(session_manager: WebRTCServerLifecycle) -> None:
    shutdown = getattr(session_manager, "shutdown", None)
    if not callable(shutdown):
        return
    result = shutdown()
    if inspect.isawaitable(result):
        asyncio.run(result)


def _record_webrtc_process_cleanup_error(
    errors: list[BaseException],
    cleanup: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        cleanup(*args, **kwargs)
    except BaseException as cleanup_error:
        errors.append(cleanup_error)


def _raise_first_webrtc_process_cleanup_error(errors: list[BaseException]) -> None:
    if not errors:
        return
    first = errors[0]
    _record_webrtc_process_cleanup_notes(first, errors[1:])
    raise first


def _record_webrtc_process_cleanup_notes(
    primary_error: BaseException,
    errors: list[BaseException],
) -> None:
    add_note = getattr(primary_error, "add_note", None)
    for cleanup_error in errors:
        if callable(add_note):
            add_note(f"Additional WebRTC process cleanup error: {cleanup_error!r}")
