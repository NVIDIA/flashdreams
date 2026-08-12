# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared WebRTC demo construction."""

from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from aiohttp import web

from flashdreams.runtime.demo.spec import WebRTCAppResources, WebRTCOutputSpec
from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import (
    close_package_resources,
    create_packaged_webrtc_app,
    create_webrtc_app,
)

CreateWebRTCApp = Callable[..., web.Application]
RunWebRTCServer = Callable[..., None]


def serve_webrtc_demo(
    *,
    output: WebRTCOutputSpec,
    model_id: str,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    app_resources: WebRTCAppResources,
    world_rank: int = 0,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> web.Application | None:
    """Serve a prepared model WebRTC runtime through the shared transport."""
    app = (
        _create_app(
            output=output,
            model_id=model_id,
            app_resources=app_resources,
            session_manager=session_manager,
            create_app_fn=create_app_fn,
        )
        if world_rank == 0
        else None
    )
    server_runner(
        world_rank=world_rank,
        session_manager=session_manager,
        app=app,
        host=output.host,
        port=output.port,
    )
    return app


def _create_app(
    *,
    output: WebRTCOutputSpec,
    model_id: str,
    app_resources: WebRTCAppResources,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    create_app_fn: CreateWebRTCApp,
) -> web.Application:
    if output.web_dir is not None:
        return _build_webrtc_app(
            output=output,
            session_manager=session_manager,
            create_app_fn=create_app_fn,
            preload_name=output.preload_name or app_resources.preload_name or model_id,
        )
    return create_packaged_webrtc_app(
        web_resource=files("flashdreams.serving.webrtc").joinpath("web"),
        model_web_resource=app_resources.model_web_resource,
        session_manager=session_manager,
        request_session_url=_request_session_url(output),
        preload_name=output.preload_name or app_resources.preload_name or model_id,
        configure_app=app_resources.configure_app,
        create_app_fn=create_app_fn,
        cleanup_callback=close_package_resources,
    )


def _build_webrtc_app(
    *,
    output: WebRTCOutputSpec,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    create_app_fn: CreateWebRTCApp,
    preload_name: str,
) -> web.Application:
    if output.web_dir is None:
        raise ValueError("WebRTC app creation requires output.web_dir.")
    return create_app_fn(
        web_dir=Path(output.web_dir),
        session_manager=session_manager,
        request_session_url=_request_session_url(output),
        preload_name=preload_name,
    )


def _request_session_url(output: WebRTCOutputSpec) -> str:
    host = "127.0.0.1" if output.host in {"0.0.0.0", "::"} else output.host
    return f"http://{host}:{output.port}{output.request_session_path}"


__all__ = [
    "CreateWebRTCApp",
    "RunWebRTCServer",
    "serve_webrtc_demo",
]
