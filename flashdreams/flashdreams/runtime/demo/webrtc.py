# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared WebRTC demo construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

from aiohttp import web

from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import WebRTCSessionRuntime
from flashdreams.serving.webrtc.server import (
    close_package_resources,
    create_packaged_webrtc_app,
    create_webrtc_app,
)

from .replay import _require_supported_mode
from .spec import DemoSpec, WebRTCAppResources, WebRTCOutputSpec


class ConfiguredWebRTCSessionRuntime(WebRTCSessionRuntime, Protocol):
    """WebRTC runtime that exposes the config used by its session manager."""

    config: Any


class WebRTCIntegration(Protocol):
    """Model-owned hooks required by the WebRTC serving transport."""

    def supported_input_modes(self) -> tuple[str, ...]:
        """Return input modes supported by this WebRTC integration."""
        ...

    def create_runtime(self, spec: DemoSpec) -> ConfiguredWebRTCSessionRuntime:
        """Create the model runtime used for WebRTC generation."""
        ...

    def create_session_manager(
        self,
        *,
        spec: DemoSpec,
        runtime: ConfiguredWebRTCSessionRuntime,
    ) -> BaseWebRTCSessionManager[Any, Any]:
        """Bind the model runtime to the shared WebRTC session manager."""
        ...

    def app_resources(self, spec: DemoSpec) -> WebRTCAppResources:
        """Return model browser assets and optional transport routes."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCDemoRuntimeConfig:
    """Runtime config consumed by the shared WebRTC session manager."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCDemo:
    """Constructed WebRTC demo pieces, before or after serving."""

    runtime: Any
    runtime_config: Any
    session_manager: BaseWebRTCSessionManager[Any, Any]
    app: web.Application | None
    host: str
    port: int


CreateWebRTCApp = Callable[..., web.Application]
RunWebRTCServer = Callable[..., None]


def build_webrtc_demo(
    *,
    spec: DemoSpec,
    integration: WebRTCIntegration,
    create_app: bool = False,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
) -> WebRTCDemo:
    """Build shared WebRTC pieces for a model transport integration."""
    if not isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("build_webrtc_demo requires WebRTCOutputSpec output.")
    _require_supported_mode(
        mode=spec.input_mode,
        supported=integration.supported_input_modes(),
        label="input_mode",
    )

    output = spec.output
    runtime = integration.create_runtime(spec)
    runtime_config = runtime.config
    manager = integration.create_session_manager(
        spec=spec,
        runtime=runtime,
    )
    app = (
        _create_app(
            spec=spec,
            integration=integration,
            session_manager=manager,
            create_app_fn=create_app_fn,
        )
        if create_app
        else None
    )
    return WebRTCDemo(
        runtime=runtime,
        runtime_config=runtime_config,
        session_manager=manager,
        app=app,
        host=output.host,
        port=output.port,
    )


def serve_webrtc_demo(
    *,
    spec: DemoSpec,
    integration: WebRTCIntegration,
    world_rank: int = 0,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> WebRTCDemo:
    """Build and serve a shared WebRTC demo."""
    demo = build_webrtc_demo(
        spec=spec,
        integration=integration,
        create_app=world_rank == 0,
        create_app_fn=create_app_fn,
    )
    server_runner(
        world_rank=world_rank,
        session_manager=demo.session_manager,
        app=demo.app,
        host=demo.host,
        port=demo.port,
    )
    return demo


def _create_app(
    *,
    spec: DemoSpec,
    integration: WebRTCIntegration,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    create_app_fn: CreateWebRTCApp,
) -> web.Application:
    output = spec.output
    if not isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC app creation requires WebRTCOutputSpec output.")
    resources = integration.app_resources(spec)
    if output.web_dir is not None:
        return _build_webrtc_app(
            output=output,
            session_manager=session_manager,
            create_app_fn=create_app_fn,
            preload_name=output.preload_name or resources.preload_name or spec.model_id,
        )
    return create_packaged_webrtc_app(
        web_resource=files("flashdreams.serving.webrtc").joinpath("web"),
        model_web_resource=resources.model_web_resource,
        session_manager=session_manager,
        request_session_url=_request_session_url(output),
        preload_name=output.preload_name or resources.preload_name or spec.model_id,
        configure_app=resources.configure_app,
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
    "ConfiguredWebRTCSessionRuntime",
    "CreateWebRTCApp",
    "RunWebRTCServer",
    "WebRTCDemo",
    "WebRTCDemoRuntimeConfig",
    "WebRTCIntegration",
    "build_webrtc_demo",
    "serve_webrtc_demo",
]
