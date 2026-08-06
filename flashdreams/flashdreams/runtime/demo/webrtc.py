# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared WebRTC demo construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web

from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import create_webrtc_app

from .replay import _require_supported_mode
from .spec import DemoAdapter, DemoSpec, WebRTCOutputSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCDemoRuntimeConfig:
    """Runtime config consumed by the shared WebRTC session manager."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


class SharedDemoWebRTCSessionManager(BaseWebRTCSessionManager[Any, Any]):
    """Generic session manager wrapper for demo adapters."""

    def __init__(
        self,
        *,
        model_name: str,
        runtime: Any,
        runtime_config: Any,
        fps: int,
        client_liveness_timeout_s: float,
    ) -> None:
        self._demo_model_name = model_name
        super().__init__(
            runtime=runtime,
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )

    def _model_name(self) -> str:
        return self._demo_model_name


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
    adapter: DemoAdapter,
    create_app: bool = False,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
) -> WebRTCDemo:
    """Build shared WebRTC manager/app pieces for a demo adapter runtime."""
    if not isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("build_webrtc_demo requires WebRTCOutputSpec output.")
    _require_supported_mode(
        mode=spec.input_mode,
        supported=adapter.supported_input_modes(),
        label="input_mode",
    )
    _require_supported_mode(
        mode=spec.output.mode,
        supported=adapter.supported_output_modes(),
        label="output.mode",
    )

    output = spec.output
    runtime = adapter.create_webrtc_runtime(spec)
    runtime_config = _create_runtime_config(
        spec=spec,
        adapter=adapter,
        runtime=runtime,
    )
    manager = _create_session_manager(
        spec=spec,
        adapter=adapter,
        runtime=runtime,
        runtime_config=runtime_config,
        fps=output.fps,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
    )
    app = (
        _create_app(
            spec=spec,
            adapter=adapter,
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
    adapter: DemoAdapter,
    world_rank: int = 0,
    create_app_fn: CreateWebRTCApp = create_webrtc_app,
    server_runner: RunWebRTCServer = run_webrtc_server,
) -> WebRTCDemo:
    """Build and serve a shared WebRTC demo."""
    demo = build_webrtc_demo(
        spec=spec,
        adapter=adapter,
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


def _create_runtime_config(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    runtime: Any,
) -> Any:
    factory = getattr(adapter, "create_webrtc_runtime_config", None)
    if callable(factory):
        return factory(spec=spec, runtime=runtime)

    runtime_config = getattr(runtime, "config", None)
    if _looks_like_webrtc_runtime_config(runtime_config):
        return runtime_config

    output = spec.output
    if not isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC runtime config creation requires WebRTCOutputSpec.")
    return WebRTCDemoRuntimeConfig(
        video_width=output.video_width,
        video_height=output.video_height,
        warmup_chunks=output.warmup_chunks,
        warmup_timeout_s=output.warmup_timeout_s,
    )


def _looks_like_webrtc_runtime_config(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in (
            "video_width",
            "video_height",
            "warmup_chunks",
            "warmup_timeout_s",
        )
    )


def _create_session_manager(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    runtime: Any,
    runtime_config: Any,
    fps: int,
    client_liveness_timeout_s: float,
) -> BaseWebRTCSessionManager[Any, Any]:
    factory = getattr(adapter, "create_webrtc_session_manager", None)
    if callable(factory):
        return factory(
            spec=spec,
            runtime=runtime,
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )

    return SharedDemoWebRTCSessionManager(
        model_name=spec.model_id,
        runtime=runtime,
        runtime_config=runtime_config,
        fps=fps,
        client_liveness_timeout_s=client_liveness_timeout_s,
    )


def _create_app(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    create_app_fn: CreateWebRTCApp,
) -> web.Application:
    output = spec.output
    if not isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC app creation requires WebRTCOutputSpec output.")
    factory = getattr(adapter, "create_webrtc_app", None)
    if callable(factory):
        return factory(
            spec=spec,
            session_manager=session_manager,
            request_session_url=_request_session_url(output),
        )
    return _build_webrtc_app(
        output=output,
        session_manager=session_manager,
        create_app_fn=create_app_fn,
        preload_name=output.preload_name or spec.model_id,
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
    "SharedDemoWebRTCSessionManager",
    "WebRTCDemo",
    "WebRTCDemoRuntimeConfig",
    "build_webrtc_demo",
    "serve_webrtc_demo",
]
