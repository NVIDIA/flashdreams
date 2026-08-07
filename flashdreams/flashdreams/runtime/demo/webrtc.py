# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared WebRTC demo construction."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any

from aiohttp import web

from flashdreams.serving.webrtc.bootstrap import run_webrtc_server
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import (
    create_packaged_webrtc_app,
    create_webrtc_app,
)

from .replay import _require_supported_mode
from .spec import DemoAdapter, DemoSpec, WebRTCOutputSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCDemoRuntimeConfig:
    """Runtime config consumed by the shared WebRTC session manager."""

    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float


ManagerResetHook = Callable[[Any, Any], Awaitable[None] | None]
PendingInputHook = Callable[[], Any]
ClearPendingInputHook = Callable[[], None]
SetPendingInputHook = Callable[[Any, Any], None]
ChunkDoneExtraHook = Callable[[Any, Any], Mapping[str, Any]]
PeerConnectionHook = Callable[[Any], None]
SdpHook = Callable[[str], None]
ConfigureWebRTCApp = Callable[[web.Application], None]


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCManagerOptions:
    """Declarative options for the shared demo WebRTC session manager."""

    model_name: str | None = None
    busy_message: str | None = None
    warmup_label: str | None = None
    runtime_error_types: tuple[type[Exception], ...] = (RuntimeError,)
    close_session_on_generation_error: bool = False
    supported_keys: AbstractSet[str] | None = None
    peek_pending_session_input: PendingInputHook | None = None
    clear_pending_session_input: ClearPendingInputHook | None = None
    set_pending_session_input: SetPendingInputHook | None = None
    reset_runtime_for_session: ManagerResetHook | None = None
    chunk_done_extra: ChunkDoneExtraHook | None = None
    register_extra_peer_handlers: PeerConnectionHook | None = None
    on_offer_received: SdpHook | None = None
    on_answer_created: SdpHook | None = None

    def __post_init__(self) -> None:
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("WebRTCManagerOptions.model_name must be non-empty.")
        if self.busy_message is not None and not self.busy_message.strip():
            raise ValueError("WebRTCManagerOptions.busy_message must be non-empty.")
        if self.warmup_label is not None and not self.warmup_label.strip():
            raise ValueError("WebRTCManagerOptions.warmup_label must be non-empty.")
        if not self.runtime_error_types:
            raise ValueError(
                "WebRTCManagerOptions.runtime_error_types cannot be empty."
            )
        for error_type in self.runtime_error_types:
            if not issubclass(error_type, Exception):
                raise TypeError(
                    "WebRTCManagerOptions.runtime_error_types must contain "
                    "Exception subclasses."
                )
        if self.supported_keys is not None:
            object.__setattr__(self, "supported_keys", frozenset(self.supported_keys))


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCAppExtension:
    """Model-owned additions to a shared demo WebRTC app."""

    web_resource: Any | None = None
    web_dir: str | Path | None = None
    preload_name: str | None = None
    index_filename: str = "request_session.html"
    configure_app: ConfigureWebRTCApp | None = None

    def __post_init__(self) -> None:
        if self.web_resource is not None and self.web_dir is not None:
            raise ValueError(
                "WebRTCAppExtension accepts either web_resource or web_dir, not both."
            )
        if self.web_dir is not None:
            object.__setattr__(self, "web_dir", Path(self.web_dir))
        if not self.index_filename.strip():
            raise ValueError("WebRTCAppExtension.index_filename must be non-empty.")
        if self.preload_name is not None and not self.preload_name.strip():
            raise ValueError("WebRTCAppExtension.preload_name must be non-empty.")


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
        manager_options: WebRTCManagerOptions | None = None,
    ) -> None:
        options = manager_options or WebRTCManagerOptions()
        self._demo_manager_options = options
        self._demo_model_name = options.model_name or model_name
        if options.busy_message is not None:
            self._busy_message = options.busy_message
        if options.warmup_label is not None:
            self._warmup_label = options.warmup_label
        self._runtime_error_types = options.runtime_error_types
        self._close_session_on_generation_error = (
            options.close_session_on_generation_error
        )
        self._resampler_supported_keys = options.supported_keys
        super().__init__(
            runtime=runtime,
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )

    def _model_name(self) -> str:
        return self._demo_model_name

    def _peek_pending_session_input(self) -> Any:
        hook = self._demo_manager_options.peek_pending_session_input
        return None if hook is None else hook()

    def _clear_pending_session_input(self) -> None:
        hook = self._demo_manager_options.clear_pending_session_input
        if hook is not None:
            hook()

    def set_pending_session_input(self, session_input: Any) -> None:
        hook = self._demo_manager_options.set_pending_session_input
        if hook is None:
            raise NotImplementedError(
                "This WebRTC demo does not accept pending session input."
            )
        hook(self, session_input)

    async def _reset_runtime_for_session(self, session_input: Any) -> None:
        hook = self._demo_manager_options.reset_runtime_for_session
        if hook is None:
            await super()._reset_runtime_for_session(session_input)
            return
        result = hook(self._runtime, session_input)
        if inspect.isawaitable(result):
            await result

    def _chunk_done_extra(self) -> dict[str, Any]:
        hook = self._demo_manager_options.chunk_done_extra
        if hook is None:
            return {}
        return dict(hook(self._runtime, self.runtime_config))

    def _register_extra_peer_handlers(self, peer_connection: Any) -> None:
        hook = self._demo_manager_options.register_extra_peer_handlers
        if hook is not None:
            hook(peer_connection)

    def _on_offer_received(self, offer_sdp: str) -> None:
        hook = self._demo_manager_options.on_offer_received
        if hook is not None:
            hook(offer_sdp)

    def _on_answer_created(self, answer_sdp: str) -> None:
        hook = self._demo_manager_options.on_answer_created
        if hook is not None:
            hook(answer_sdp)


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

    manager_options = _create_manager_options(
        spec=spec,
        adapter=adapter,
        runtime=runtime,
        runtime_config=runtime_config,
    )
    return SharedDemoWebRTCSessionManager(
        model_name=manager_options.model_name or spec.model_id,
        runtime=runtime,
        runtime_config=runtime_config,
        fps=fps,
        client_liveness_timeout_s=client_liveness_timeout_s,
        manager_options=manager_options,
    )


def _create_manager_options(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    runtime: Any,
    runtime_config: Any,
) -> WebRTCManagerOptions:
    factory = getattr(adapter, "create_webrtc_manager_options", None)
    if not callable(factory):
        return WebRTCManagerOptions()
    options = factory(spec=spec, runtime=runtime, runtime_config=runtime_config)
    if not isinstance(options, WebRTCManagerOptions):
        raise TypeError(
            "create_webrtc_manager_options must return WebRTCManagerOptions."
        )
    return options


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
    extension = _create_app_extension(
        spec=spec,
        adapter=adapter,
        session_manager=session_manager,
        request_session_url=_request_session_url(output),
    )
    if extension is not None:
        return _build_webrtc_app_from_extension(
            output=output,
            extension=extension,
            session_manager=session_manager,
            request_session_url=_request_session_url(output),
            create_app_fn=create_app_fn,
            fallback_preload_name=output.preload_name or spec.model_id,
        )

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


def _create_app_extension(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    request_session_url: str,
) -> WebRTCAppExtension | None:
    factory = getattr(adapter, "create_webrtc_app_extension", None)
    if not callable(factory):
        return None
    extension = factory(
        spec=spec,
        session_manager=session_manager,
        request_session_url=request_session_url,
    )
    if extension is None:
        return None
    if not isinstance(extension, WebRTCAppExtension):
        raise TypeError("create_webrtc_app_extension must return WebRTCAppExtension.")
    return extension


def _build_webrtc_app_from_extension(
    *,
    output: WebRTCOutputSpec,
    extension: WebRTCAppExtension,
    session_manager: BaseWebRTCSessionManager[Any, Any],
    request_session_url: str,
    create_app_fn: CreateWebRTCApp,
    fallback_preload_name: str,
) -> web.Application:
    preload_name = extension.preload_name or fallback_preload_name
    if extension.web_resource is not None:
        return create_packaged_webrtc_app(
            web_resource=extension.web_resource,
            session_manager=session_manager,
            request_session_url=request_session_url,
            preload_name=preload_name,
            configure_app=extension.configure_app,
            index_filename=extension.index_filename,
            create_app_fn=create_app_fn,
        )

    web_dir = extension.web_dir or output.web_dir
    if web_dir is None:
        raise ValueError("WebRTC app extension requires web_resource or web_dir.")
    app = create_app_fn(
        web_dir=Path(web_dir),
        session_manager=session_manager,
        request_session_url=request_session_url,
        preload_name=preload_name,
        index_filename=extension.index_filename,
    )
    if extension.configure_app is not None:
        extension.configure_app(app)
    return app


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
    "WebRTCAppExtension",
    "WebRTCDemo",
    "WebRTCManagerOptions",
    "WebRTCDemoRuntimeConfig",
    "build_webrtc_demo",
    "serve_webrtc_demo",
]
