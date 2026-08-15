# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager, ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Protocol

from aiohttp import web
from loguru import logger

from flashdreams.demo.io import OutputDecision, OutputSink, SessionInfo
from flashdreams.demo.outputs import DeferredWebRTCOutputSink
from flashdreams.infra.results import StepResult
from flashdreams.runtime.output import OutputArtifact


class SessionBusyError(RuntimeError):
    """Raised when a second peer tries to open a single-session server."""


class WebRTCSessionManager(Protocol):
    def has_active_session(self) -> bool: ...
    def is_runtime_ready(self) -> bool: ...
    async def preload_runtime(self) -> None: ...
    async def create_answer(
        self, *, offer_sdp: str, offer_type: str
    ) -> dict[str, str]: ...
    async def shutdown(self) -> None: ...


SESSION_MANAGER_KEY = web.AppKey("session_manager", WebRTCSessionManager)
PACKAGE_RESOURCE_STACK_KEY = web.AppKey("package_resource_stack", ExitStack)


def create_webrtc_app(
    *,
    web_dir: Path,
    model_web_dir: Path | None = None,
    session_manager: WebRTCSessionManager,
    request_session_url: str,
    index_filename: str = "request_session.html",
    preload_name: str = "WebRTC",
) -> web.Application:
    app = web.Application()
    app[SESSION_MANAGER_KEY] = session_manager

    async def request_session_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(web_dir / index_filename)

    async def offer(request: web.Request) -> web.StreamResponse:
        try:
            payload = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(reason="Expected JSON offer payload.") from exc

        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="Offer payload must be a JSON object.")

        sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not isinstance(sdp, str) or not sdp:
            raise web.HTTPBadRequest(
                reason="Offer payload must include non-empty 'sdp'."
            )
        if not isinstance(offer_type, str) or not offer_type:
            raise web.HTTPBadRequest(
                reason="Offer payload must include non-empty 'type'."
            )

        manager = request.app[SESSION_MANAGER_KEY]
        try:
            answer_payload = await manager.create_answer(
                offer_sdp=sdp,
                offer_type=offer_type,
            )
        except SessionBusyError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to process WebRTC offer.")
            raise web.HTTPInternalServerError(reason=str(exc)) from exc

        return web.json_response(answer_payload)

    async def healthz(request: web.Request) -> web.StreamResponse:
        manager = request.app[SESSION_MANAGER_KEY]
        return web.json_response(
            {
                "status": "ok",
                "runtime_ready": manager.is_runtime_ready(),
                "session_active": manager.has_active_session(),
            }
        )

    async def ui_config(_: web.Request) -> web.StreamResponse:
        payload: dict[str, object] = {"adapter_module": None}
        if model_web_dir is not None and (model_web_dir / "adapter.js").is_file():
            payload["adapter_module"] = "/model-static/adapter.js?v=model-ui-v2"
        if model_web_dir is not None and (model_web_dir / "adapter.css").is_file():
            payload["model_stylesheet"] = "/model-static/adapter.css?v=model-ui-v2"
        manager_config = getattr(session_manager, "browser_ui_config", None)
        if callable(manager_config):
            payload.update(manager_config())
        return web.json_response(payload)

    async def on_startup(app: web.Application) -> None:
        manager = app[SESSION_MANAGER_KEY]
        logger.info("Preloading {} runtime on startup.", preload_name)
        await manager.preload_runtime()
        logger.info("{} runtime preload complete.", preload_name)
        print(f"Connect via {request_session_url}")

    async def on_shutdown(app: web.Application) -> None:
        manager = app[SESSION_MANAGER_KEY]
        logger.info("Shutting down {} runtime.", preload_name)
        await manager.shutdown()

    app.router.add_get("/request_session", request_session_page)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/ui/config", ui_config)
    app.router.add_static("/static/", web_dir, show_index=False)
    if model_web_dir is not None:
        app.router.add_static("/model-static/", model_web_dir, show_index=False)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


async def close_package_resources(app: web.Application) -> None:
    app[PACKAGE_RESOURCE_STACK_KEY].close()


def create_packaged_webrtc_app(
    *,
    web_resource: Any,
    model_web_resource: Any | None = None,
    session_manager: WebRTCSessionManager,
    request_session_url: str,
    preload_name: str,
    configure_app: Callable[[web.Application], None] | None = None,
    index_filename: str = "request_session.html",
    as_file_fn: Callable[[Any], AbstractContextManager[Path]] = as_file,
    create_app_fn: Callable[..., web.Application] = create_webrtc_app,
    cleanup_callback: Callable[[web.Application], Any] = close_package_resources,
) -> web.Application:
    """Create a WebRTC app from packaged static assets.

    ``importlib.resources.as_file`` can materialize package resources into a
    temporary directory. The returned app owns that context until aiohttp
    cleanup, so demos can serve static browser assets from packages and tests
    can still inspect the materialized directory.
    """
    resource_stack = ExitStack()
    try:
        web_dir = resource_stack.enter_context(as_file_fn(web_resource))
        create_kwargs: dict[str, Any] = {
            "web_dir": web_dir,
            "session_manager": session_manager,
            "preload_name": preload_name,
            "request_session_url": request_session_url,
            "index_filename": index_filename,
        }
        if model_web_resource is not None:
            create_kwargs["model_web_dir"] = resource_stack.enter_context(
                as_file_fn(model_web_resource)
            )
        app = create_app_fn(**create_kwargs)
        if configure_app is not None:
            configure_app(app)
        app[PACKAGE_RESOURCE_STACK_KEY] = resource_stack
        app.on_cleanup.append(cleanup_callback)
    except Exception:
        resource_stack.close()
        raise
    return app


class _BackgroundWebRTCServer:
    """Run one aiohttp application on its own event-loop thread."""

    def __init__(
        self,
        *,
        app: web.Application,
        host: str,
        port: int,
        thread_name: str,
    ) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._ready: Future[None] = Future()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._stop_lock = threading.Lock()
        self._stop_requested = False

    def start(self) -> None:
        """Start serving and raise any bind/startup failure synchronously."""
        self._thread.start()
        try:
            self._ready.result()
        except BaseException:
            self._thread.join()
            raise

    def stop(self) -> None:
        """Stop serving, run aiohttp cleanup, and join the event-loop thread."""
        with self._stop_lock:
            if self._stop_requested:
                return
            self._stop_requested = True
            loop = self._loop
        if loop is not None and self._thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(self._app)
        try:
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, host=self._host, port=self._port)
            loop.run_until_complete(site.start())
            self._ready.set_result(None)
            loop.run_forever()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                logger.exception("WebRTC server event loop failed.")
        finally:
            try:
                loop.run_until_complete(runner.cleanup())
            except Exception:
                logger.exception("WebRTC server cleanup failed.")
            finally:
                loop.close()


class ApplicationWebRTCOutputSink(OutputSink):
    """Bind a regular application output loop to a background WebRTC server."""

    produces_artifacts = False

    def __init__(
        self,
        *,
        application_slug: str,
        host: str,
        port: int,
        peer_timeout_s: float,
        client_liveness_timeout_s: float,
        input_bridge: Any,
    ) -> None:
        self._application_slug = application_slug
        self._host = host
        self._port = port
        self._delegate: DeferredWebRTCOutputSink | None = None
        self._peer_timeout_s = peer_timeout_s
        self._client_liveness_timeout_s = client_liveness_timeout_s
        self._input_bridge = input_bridge
        self._manager: Any | None = None
        self._server: _BackgroundWebRTCServer | None = None
        self._opened = False
        self._closed = False
        self._artifacts: tuple[OutputArtifact, ...] = ()

    def open(self, session_info: SessionInfo) -> None:
        """Start the transport and wait for a negotiated browser peer."""
        if self._opened or self._closed:
            raise RuntimeError("WebRTC application output sink is not reusable.")

        from flashdreams.serving.webrtc.manager import (
            ApplicationWebRTCSessionManager,
        )

        manager = ApplicationWebRTCSessionManager(
            peer_timeout_s=self._peer_timeout_s,
            client_liveness_timeout_s=self._client_liveness_timeout_s,
            input_bridge=self._input_bridge,
        )
        request_host = "127.0.0.1" if self._host in {"0.0.0.0", "::"} else self._host
        if ":" in request_host:
            request_host = f"[{request_host}]"
        app = create_packaged_webrtc_app(
            web_resource=files("flashdreams.serving.webrtc").joinpath("web"),
            session_manager=manager,
            request_session_url=(f"http://{request_host}:{self._port}/request_session"),
            preload_name=self._application_slug,
        )
        server = _BackgroundWebRTCServer(
            app=app,
            host=self._host,
            port=self._port,
            thread_name=f"{self._application_slug}-webrtc",
        )
        delegate = DeferredWebRTCOutputSink(manager.connect_output)
        self._manager = manager
        self._server = server
        self._delegate = delegate
        try:
            server.start()
            delegate.open(session_info)
            self._opened = True
        except BaseException:
            self.close()
            raise

    def begin_generation(self, generation: int) -> None:
        """Begin a generation on the negotiated peer sink."""
        self._required_delegate().begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        """Deliver one application result to the negotiated peer sink."""
        return self._required_delegate().write(result)

    def close(self) -> tuple[OutputArtifact, ...]:
        """Close the peer sink and its background transport exactly once."""
        if self._closed:
            return self._artifacts
        self._closed = True
        self._opened = False
        try:
            if self._delegate is not None:
                self._artifacts = tuple(self._delegate.close())
        finally:
            try:
                if self._manager is not None:
                    self._manager.finish_application()
            finally:
                if self._server is not None:
                    self._server.stop()
        return self._artifacts

    def _required_delegate(self) -> DeferredWebRTCOutputSink:
        if not self._opened or self._delegate is None:
            raise RuntimeError("Cannot write to a closed WebRTC output sink.")
        return self._delegate


__all__ = [
    "ApplicationWebRTCOutputSink",
    "SessionBusyError",
    "WebRTCSessionManager",
    "create_packaged_webrtc_app",
    "create_webrtc_app",
]
