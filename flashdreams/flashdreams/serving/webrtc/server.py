# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from aiohttp import web

LOGGER = logging.getLogger(__name__)


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


def create_webrtc_app(
    *,
    web_dir: Path,
    session_manager: WebRTCSessionManager,
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
            LOGGER.exception("Failed to process WebRTC offer.")
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

    async def on_startup(app: web.Application) -> None:
        manager = app[SESSION_MANAGER_KEY]
        LOGGER.info("Preloading %s runtime on startup.", preload_name)
        await manager.preload_runtime()
        LOGGER.info("%s runtime preload complete.", preload_name)

    async def on_shutdown(app: web.Application) -> None:
        manager = app[SESSION_MANAGER_KEY]
        LOGGER.info("Shutting down %s runtime.", preload_name)
        await manager.shutdown()

    app.router.add_get("/request_session", request_session_page)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_get("/healthz", healthz)
    app.router.add_static("/static/", web_dir, show_index=False)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app
