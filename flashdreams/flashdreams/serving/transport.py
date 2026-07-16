# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common serving transport contract and network protocol adapters."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from typing import Any

from flashdreams.serving.api import (
    SessionCreateRequest,
    SessionSnapshot,
    StreamInput,
    StreamOutput,
)
from flashdreams.serving.service import SessionService


class ServingTransport(ABC):
    """Map a network protocol onto the shared serving lifecycle."""

    def __init__(self, service: SessionService) -> None:
        """Bind a transport to the protocol-neutral session service."""
        self.service = service

    def list_models(self) -> list[dict[str, Any]]:
        """Return model discovery payloads for transport serialization."""
        return [model.to_dict() for model in self.service.list_models()]

    async def preload_models(self) -> None:
        """Start one idle worker for every model exposed by this transport."""
        await self.service.preload_models()

    async def create_session(self, payload: Mapping[str, Any]) -> SessionSnapshot:
        """Validate a transport payload and create a session."""
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string.")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object.")
        lease_seconds = payload.get("lease_seconds")
        if lease_seconds is not None and not isinstance(lease_seconds, (int, float)):
            raise ValueError("lease_seconds must be a number.")
        routing_hint = payload.get("routing_hint")
        if routing_hint is not None and not isinstance(routing_hint, str):
            raise ValueError("routing_hint must be a string.")
        return await self.service.create_session(
            SessionCreateRequest(
                model=model,
                parameters=parameters,
                lease_seconds=lease_seconds,
                routing_hint=routing_hint,
            )
        )

    async def get_session(self, session_id: str) -> SessionSnapshot:
        """Return a session snapshot through the shared lifecycle."""
        return await self.service.get_session(session_id)

    def stream(
        self, session_id: str, payload: Mapping[str, Any]
    ) -> AsyncIterator[StreamOutput]:
        """Validate and dispatch one ordered streaming request."""
        sequence_number = payload.get("sequence_number")
        if not isinstance(sequence_number, int):
            raise ValueError("sequence_number must be an integer.")
        step_input = payload.get("input")
        if not isinstance(step_input, dict):
            raise ValueError("input must be an object.")
        return self.service.stream(
            session_id,
            StreamInput(sequence_number=sequence_number, input=step_input),
        )

    async def delete_session(self, session_id: str) -> SessionSnapshot:
        """Close a session through the shared lifecycle."""
        return await self.service.close_session(session_id)

    @abstractmethod
    async def serve(self, host: str, port: int) -> None:
        """Run the protocol listener until interrupted."""


class WebSocketTransport(ServingTransport):
    """Expose REST session lifecycle endpoints and a WebSocket stream."""

    def __init__(self, service: SessionService) -> None:
        """Bind the transport and initialize lease-reaper state."""
        super().__init__(service)
        self._lease_reaper: asyncio.Task[None] | None = None

    def create_app(self) -> Any:
        """Build the aiohttp application without starting a listener."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/healthz", self._health)
        app.router.add_get("/v1/models", self._list_models)
        app.router.add_post("/v1/sessions", self._create_session)
        app.router.add_get("/v1/sessions/{session_id}", self._get_session)
        app.router.add_get("/v1/sessions/{session_id}/stream", self._stream_session)
        app.router.add_delete("/v1/sessions/{session_id}", self._delete_session)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def serve(self, host: str, port: int) -> None:
        """Run the aiohttp REST and WebSocket listener."""
        from aiohttp import web

        runner = web.AppRunner(self.create_app())
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        print(f"FlashDreams server listening on http://{host}:{port}", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    async def _health(self, request: Any) -> Any:
        from aiohttp import web

        del request
        return web.json_response({"status": "ok"})

    async def _list_models(self, request: Any) -> Any:
        from aiohttp import web

        del request
        return web.json_response({"object": "list", "data": self.list_models()})

    async def _create_session(self, request: Any) -> Any:
        from aiohttp import web

        try:
            snapshot = await self.create_session(await request.json())
            return web.json_response(snapshot.to_dict(), status=201)
        except Exception as exc:  # noqa: BLE001 - HTTP error boundary
            return self._error_response(exc)

    async def _get_session(self, request: Any) -> Any:
        from aiohttp import web

        try:
            snapshot = await self.get_session(request.match_info["session_id"])
            return web.json_response(snapshot.to_dict())
        except Exception as exc:  # noqa: BLE001 - HTTP error boundary
            return self._error_response(exc)

    async def _delete_session(self, request: Any) -> Any:
        from aiohttp import web

        try:
            snapshot = await self.delete_session(request.match_info["session_id"])
            return web.json_response(snapshot.to_dict())
        except Exception as exc:  # noqa: BLE001 - HTTP error boundary
            return self._error_response(exc)

    async def _stream_session(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        socket = web.WebSocketResponse(heartbeat=30.0)
        await socket.prepare(request)
        session_id = request.match_info["session_id"]
        async for message in socket:
            if message.type is WSMsgType.TEXT:
                try:
                    payload = json.loads(message.data)
                    async for output in self.stream(session_id, payload):
                        await socket.send_json(output.to_dict())
                except Exception as exc:  # noqa: BLE001 - socket error boundary
                    await socket.send_json(self._error_payload(exc))
            elif message.type is WSMsgType.ERROR:
                break
        return socket

    async def _startup(self, app: Any) -> None:
        del app
        self._lease_reaper = asyncio.create_task(self._reap_expired_sessions())

    async def _cleanup(self, app: Any) -> None:
        del app
        if self._lease_reaper is not None:
            self._lease_reaper.cancel()
            with suppress(asyncio.CancelledError):
                await self._lease_reaper
        await self.service.close()

    async def _reap_expired_sessions(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            await self.service.expire_sessions()

    @staticmethod
    def _error_payload(exc: Exception) -> dict[str, Any]:
        return {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        }

    @classmethod
    def _error_response(cls, exc: Exception) -> Any:
        from aiohttp import web

        if isinstance(exc, KeyError):
            status = 404
        elif isinstance(exc, (ValueError, RuntimeError)):
            status = 409 if isinstance(exc, RuntimeError) else 400
        else:
            status = 500
        return web.json_response(cls._error_payload(exc), status=status)


class WebRTCTransport(WebSocketTransport):
    """Add session-scoped WebRTC offer negotiation to the HTTP API."""

    def create_app(self) -> Any:
        """Build the HTTP, WebSocket, and WebRTC signaling application."""
        app = super().create_app()
        app.router.add_post(
            "/v1/sessions/{session_id}/webrtc/offer", self._webrtc_offer
        )
        return app

    async def _webrtc_offer(self, request: Any) -> Any:
        from aiohttp import web

        try:
            answer = await self.service.create_webrtc_answer(
                request.match_info["session_id"], await request.json()
            )
            return web.json_response(dict(answer))
        except Exception as exc:  # noqa: BLE001 - HTTP error boundary
            return self._error_response(exc)


class GRPCTransport(ServingTransport):
    """Define the gRPC mapping while keeping protobuf bindings model-independent."""

    async def serve(self, host: str, port: int) -> None:
        """Run a gRPC listener when generated bindings are installed.

        Raises:
            RuntimeError: Core protobuf bindings have not been generated.
        """
        del host, port
        raise RuntimeError(
            "gRPC network bindings are not packaged yet; subclass GRPCTransport "
            "and map the shared lifecycle methods to generated protobuf handlers."
        )
