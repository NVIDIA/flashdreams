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

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer
from lingbot.webrtc.server import create_app
from lingbot.webrtc.session import SessionBusyError

pytestmark = pytest.mark.ci_gpu


class FakeSessionManager:
    def __init__(self) -> None:
        self.answer_payload = {"sdp": "fake-answer-sdp", "type": "answer"}
        self.raise_busy = False
        self.close_calls = 0
        self.preload_calls = 0
        self.offers: list[tuple[str, str]] = []
        self.active = False
        self.runtime_ready = False

    def has_active_session(self) -> bool:
        return self.active

    def is_runtime_ready(self) -> bool:
        return self.runtime_ready

    async def preload_runtime(self) -> None:
        self.preload_calls += 1
        self.runtime_ready = True

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        self.offers.append((offer_sdp, offer_type))
        if self.raise_busy:
            raise SessionBusyError("A Lingbot session is already active.")
        self.active = True
        return self.answer_payload

    async def close_active_session(self) -> None:
        self.close_calls += 1
        self.active = False

    async def shutdown(self) -> None:
        await self.close_active_session()
        self.runtime_ready = False


async def _build_client(manager: FakeSessionManager) -> TestClient:
    app = create_app(
        session_manager=manager,
        request_session_url="http://127.0.0.1:8080/request_session",
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_request_session_serves_html() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        assert manager.preload_calls == 1
        response = await client.get("/request_session")
        body = await response.text()
        assert response.status == 200
        assert "Lingbot WebRTC Viewer" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_offer_returns_answer_payload() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.post(
            "/api/webrtc/offer",
            json={"sdp": "offer-sdp", "type": "offer"},
        )
        payload = await response.json()
        assert response.status == 200
        assert payload == manager.answer_payload
        assert manager.offers == [("offer-sdp", "offer")]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_healthz_reports_runtime_ready() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/healthz")
        payload = await response.json()
        assert response.status == 200
        assert payload["runtime_ready"] is True
        assert payload["session_active"] is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_offer_requires_sdp_and_type() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.post("/api/webrtc/offer", json={"type": "offer"})
        assert response.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_offer_returns_conflict_when_session_busy() -> None:
    manager = FakeSessionManager()
    manager.raise_busy = True
    client = await _build_client(manager)
    try:
        response = await client.post(
            "/api/webrtc/offer",
            json={"sdp": "offer-sdp", "type": "offer"},
        )
        assert response.status == 409
    finally:
        await client.close()
