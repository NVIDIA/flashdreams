# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest
from aiohttp.test_utils import TestClient, TestServer
from omnidreams.webrtc.server import configure_logging, create_app

from flashdreams.serving.webrtc.server import SessionBusyError

pytestmark = pytest.mark.ci_gpu


class FakeSessionManager:
    def __init__(self) -> None:
        self.answer_payload = {"sdp": "fake-answer-sdp", "type": "answer"}
        self.raise_busy = False
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
            raise SessionBusyError("An Omnidreams session is already active.")
        self.active = True
        return self.answer_payload

    async def shutdown(self) -> None:
        self.active = False
        self.runtime_ready = False


async def _build_client(manager: FakeSessionManager) -> TestClient:
    app = create_app(session_manager=manager)  # ty:ignore[invalid-argument-type]
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
        assert "Omnidreams WebRTC Drive" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_static_js_requests_recvonly_video_transceiver() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/static/request_session.js")
        body = await response.text()
        assert response.status == 200
        assert 'addTransceiver("video", { direction: "recvonly" })' in body
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
async def test_offer_busy_returns_conflict() -> None:
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


def test_configure_logging_suppresses_ice_info_spam() -> None:
    configure_logging()

    assert logging.getLogger("aioice").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("aioice.ice").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("aiortc").getEffectiveLevel() == logging.WARNING
