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
        assert "Omnidreams WebRTC Drive" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_session_uses_lingbot_aligned_viewer_shell() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/request_session")
        body = await response.text()
        assert response.status == 200
        assert 'class="brandOverlay"' in body
        assert "FlashDreams" in body
        assert "/assets/logo/horizontal-light.svg" in body
        assert 'class="statusCard overlayPanel"' in body
        assert 'class="controlCard overlayPanel"' in body
        assert 'class="logCard overlayPanel"' in body
        assert "Connect Session" in body
        assert 'id="logState"' in body
        assert "World Model" in body
        for key in ("w", "a", "s", "d"):
            assert f'data-control-key="{key}"' in body
        for key in ("q", "e", "i", "j", "k", "l"):
            assert f'data-control-key="{key}"' not in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_session_includes_idle_animation_canvas() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/request_session")
        body = await response.text()
        assert response.status == 200
        assert (
            '<canvas id="idleCanvas" class="idleCanvas" aria-hidden="true"></canvas>'
            in body
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shared_flashdreams_brand_asset_is_served() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/assets/logo/horizontal-light.svg")
        assert response.status == 200
        assert response.content_type == "image/svg+xml"
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
async def test_static_js_keeps_omnidreams_controls_and_lingbot_status_helpers() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/static/request_session.js")
        body = await response.text()
        assert response.status == 200
        assert 'const allowedKeys = new Set(["w", "a", "s", "d"])' in body
        assert 'const logState = document.getElementById("logState")' in body
        assert 'logState.textContent = state === "idle" ? "Waiting" : message' in body
        assert "eventLog.prepend(entry)" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_static_js_draws_idle_animation_until_video_arrives() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/static/request_session.js")
        body = await response.text()
        assert response.status == 200
        assert 'const idleCanvas = document.getElementById("idleCanvas")' in body
        assert "function drawIdleScene(now)" in body
        assert "window.requestAnimationFrame(drawIdleScene)" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_static_css_uses_lingbot_overlay_classes() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/static/request_session.css")
        body = await response.text()
        assert response.status == 200
        for selector in (
            ".overlayPanel",
            ".brandOverlay",
            ".statusCard",
            ".controlCard",
            ".logCard",
        ):
            assert selector in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_static_css_fades_idle_animation_after_video_arrives() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/static/request_session.css")
        body = await response.text()
        assert response.status == 200
        assert ".idleCanvas" in body
        assert "body.has-video .idleCanvas" in body
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
