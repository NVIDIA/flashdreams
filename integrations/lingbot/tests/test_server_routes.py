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

import base64
from contextlib import ExitStack

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from lingbot.webrtc import server as webrtc_server
from lingbot.webrtc.server import _close_package_resources, create_app
from lingbot.webrtc.session import (
    LingbotImagePayload,
    LingbotSessionInput,
    SessionBusyError,
)

pytestmark = pytest.mark.ci_gpu


class FakeSessionManager:
    def __init__(self) -> None:
        self.answer_payload = {"sdp": "fake-answer-sdp", "type": "answer"}
        self.raise_busy = False
        self.close_calls = 0
        self.preload_calls = 0
        self.offers: list[tuple[str, str]] = []
        self.pending_inputs: list[LingbotSessionInput] = []
        self.active = False
        self.runtime_ready = False
        self.initial_scene: dict[str, object] = {
            "first_frame_url": "/api/session/first_frame",
            "prompt": "drive through a city",
            "model": "FakeLingbot",
            "resolution": {"width": 832, "height": 464},
        }
        self.first_frame = LingbotImagePayload(
            data=b"fake-first-frame",
            content_type="image/jpeg",
        )

    def has_active_session(self) -> bool:
        return self.active

    def is_runtime_ready(self) -> bool:
        return self.runtime_ready

    def get_initial_scene(self) -> dict[str, object]:
        return self.initial_scene

    def get_first_frame(self) -> LingbotImagePayload:
        return self.first_frame

    def set_pending_session_input(self, session_input: LingbotSessionInput) -> None:
        self.pending_inputs.append(session_input)
        self.initial_scene = {
            **self.initial_scene,
            "prompt": session_input.prompt or self.initial_scene["prompt"],
            "input_source": "uploaded",
        }

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


def test_create_app_keeps_package_web_resource_materialized() -> None:
    app = create_app(
        session_manager=FakeSessionManager(),
        request_session_url="http://127.0.0.1:8080/request_session",
    )
    try:
        assert isinstance(app["package_resource_stack"], ExitStack)
        assert _close_package_resources in app.on_cleanup

        static_resources = [
            resource
            for resource in app.router.resources()
            if getattr(resource, "canonical", "") == "/static"
            or resource.get_info().get("prefix") in {"/static", "/static/"}
        ]
        assert len(static_resources) == 1
        web_dir = static_resources[0].get_info()["directory"]
        assert web_dir.is_dir()
        assert "Lingbot WebRTC Viewer" in (web_dir / "request_session.html").read_text()
    finally:
        app["package_resource_stack"].close()


def test_create_app_closes_package_resource_when_app_creation_fails(
    monkeypatch, tmp_path
) -> None:
    class TrackedResource:
        closed = False

        def __enter__(self):
            return tmp_path

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True

    tracked_resource = TrackedResource()

    def raise_app_creation_failure(**_kwargs):
        raise RuntimeError("app creation failed")

    monkeypatch.setattr(webrtc_server, "as_file", lambda _resource: tracked_resource)
    monkeypatch.setattr(
        webrtc_server,
        "create_webrtc_app",
        raise_app_creation_failure,
    )

    with pytest.raises(RuntimeError, match="app creation failed"):
        create_app(
            session_manager=FakeSessionManager(),
            request_session_url="http://127.0.0.1:8080/request_session",
        )

    assert tracked_resource.closed


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
async def test_initial_scene_route_returns_preview_metadata() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/api/session/initial_scene")
        payload = await response.json()
        assert response.status == 200
        assert payload == manager.initial_scene
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_first_frame_route_serves_manager_image() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        response = await client.get("/api/session/first_frame")
        body = await response.read()
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/jpeg"
        assert body == b"fake-first-frame"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_input_upload_stores_prompt_and_image() -> None:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        form = FormData()
        form.add_field(
            "prompt",
            "turn onto a rain-soaked neon street\nwith reflective traffic lights",
        )
        form.add_field(
            "image",
            png_bytes,
            filename="scene.png",
            content_type="image/png",
        )

        response = await client.post("/api/session/input", data=form)
        payload = await response.json()

        assert response.status == 200
        assert (
            payload["prompt"]
            == "turn onto a rain-soaked neon street with reflective traffic lights"
        )
        assert payload["input_source"] == "uploaded"
        assert len(manager.pending_inputs) == 1
        session_input = manager.pending_inputs[0]
        assert (
            session_input.prompt
            == "turn onto a rain-soaked neon street with reflective traffic lights"
        )
        assert session_input.first_frame_image_bytes == png_bytes
        assert session_input.first_frame_content_type == "image/png"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_input_accepts_image_url() -> None:
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        form = FormData()
        form.add_field("image_url", "https://example.test/scene.jpg")

        response = await client.post("/api/session/input", data=form)

        assert response.status == 200
        assert len(manager.pending_inputs) == 1
        session_input = manager.pending_inputs[0]
        assert session_input.first_frame_image_url == "https://example.test/scene.jpg"
        assert session_input.first_frame_image_bytes is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_input_file_upload_overrides_image_url() -> None:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    manager = FakeSessionManager()
    client = await _build_client(manager)
    try:
        form = FormData()
        form.add_field("image_url", "https://example.test/scene.jpg")
        form.add_field(
            "image",
            png_bytes,
            filename="scene.png",
            content_type="image/png",
        )

        response = await client.post("/api/session/input", data=form)

        assert response.status == 200
        assert len(manager.pending_inputs) == 1
        session_input = manager.pending_inputs[0]
        assert session_input.first_frame_image_url is None
        assert session_input.first_frame_image_bytes == png_bytes
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
