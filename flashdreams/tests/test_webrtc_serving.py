# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import nullcontext
from importlib.resources import files
from typing import Any

import numpy as np
import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from flashdreams.runtime.demo import RealtimeEventResampler
from flashdreams.runtime.keyboard import WSAD_SUPPORTED_KEYS, KeyboardState
from flashdreams.serving.webrtc.manager import (
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
)
from flashdreams.serving.webrtc.media import tensor_chunk_to_rgb_frames
from flashdreams.serving.webrtc.messages import (
    make_chunk_done_payload,
    make_error_payload,
    make_event_ack_payload,
)
from flashdreams.serving.webrtc.server import (
    PACKAGE_RESOURCE_STACK_KEY,
    close_package_resources,
    create_packaged_webrtc_app,
)

pytestmark = pytest.mark.ci_cpu


def test_wsad_keyboard_state_rejects_non_driving_keys() -> None:
    state = KeyboardState(supported_keys=WSAD_SUPPORTED_KEYS)

    assert state.apply_event(event="keydown", key="ArrowUp")
    assert state.resolved_effective_keys() == frozenset({"w"})
    assert not state.apply_event(event="keydown", key="q")
    assert state.resolved_effective_keys() == frozenset({"w"})


def test_tensor_chunk_to_rgb_frames_supports_omnidreams_layout() -> None:
    chunk = torch.zeros((1, 1, 2, 3, 4, 5), dtype=torch.uint8)
    chunk[0, 0, 1, 0] = 255

    frames = tensor_chunk_to_rgb_frames(chunk)

    assert len(frames) == 2
    assert frames[0].shape == (4, 5, 3)
    assert frames[0].dtype == np.uint8
    assert frames[1][0, 0, 0] == 255


class _FakeSessionManager:
    def __init__(self) -> None:
        self.preload_calls = 0
        self.shutdown_calls = 0

    def has_active_session(self) -> bool:
        return False

    def is_runtime_ready(self) -> bool:
        return self.preload_calls > 0

    async def preload_runtime(self) -> None:
        self.preload_calls += 1

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        del offer_sdp, offer_type
        return {"sdp": "answer-sdp", "type": "answer"}

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeCloseable:
    async def close(self) -> None:
        return


class _FakeControlChannel:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send(self, payload: str) -> None:
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)
        self.messages.append(decoded)


class _Manager(BaseWebRTCSessionManager[Any, Any]):
    def _model_name(self) -> str:
        return "fake"


def _managed_session_with_channel(
    runtime: object,
) -> tuple[ManagedWebRTCSession, _FakeControlChannel]:
    channel = _FakeControlChannel()
    managed_session = ManagedWebRTCSession(
        runtime=runtime,
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        video_encoder=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=RealtimeEventResampler(fps=30, start_v=0.0),
        control_channel=channel,
    )
    return managed_session, channel


@pytest.mark.asyncio
async def test_event_message_dispatches_to_runtime_and_acknowledges() -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def trigger_event(
            self, *, event_id: str, state: str
        ) -> dict[str, object]:
            self.calls.append((event_id, state))
            return {"active_event_id": event_id}

    runtime = _FakeRuntime()
    manager = _Manager(
        runtime=runtime, runtime_config=object(), fps=30, identity="fake"
    )
    managed_session, channel = _managed_session_with_channel(runtime)

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"event","event_id":"portal","state":"trigger"}',
    )

    assert runtime.calls == [("portal", "trigger")]
    assert channel.messages == [
        {
            "type": "event_ack",
            "event_id": "portal",
            "state": "trigger",
            "active_event_id": "portal",
        }
    ]
    assert managed_session.first_action_received.is_set()


@pytest.mark.asyncio
async def test_clear_event_message_preserves_ack_fields() -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def trigger_event(
            self, *, event_id: str, state: str
        ) -> dict[str, object]:
            self.calls.append((event_id, state))
            return {
                "type": "not_event_ack",
                "event_id": "overwritten",
                "state": "overwritten",
                "active_event_id": None,
            }

    runtime = _FakeRuntime()
    manager = _Manager(
        runtime=runtime, runtime_config=object(), fps=30, identity="fake"
    )
    managed_session, channel = _managed_session_with_channel(runtime)

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"event","state":"clear"}',
    )

    assert runtime.calls == [("", "clear")]
    assert channel.messages == [
        {
            "type": "event_ack",
            "event_id": None,
            "state": "clear",
            "active_event_id": None,
        }
    ]
    assert managed_session.first_action_received.is_set()


@pytest.mark.asyncio
async def test_event_message_without_id_is_rejected_for_trigger() -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        async def trigger_event(
            self, *, event_id: str, state: str
        ) -> dict[str, object]:
            del event_id, state
            self.calls += 1
            return {}

    runtime = _FakeRuntime()
    manager = _Manager(
        runtime=runtime, runtime_config=object(), fps=30, identity="fake"
    )
    managed_session, channel = _managed_session_with_channel(runtime)

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"event","state":"trigger"}',
    )

    assert runtime.calls == 0
    assert channel.messages == [
        {
            "type": "error",
            "message": (
                "Event payload must include non-empty 'event_id' "
                "unless state clears the active event."
            ),
        }
    ]
    assert not managed_session.first_action_received.is_set()


def test_packaged_webrtc_app_keeps_resource_materialized(tmp_path) -> None:
    (tmp_path / "request_session.html").write_text(
        "<html>session</html>", encoding="utf-8"
    )
    (tmp_path / "client.js").write_text("", encoding="utf-8")

    app = create_packaged_webrtc_app(
        web_resource=tmp_path,
        session_manager=_FakeSessionManager(),
        request_session_url="http://127.0.0.1:8080/request_session",
        preload_name="Test",
        as_file_fn=lambda resource: nullcontext(resource),
    )
    try:
        assert PACKAGE_RESOURCE_STACK_KEY in app
        assert close_package_resources in app.on_cleanup

        static_resources = [
            resource
            for resource in app.router.resources()
            if resource.get_info().get("prefix") in {"/static", "/static/"}
        ]
        assert len(static_resources) == 1
        assert static_resources[0].get_info()["directory"] == tmp_path
    finally:
        app[PACKAGE_RESOURCE_STACK_KEY].close()


def test_packaged_webrtc_app_closes_resource_when_setup_fails(tmp_path) -> None:
    closed = False

    class _TrackedContext:
        def __enter__(self):
            return tmp_path

        def __exit__(self, exc_type, exc_value, traceback):
            nonlocal closed
            closed = True

    def _raise_creation_failure(**_kwargs) -> web.Application:
        raise RuntimeError("app creation failed")

    with pytest.raises(RuntimeError, match="app creation failed"):
        create_packaged_webrtc_app(
            web_resource=tmp_path,
            session_manager=_FakeSessionManager(),
            request_session_url="http://127.0.0.1:8080/request_session",
            preload_name="Test",
            as_file_fn=lambda _resource: _TrackedContext(),
            create_app_fn=_raise_creation_failure,
        )

    assert closed


def test_shared_viewer_exposes_model_extension_slots() -> None:
    web_dir = files("flashdreams.serving.webrtc").joinpath("web")
    html = web_dir.joinpath("request_session.html").read_text(encoding="utf-8")
    javascript = web_dir.joinpath("request_session.js").read_text(encoding="utf-8")

    assert "/static/request_session.js?v=shared-webrtc-v7" in html
    assert "attemptsRemaining: autoConnectMaxAttempts" in javascript
    assert javascript.count("connected = true") == 1
    assert 'pc.connectionState !== "connected"' in javascript
    assert 'channel.readyState !== "open"' in javascript
    assert 'setStatus("Connected", "connected")' in javascript
    assert "isTransientFetchError(error)" in javascript
    assert "peerConnection !== pc || controlChannel !== channel" in javascript
    for slot in (
        "modelStageSlot",
        "modelStatusSlot",
        "modelPanelSlot",
        "modelControlSlot",
    ):
        assert f'id="{slot}"' in html
    assert 'fetch("/api/ui/config")' in javascript
    assert "config.model_stylesheet" in javascript
    assert "stylesheetHrefs" in javascript
    assert "await modelAdapter?.beforeConnect?.(modelContext)" in javascript
    assert "sendCommand: sendModelCommand" in javascript
    assert 'id="postprocessField"' in html
    assert 'fetch("/api/postprocess/options")' in javascript
    assert "@typedef {Object} WebRTCModelAdapter" in javascript
    assert "adapter.capabilities?.postprocess === true" in javascript
    assert "config.accepted_keys" in javascript
    assert 'label: "Controls"' in javascript
    assert "Array.isArray(adapter.controls)" in javascript
    assert "renderControls(modelControls)" in javascript
    assert "/api/session/initial_scene" not in javascript


@pytest.mark.asyncio
async def test_packaged_webrtc_app_serves_common_routes(tmp_path) -> None:
    (tmp_path / "request_session.html").write_text(
        "<html>session</html>", encoding="utf-8"
    )
    manager = _FakeSessionManager()
    app = create_packaged_webrtc_app(
        web_resource=tmp_path,
        session_manager=manager,
        request_session_url="http://127.0.0.1:8080/request_session",
        preload_name="Test",
        as_file_fn=lambda resource: nullcontext(resource),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        response = await client.get("/request_session")
        body = await response.text()

        assert response.status == 200
        assert body == "<html>session</html>"
        assert manager.preload_calls == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_packaged_webrtc_app_serves_model_adapter(tmp_path) -> None:
    shared_dir = tmp_path / "shared"
    model_dir = tmp_path / "model"
    shared_dir.mkdir()
    model_dir.mkdir()
    (shared_dir / "request_session.html").write_text("<html>session</html>")
    (model_dir / "adapter.js").write_text("export default {}")
    app = create_packaged_webrtc_app(
        web_resource=shared_dir,
        model_web_resource=model_dir,
        session_manager=_FakeSessionManager(),
        request_session_url="http://127.0.0.1:8080/request_session",
        preload_name="Test",
        as_file_fn=lambda resource: nullcontext(resource),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        config_response = await client.get("/api/ui/config")
        assert await config_response.json() == {
            "adapter_module": "/model-static/adapter.js?v=model-ui-v2"
        }
        adapter_response = await client.get("/model-static/adapter.js")
        assert adapter_response.status == 200
        assert await adapter_response.text() == "export default {}"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_packaged_webrtc_app_serves_model_stylesheet(tmp_path) -> None:
    shared_dir = tmp_path / "shared"
    model_dir = tmp_path / "model"
    shared_dir.mkdir()
    model_dir.mkdir()
    (shared_dir / "request_session.html").write_text("<html>session</html>")
    (model_dir / "adapter.css").write_text(".stageVideo { object-fit: contain; }")
    app = create_packaged_webrtc_app(
        web_resource=shared_dir,
        model_web_resource=model_dir,
        session_manager=_FakeSessionManager(),
        request_session_url="http://127.0.0.1:8080/request_session",
        preload_name="Test",
        as_file_fn=lambda resource: nullcontext(resource),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        config_response = await client.get("/api/ui/config")
        assert await config_response.json() == {
            "adapter_module": None,
            "model_stylesheet": "/model-static/adapter.css?v=model-ui-v2",
        }
        stylesheet_response = await client.get("/model-static/adapter.css")
        assert stylesheet_response.status == 200
        assert (
            await stylesheet_response.text() == ".stageVideo { object-fit: contain; }"
        )
    finally:
        await client.close()


def test_webrtc_message_helpers_preserve_public_payload_shape() -> None:
    assert make_error_payload("boom") == {"type": "error", "message": "boom"}
    assert make_event_ack_payload(
        event_id="rain",
        state="trigger",
        result={"state": "ignored", "active_event_id": "rain"},
    ) == {
        "type": "event_ack",
        "event_id": "rain",
        "state": "trigger",
        "active_event_id": "rain",
    }

    assert make_chunk_done_payload(
        chunk_index=2,
        num_frames=3,
        enqueued_frames=3,
        fps=30,
        width=1280,
        height=704,
        model="demo",
        gen_ms=12.34,
        enqueue_ms=0.56,
        play_ms=100.0,
        queue_depth=1,
        lag_ms=4.44,
        control_latency_ms=20.04,
        consumed_actions=2,
        extra={"stream": "rgb"},
    ) == {
        "type": "chunk_done",
        "chunk_index": 2,
        "num_frames": 3,
        "enqueued_frames": 3,
        "fps": 30,
        "resolution": {"width": 1280, "height": 704},
        "model": "demo",
        "gen_ms": 12.3,
        "enqueue_ms": 0.6,
        "play_ms": 100.0,
        "queue_depth": 1,
        "lag_ms": 4.4,
        "stream": "rgb",
        "latency_ms": 20.0,
        "control_latency_ms": 20.0,
        "consumed_actions": 2,
    }
