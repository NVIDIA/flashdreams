# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from importlib.resources import files

import numpy as np
import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from flashdreams.serving.webrtc.controls import (
    WSAD_SUPPORTED_KEYS,
    CameraPoseIntegrator,
    KeyboardResampler,
    KeyboardState,
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


def test_wsad_resampler_preserves_held_key() -> None:
    resampler = KeyboardResampler(
        fps=30,
        start_v=1.0,
        supported_keys=WSAD_SUPPORTED_KEYS,
    )
    resampler.on_edge(arrival_t=0.5, event="keydown", key="w")

    segments, frame_times = resampler.sample_chunk(num_frames=2)

    assert segments == [(1.0, 1.0 + 2 / 30, frozenset({"w"}))]
    assert frame_times == pytest.approx([1.0 + 1 / 30, 1.0 + 2 / 30])


def test_camera_pose_integrator_flu_uses_driving_axes() -> None:
    integrator = CameraPoseIntegrator(
        move_speed_per_s=2.0,
        rotate_speed_rad_per_s=float(np.pi / 2),
        coordinate_system="FLU",
    )

    integrator.reset()
    poses = integrator.integrate_chunk(
        segments=[(0.0, 1.0, frozenset({"w"}))],
        frame_times=[1.0],
    )
    assert poses[-1][:3, 3] == pytest.approx([2.0, 0.0, 0.0])

    integrator.reset()
    poses = integrator.integrate_chunk(
        segments=[(0.0, 1.0, frozenset({"a"}))],
        frame_times=[1.0],
    )
    assert poses[-1][:3, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    integrator.reset()
    poses = integrator.integrate_chunk(
        segments=[(0.0, 1.0, frozenset({"d"}))],
        frame_times=[1.0],
    )
    assert poses[-1][:3, 0] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)


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


def test_shared_viewer_treats_postprocess_routes_as_optional() -> None:
    web_dir = files("flashdreams.serving.webrtc").joinpath("web")
    html = web_dir.joinpath("request_session.html").read_text(encoding="utf-8")
    javascript = web_dir.joinpath("request_session.js").read_text(encoding="utf-8")

    assert "/static/request_session.js?v=shared-webrtc-v1" in html
    assert "if (!postprocessField || !postprocessSelect)" in javascript
    assert "if (response.status === 404)" in javascript
    assert "if (!postprocessControlAvailable || !postprocessSelect)" in javascript


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
