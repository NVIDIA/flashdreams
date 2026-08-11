# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, WebRTCOutputSpec
from flashdreams.serving.webrtc.server import PACKAGE_RESOURCE_STACK_KEY
from flashvsr.corrector import FlashVSRColorCorrector
from flashvsr.demo.app import _webrtc_spec, parse_args
from flashvsr.demo.providers import PREPARED_VIDEO_METADATA_KEY
from flashvsr.demo.server import (
    FlashVSRUploadController,
    FlashVSRWebRTCSessionInput,
    configure_flashvsr_webrtc_app,
)
from flashvsr.demo.spec import FlashVSRVideoScenario, PreparedFlashVSRVideo
from flashvsr.demo.webrtc import serve_flashvsr_webrtc_demo
from flashvsr.runtime import FLASHVSR_MODEL_ID

pytestmark = pytest.mark.ci_cpu


class _FakeSessionManager:
    def __init__(self) -> None:
        self.pending_session_input: Any = None
        self.active = False

    def has_active_session(self) -> bool:
        return self.active

    def set_pending_session_input(self, session_input: Any) -> None:
        self.pending_session_input = session_input


class _FakeUploadAdapter:
    def __init__(self, prepared: PreparedFlashVSRVideo) -> None:
        self.prepared = prepared
        self.upload_bytes: bytes | None = None
        self.upload_path: Path | None = None
        self.error: Exception | None = None

    def prepare_uploaded_video(
        self,
        spec: DemoSpec,
        *,
        upload_path: Path,
        original_name: str,
    ) -> PreparedFlashVSRVideo:
        del spec, original_name
        self.upload_path = upload_path
        self.upload_bytes = upload_path.read_bytes()
        if self.error is not None:
            raise self.error
        return self.prepared


class _FakeRuntime:
    def __init__(self, *, config: InferenceConfig, options: Any) -> None:
        self.config = config
        self.options = options
        self.closed = False

    def preload(self) -> None:
        return

    def peek_input_fps(self) -> float:
        return float(self.config.runtime_options["fps"])

    def peek_steady_output_num_frames(self) -> int:
        return int(self.config.runtime_options["chunk_size"])

    def start_session(self, inputs: Any) -> Any:
        raise AssertionError(f"test should not start a session: {inputs}")

    def close(self) -> None:
        self.closed = True


def _pipeline_config() -> Any:
    return SimpleNamespace(encoder=SimpleNamespace(scale=2))


def _prepared_video(
    *,
    filename: str = "upload.mp4",
    frames: int = 5,
    input_height: int = 64,
    input_width: int = 96,
    target_height: int = 128,
    target_width: int = 128,
    fps: float = 20.0,
) -> PreparedFlashVSRVideo:
    scenario = FlashVSRVideoScenario(
        input_path=filename,
        chunk_size=8,
        fps=fps,
        loop_input=True,
    )
    return PreparedFlashVSRVideo(
        scenario=scenario,
        resolved_path=Path(filename),
        video=torch.zeros(1, 3, frames, input_height, input_width),
        input_height=input_height,
        input_width=input_width,
        target_height=target_height,
        target_width=target_width,
        fps=fps,
    )


def _upload_spec(*, input_path: str | Path | None = None) -> DemoSpec:
    return DemoSpec(
        model_id=FLASHVSR_MODEL_ID,
        input_mode="replay",
        scenario=FlashVSRVideoScenario(
            input_path=input_path,
            chunk_size=8,
            fps=20.0,
            loop_input=True,
        ),
        output=WebRTCOutputSpec(
            host="127.0.0.1",
            port=8088,
            fps=20,
            video_height=128,
            video_width=128,
            warmup_chunks=1,
        ),
        config=InferenceConfig(
            model_id=FLASHVSR_MODEL_ID,
            device="cpu",
            runtime_options={
                "pipeline_config": _pipeline_config(),
                "chunk_size": 8,
                "fps": 20.0,
            },
        ),
    )


async def _build_upload_client(
    controller: FlashVSRUploadController,
) -> TestClient:
    app = web.Application()

    async def offer(_: web.Request) -> web.StreamResponse:
        return web.json_response({"sdp": "answer", "type": "answer"})

    app.router.add_post("/api/webrtc/offer", offer)
    configure_flashvsr_webrtc_app(app, controller=controller)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_webrtc_cli_allows_browser_upload_without_input() -> None:
    args = parse_args(["webrtc"])
    spec = _webrtc_spec(args, device="cuda:0")

    assert args.input_path is None
    assert isinstance(spec.scenario, FlashVSRVideoScenario)
    assert spec.scenario.input_path is None
    assert spec.scenario.fps == 30.0
    assert isinstance(spec.output, WebRTCOutputSpec)
    assert spec.output.fps == 30


def test_webrtc_server_defers_warmup_without_default_input() -> None:
    calls: list[dict[str, Any]] = []
    app = serve_flashvsr_webrtc_demo(
        spec=_upload_spec(),
        runtime_factory=_FakeRuntime,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )

    assert app is not None
    manager = calls[0]["session_manager"]
    assert manager._shared_scenario is None
    assert callable(manager._shared_spec_factory)
    assert manager.runtime_config.warmup_chunks == 0
    assert manager.runtime_config.encoder_backend == "default"
    routes = {resource.canonical for resource in app.router.resources()}
    assert "/api/session/input" in routes

    manager._shared_host.close()
    app[PACKAGE_RESOURCE_STACK_KEY].close()


def test_uploaded_spec_uses_prepared_video_and_output_dimensions() -> None:
    calls: list[dict[str, Any]] = []
    app = serve_flashvsr_webrtc_demo(
        spec=_upload_spec(),
        runtime_factory=_FakeRuntime,
        server_runner=lambda **kwargs: calls.append(kwargs),
    )
    assert app is not None
    manager = calls[0]["session_manager"]
    prepared = _prepared_video(target_height=256, target_width=384)
    session_spec = manager._shared_spec_factory(
        FlashVSRWebRTCSessionInput(
            prepared_video=prepared,
            original_name="upload.mp4",
        )
    )

    assert session_spec.scenario is prepared
    assert isinstance(session_spec.output, WebRTCOutputSpec)
    assert session_spec.output.video_height == 256
    assert session_spec.output.video_width == 384
    scenario = manager._shared_adapter.prepare_scenario(session_spec)
    assert scenario.metadata[PREPARED_VIDEO_METADATA_KEY] is prepared

    manager._shared_host.close()
    app[PACKAGE_RESOURCE_STACK_KEY].close()


@pytest.mark.asyncio
async def test_offer_requires_upload_when_no_default_input() -> None:
    manager = _FakeSessionManager()
    controller = FlashVSRUploadController(
        manager=manager,
        adapter=_FakeUploadAdapter(_prepared_video()),
        spec=_upload_spec(),
        default_video=None,
    )
    client = await _build_upload_client(controller)
    try:
        status_response = await client.get("/api/session/input")
        status = await status_response.json()
        assert status_response.status == 200
        assert status["upload_required"] is True

        offer = await client.post(
            "/api/webrtc/offer",
            json={"sdp": "offer", "type": "offer"},
        )
        assert offer.status == 400
        assert "Upload an MP4" in await offer.text()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mp4_upload_is_decoded_staged_and_temp_file_removed() -> None:
    prepared = _prepared_video(target_height=256, target_width=384)
    adapter = _FakeUploadAdapter(prepared)
    manager = _FakeSessionManager()
    controller = FlashVSRUploadController(
        manager=manager,
        adapter=adapter,
        spec=_upload_spec(),
        default_video=None,
    )
    client = await _build_upload_client(controller)
    try:
        form = FormData()
        form.add_field(
            "video",
            b"fake-mp4",
            filename="../uploaded.mp4",
            content_type="video/mp4",
        )
        response = await client.post("/api/session/input", data=form)
        payload = await response.json()

        assert response.status == 200
        assert payload["input_source"] == "uploaded"
        assert payload["filename"] == "uploaded.mp4"
        assert payload["resolution"] == {"width": 384, "height": 256}
        assert adapter.upload_bytes == b"fake-mp4"
        assert adapter.upload_path is not None
        assert not adapter.upload_path.exists()
        assert isinstance(manager.pending_session_input, FlashVSRWebRTCSessionInput)
        assert manager.pending_session_input.prepared_video is prepared
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_rejects_empty_wrong_type_and_decode_failure() -> None:
    adapter = _FakeUploadAdapter(_prepared_video())
    manager = _FakeSessionManager()
    controller = FlashVSRUploadController(
        manager=manager,
        adapter=adapter,
        spec=_upload_spec(),
        default_video=None,
    )
    client = await _build_upload_client(controller)
    try:
        wrong_type = FormData()
        wrong_type.add_field(
            "video",
            b"not-video",
            filename="input.txt",
            content_type="text/plain",
        )
        response = await client.post("/api/session/input", data=wrong_type)
        assert response.status == 400

        empty = FormData()
        empty.add_field(
            "video",
            b"",
            filename="empty.mp4",
            content_type="video/mp4",
        )
        response = await client.post("/api/session/input", data=empty)
        assert response.status == 400

        adapter.error = ValueError("decoder rejected container")
        corrupt = FormData()
        corrupt.add_field(
            "video",
            b"corrupt",
            filename="corrupt.mp4",
            content_type="video/mp4",
        )
        response = await client.post("/api/session/input", data=corrupt)
        assert response.status == 400
        assert "decoder rejected container" in await response.text()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_returns_conflict_while_session_is_active() -> None:
    manager = _FakeSessionManager()
    manager.active = True
    controller = FlashVSRUploadController(
        manager=manager,
        adapter=_FakeUploadAdapter(_prepared_video()),
        spec=_upload_spec(),
        default_video=None,
    )
    client = await _build_upload_client(controller)
    try:
        form = FormData()
        form.add_field(
            "video",
            b"fake-mp4",
            filename="input.mp4",
            content_type="video/mp4",
        )
        response = await client.post("/api/session/input", data=form)
        assert response.status == 409
    finally:
        await client.close()


def test_torch_adain_accepts_center_cropped_noncontiguous_video() -> None:
    torch.manual_seed(0)
    content = torch.randn(1, 3, 8, 8, 16)[:, :, :5, :, 1:15]
    style = torch.randn(1, 3, 8, 8, 16)[:, :, :5, :, 1:15]
    assert not content.is_contiguous()
    assert not style.is_contiguous()
    corrector = FlashVSRColorCorrector(implementation="torch")

    expected = corrector(
        content.contiguous(),
        style.contiguous(),
        method="adain",
    )
    actual = corrector(content, style, method="adain")

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_flashvsr_browser_adapter_uploads_before_connect() -> None:
    adapter_path = (
        Path(__file__).parents[1] / "flashvsr" / "demo" / "web" / "adapter.js"
    )
    stylesheet_path = adapter_path.with_name("adapter.css")
    source = adapter_path.read_text()
    stylesheet = stylesheet_path.read_text()

    assert 'type="file"' in source
    assert "video/mp4" in source
    assert "/api/session/input" in source
    assert 'class="flashvsrStartButton"' in source
    assert 'action: { event: "step" }' in source
    assert "beforeConnect" in source
    assert "onConnect" in source
    assert "sendCommand(START_ACTION" in source
    assert 'startButton.addEventListener("click"' in source
    assert "  controls," not in source
    assert "RTCPeerConnection" not in source
    on_connect = source.split("onConnect() {", maxsplit=1)[1].split("},", maxsplit=1)[0]
    assert "sendCommand" not in on_connect
    assert ".flashvsrUploadCard" in stylesheet
    assert ".flashvsrStartButton" in stylesheet
    assert "position: absolute" in stylesheet


def test_shared_webrtc_shell_hides_empty_controls_and_notifies_connect() -> None:
    web_dir = (
        Path(__file__).parents[3]
        / "flashdreams"
        / "flashdreams"
        / "serving"
        / "webrtc"
        / "web"
    )

    assert 'id="controlCard"' in (web_dir / "request_session.html").read_text()
    source = (web_dir / "request_session.js").read_text()
    assert "syncControlCardVisibility" in source
    assert "modelAdapter?.onConnect?.(modelContext)" in source
    assert "setFlow," in source
    assert "setStatus," in source
