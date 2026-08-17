# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for application serving through the shared WebRTC manager."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from flashdreams.demo import (
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    SessionInfo,
    WebRTCApplicationServing,
)
from flashdreams.demo import application as application_module
from flashdreams.demo.bridge import ApplicationCanonicalInputProvider
from flashdreams.runtime import (
    CAMERA_COMMAND,
    DRIVER_COMMAND,
    CanonicalInputSchema,
    CanonicalInputWindow,
    InferenceInput,
    StepRequirements,
    StepResult,
)
from flashdreams.runtime.demo import bootstrap as demo_bootstrap
from flashdreams.serving.webrtc import demo as webrtc_demo
from flashdreams.serving.webrtc import manager as manager_module
from flashdreams.serving.webrtc.encoders import ChunkDeliveryResult

pytestmark = pytest.mark.ci_cpu


class _AffinitySession(IFlashDreamsApplicationSession):
    def __init__(self, app: _AffinityApplication) -> None:
        self.app = app
        self.init_thread_id: int | None = None
        self.closed = False

    def init(self) -> None:
        self.init_thread_id = threading.get_ident()

    def session_info(self) -> SessionInfo:
        current_thread_id = threading.get_ident()
        assert current_thread_id == self.init_thread_id
        assert current_thread_id != self.app.calling_thread_id
        self.app.session_info_thread_ids.append(current_thread_id)
        return SessionInfo(steady_output_frame_count=7)

    def next_step_requirements(self) -> StepRequirements | None:
        return None

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        del inputs
        raise AssertionError("The preload test must not run an application step.")

    def close(self) -> None:
        self.closed = True


class _AffinityApplication(IFlashDreamsApplication):
    def __init__(self) -> None:
        self.calling_thread_id = threading.get_ident()
        self.init_args: tuple[str, ...] | None = None
        self.sessions: list[_AffinitySession] = []
        self.session_info_thread_ids: list[int] = []

    @property
    def input_schema(self) -> CanonicalInputSchema:
        return CanonicalInputSchema()

    def init(self, commandline_args: Sequence[str]) -> None:
        self.init_args = tuple(commandline_args)

    def create_session(self) -> IFlashDreamsApplicationSession:
        session = _AffinitySession(self)
        self.sessions.append(session)
        return session


class _SizedTrack:
    fps = 30

    def __init__(self) -> None:
        self.closed = False

    def qsize(self) -> int:
        return 0

    async def enqueue_result(self, result: StepResult) -> int:
        return result.frame_count

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        self.closed = True


class _RecordingEncoder:
    fps = 30
    backend = "fake"
    prefers_codec: str | None = None

    def __init__(self) -> None:
        self.maxsizes: list[int] = []
        self.tracks: list[_SizedTrack] = []

    def create_track(self, *, maxsize: int) -> _SizedTrack:
        self.maxsizes.append(maxsize)
        track = _SizedTrack()
        self.tracks.append(track)
        return track

    def prepare_chunk_payload(self, result: StepResult, track: object) -> StepResult:
        del track
        return result

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: _SizedTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        del force_keyframe
        if not isinstance(payload, StepResult):
            raise TypeError("Fake encoder payload must be a StepResult.")
        enqueued = await track.enqueue_result(payload)
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=0,
            encode_ms=0.0,
        )

    def close(self) -> None:
        return


class _FakePeerConnection:
    def __init__(self, _configuration: object = None) -> None:
        self.connectionState = "connected"
        self.localDescription: object | None = None
        self.handlers: dict[str, Any] = {}

    def on(self, event: str) -> Any:
        def register(handler: Any) -> Any:
            self.handlers[event] = handler
            return handler

        return register

    def addTransceiver(self, track: object, *, direction: str) -> object:
        assert track is not None
        assert direction == "sendonly"
        return SimpleNamespace(sender=SimpleNamespace(replaceTrack=lambda _track: None))

    async def setRemoteDescription(self, description: object) -> None:
        del description

    async def createAnswer(self) -> object:
        return SimpleNamespace(sdp="answer", type="answer")

    async def setLocalDescription(self, description: object) -> None:
        self.localDescription = description

    async def close(self) -> None:
        self.connectionState = "closed"


class _FakeDataChannel:
    readyState = "open"

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.messages: list[str] = []

    def on(self, event: str) -> Any:
        def register(handler: Any) -> Any:
            self.handlers[event] = handler
            return handler

        return register

    def send(self, message: str) -> None:
        self.messages.append(message)


class _InteractiveSession(IFlashDreamsApplicationSession):
    def __init__(self, app: _InteractiveApplication) -> None:
        self.app = app
        self.step_index = 0
        self.init_thread_id: int | None = None

    def init(self) -> None:
        self.init_thread_id = threading.get_ident()

    def session_info(self) -> SessionInfo:
        return SessionInfo(steady_output_frame_count=1)

    def next_step_requirements(self) -> StepRequirements | None:
        if self.step_index > 0:
            return None
        return StepRequirements(
            step_index=self.step_index,
            input_frame_count=1,
            steady_output_frame_count=1,
        )

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        self.app.windows.append(inputs)
        self.app.step_thread_ids.append(threading.get_ident())
        result = StepResult(
            step_index=self.step_index,
            output="frame",
            frame_count=1,
        )
        self.step_index += 1
        return result


class _InteractiveApplication(IFlashDreamsApplication):
    def __init__(self, *, live: bool = True) -> None:
        self.live = live
        self.windows: list[CanonicalInputWindow] = []
        self.step_thread_ids: list[int] = []
        self.sessions: list[_InteractiveSession] = []

    @property
    def input_schema(self) -> CanonicalInputSchema:
        if not self.live:
            return CanonicalInputSchema()
        return CanonicalInputSchema(modalities=(DRIVER_COMMAND, CAMERA_COMMAND))

    def init(self, commandline_args: Sequence[str]) -> None:
        del commandline_args

    def create_session(self) -> IFlashDreamsApplicationSession:
        session = _InteractiveSession(self)
        self.sessions.append(session)
        return session


def test_run_application_routes_serving_descriptor_without_creating_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AffinityApplication()
    serving = WebRTCApplicationServing("fake-app", host="0.0.0.0", port=9000)
    sentinel = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (app, ["--package-arg"]),
    )

    def fake_serve_application_webrtc(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        application_module,
        "serve_application_webrtc",
        fake_serve_application_webrtc,
    )

    result = application_module.run_application(
        "fake-app",
        ["--user-arg"],
        io_factory=serving,
    )

    assert result is sentinel
    assert app.init_args == ("--package-arg", "--user-arg")
    assert app.sessions == []
    assert captured == {
        "app": app,
        "application_slug": "fake-app",
        "serving": serving,
    }


def test_webrtc_cli_options_build_serving_descriptor() -> None:
    serving, application_args = application_module._parse_host_io(
        "fake-app",
        [
            "--output",
            "webrtc",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--prompt",
            "hello",
        ],
    )

    assert serving == WebRTCApplicationServing(
        "fake-app",
        host="0.0.0.0",
        port=9000,
    )
    assert application_args == ["--prompt", "hello"]


def test_application_serving_cleans_up_when_server_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AffinityApplication()
    app.init([])
    manager_holder: list[Any] = []
    cleanup_calls: list[dict[str, object]] = []

    async def run_inline(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(demo_bootstrap, "configure_logging", lambda: None)
    monkeypatch.setattr(
        demo_bootstrap,
        "initialize_cuda_distributed",
        lambda **_kwargs: SimpleNamespace(device="cuda:0", world_rank=3),
    )
    monkeypatch.setattr(manager_module.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(
        demo_bootstrap,
        "cleanup_cuda_distributed",
        lambda **kwargs: cleanup_calls.append(kwargs),
    )

    def fail_server_startup(**kwargs: Any) -> object:
        manager_holder.append(kwargs["session_manager"])
        raise RuntimeError("server startup failed")

    monkeypatch.setattr(webrtc_demo, "serve_webrtc_demo", fail_server_startup)

    with pytest.raises(RuntimeError, match="server startup failed"):
        application_module.serve_application_webrtc(
            app=app,
            application_slug="fake-app",
            serving=WebRTCApplicationServing("fake-app"),
        )

    assert manager_holder[0]._shared_host is None
    assert cleanup_calls == [{"world_rank": 3, "synchronize_distributed": False}]


@pytest.mark.asyncio
async def test_application_serving_preloads_on_worker_and_sizes_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AffinityApplication()
    app.init([])
    serving = WebRTCApplicationServing("fake-app")
    encoder = _RecordingEncoder()
    captured: dict[str, Any] = {}

    async def run_inline(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(demo_bootstrap, "configure_logging", lambda: None)
    monkeypatch.setattr(
        demo_bootstrap,
        "initialize_cuda_distributed",
        lambda **_kwargs: SimpleNamespace(device="cuda:2", world_rank=0),
    )
    monkeypatch.setattr(manager_module.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(manager_module, "RTCPeerConnection", _FakePeerConnection)

    async def skip_ice_wait(_peer: object) -> None:
        return None

    monkeypatch.setattr(
        manager_module,
        "wait_for_ice_gathering_complete",
        skip_ice_wait,
    )

    def fake_serve_webrtc_demo(**kwargs: Any) -> object:
        captured.update(kwargs)
        return "served"

    monkeypatch.setattr(webrtc_demo, "serve_webrtc_demo", fake_serve_webrtc_demo)

    result = application_module.serve_application_webrtc(
        app=app,
        application_slug="fake-app",
        serving=serving,
    )
    manager = captured["session_manager"]
    manager._shared_video_encoder = encoder
    assert manager._activate_without_input
    host = manager._shared_host
    assert host is not None

    try:
        await asyncio.wait_for(manager.preload_runtime(), timeout=1.0)
        assert manager.is_runtime_ready()
        assert manager.client_liveness_timeout_s == pytest.approx(30.0)
        assert manager._keep_connection_after_completed
        assert manager.runtime.peek_steady_output_num_frames() == 7
        assert len(app.sessions) == 1

        answer = await asyncio.wait_for(
            manager.create_answer(offer_sdp="offer", offer_type="offer"),
            timeout=1.0,
        )
        assert answer == {"sdp": "answer", "type": "answer"}
        assert encoder.maxsizes == [7]

        session = await asyncio.wait_for(
            host.call_async(host.start_session, InferenceInput()),
            timeout=1.0,
        )
        assert len(app.sessions) == 1
        await asyncio.wait_for(host.call_async(session.close), timeout=1.0)
    finally:
        await manager.close_active_session()
        if manager._shared_context is not None:
            await manager._shared_context.close_async()
            manager._shared_context = None
        host.close()
        manager._shared_host = None
        await manager.shutdown()

    assert result == "served"
    assert app.session_info_thread_ids == [app.sessions[0].init_thread_id]
    assert app.sessions[0].closed
    assert encoder.tracks[0].closed
    output = captured["output"]
    assert output.client_liveness_timeout_s == pytest.approx(30.0)
    assert output.preload_name == "fake-app"


@pytest.mark.asyncio
async def test_interactive_application_control_reaches_step_on_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _InteractiveApplication()
    app.init([])
    encoder = _RecordingEncoder()
    captured: dict[str, Any] = {}
    prepare_step_thread_ids: list[int] = []
    original_prepare_step = ApplicationCanonicalInputProvider.prepare_step

    def recording_prepare_step(
        provider: ApplicationCanonicalInputProvider,
        **kwargs: Any,
    ) -> Any:
        prepare_step_thread_ids.append(threading.get_ident())
        return original_prepare_step(provider, **kwargs)

    async def run_inline(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(demo_bootstrap, "configure_logging", lambda: None)
    monkeypatch.setattr(
        demo_bootstrap,
        "initialize_cuda_distributed",
        lambda **_kwargs: SimpleNamespace(device="cuda:0", world_rank=0),
    )
    monkeypatch.setattr(manager_module.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(manager_module, "RTCPeerConnection", _FakePeerConnection)
    monkeypatch.setattr(
        ApplicationCanonicalInputProvider,
        "prepare_step",
        recording_prepare_step,
    )

    async def skip_ice_wait(_peer: object) -> None:
        return None

    monkeypatch.setattr(
        manager_module,
        "wait_for_ice_gathering_complete",
        skip_ice_wait,
    )

    def fake_serve_webrtc_demo(**kwargs: Any) -> object:
        captured.update(kwargs)
        return "served"

    monkeypatch.setattr(webrtc_demo, "serve_webrtc_demo", fake_serve_webrtc_demo)

    result = application_module.serve_application_webrtc(
        app=app,
        application_slug="interactive-app",
        serving=WebRTCApplicationServing("interactive-app"),
    )
    manager = captured["session_manager"]
    manager._shared_video_encoder = encoder
    manager._keep_connection_after_completed = False
    assert not manager._activate_without_input
    callback_thread_id = threading.get_ident()
    host = manager._shared_host
    assert host is not None

    try:
        assert manager.browser_ui_config() == {
            "accepted_keys": [
                "a",
                "d",
                "down",
                "e",
                "i",
                "j",
                "k",
                "l",
                "left",
                "q",
                "right",
                "s",
                "space",
                "up",
                "w",
            ]
        }
        await asyncio.wait_for(manager.preload_runtime(), timeout=1.0)
        await asyncio.wait_for(
            manager.create_answer(offer_sdp="offer", offer_type="offer"),
            timeout=1.0,
        )
        managed = manager._active_session
        assert managed is not None
        peer = managed.peer_connection
        channel = _FakeDataChannel()
        peer.handlers["datachannel"](channel)
        generation_task = managed.generation_task
        assert generation_task is not None

        channel.handlers["message"](
            json.dumps(
                {
                    "type": "action",
                    "action": {"event": "keydown", "key": "z"},
                }
            )
        )
        await asyncio.sleep(0)
        assert not managed.first_action_received.is_set()
        assert app.windows == []

        channel.handlers["message"](
            json.dumps(
                {
                    "type": "action",
                    "action": {"event": "keydown", "key": "q"},
                }
            )
        )
        await asyncio.wait_for(generation_task, timeout=2.0)
    finally:
        await manager.close_active_session()
        if manager._shared_context is not None:
            await manager._shared_context.close_async()
            manager._shared_context = None
        host.close()
        manager._shared_host = None
        await manager.shutdown()

    assert result == "served"
    assert len(app.windows) == 1
    window = app.windows[0]
    assert window.values[DRIVER_COMMAND.name]["stop"] is False
    camera = window.values[CAMERA_COMMAND.name]
    assert camera["move_right"] == -1.0
    segments = camera["segments"]
    assert segments[0][0] == pytest.approx(window.window.start_s)
    assert segments[-1][1] == pytest.approx(window.window.end_s)
    assert len(app.sessions) == 1
    worker_thread_id = app.sessions[0].init_thread_id
    assert worker_thread_id is not None
    assert prepare_step_thread_ids == [worker_thread_id]
    assert app.step_thread_ids == [worker_thread_id]
    assert worker_thread_id != callback_thread_id
