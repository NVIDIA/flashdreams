# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    CanonicalInputs,
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputCanonicalizer,
    InputField,
    InputMapping,
    InputMappingSchema,
    NullMetricsRecorder,
    NullOutputTarget,
    OutputArtifact,
    OutputTarget,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
    build_output_target,
    run_replay_demo,
)
from flashdreams.runtime.demo.webrtc import (
    PendingSessionInputState,
    SharedDemoWebRTCSessionManager,
    WebRTCAppExtension,
    WebRTCManagerOptions,
    WebRTCRoute,
    build_webrtc_demo,
    json_get_route,
    session_input_route,
)
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.server import (
    PACKAGE_RESOURCE_STACK_KEY,
    SessionBusyError,
)

pytestmark = pytest.mark.ci_cpu


def _json_response_payload(response: web.StreamResponse) -> dict[str, Any]:
    assert isinstance(response, web.Response)
    text = response.text
    assert text is not None
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def test_replay_demo_uses_shared_runner() -> None:
    adapter = _FakeDemoAdapter()
    output = _RecordingOutputTarget()
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> Sequence[OutputArtifact]:
        calls.append(kwargs)
        return (OutputArtifact(kind="test/artifact", uri="memory://artifact"),)

    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="replay",
        output=NullOutputSpec(),
    )

    artifacts = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: output,
        metrics=NullMetricsRecorder(),
        runner=fake_runner,
    )

    assert artifacts == (OutputArtifact(kind="test/artifact", uri="memory://artifact"),)
    assert len(calls) == 1
    assert calls[0]["adapter"] is adapter
    assert calls[0]["config"] == spec.config
    assert calls[0]["mapping"] is adapter.prepared_scenario.mapping
    assert calls[0]["canonicalizer"] is adapter.prepared_scenario.canonicalizer
    assert calls[0]["source_schema"] is adapter.prepared_scenario.source_schema
    assert calls[0]["user_inputs"] is adapter.prepared_scenario.user_inputs
    assert calls[0]["initial_inputs"] is adapter.prepared_scenario.initial_inputs
    assert calls[0]["output"] is output
    assert adapter.prepare_scenario_calls == [spec]
    assert not adapter.create_runtime_called


def test_replay_demo_builds_output_target_from_spec(tmp_path: Path) -> None:
    writer_calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
        layout: str,
        install_hint: str,
    ) -> Path:
        del install_hint
        writer_calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
            }
        )
        return path

    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="replay",
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=12),
    )

    artifacts = run_replay_demo(
        spec=spec,
        adapter=_FakeDemoAdapter(video_output=True),
        output_target_factory=lambda output_spec: build_output_target(
            output_spec,
            mp4_writer=fake_writer,
        ),
    )

    assert len(artifacts) == 1
    assert artifacts[0].kind == "video/mp4"
    assert artifacts[0].uri == str(tmp_path / "demo.mp4")
    assert writer_calls == [
        {
            "shape": (2, 2, 2, 3),
            "path": tmp_path / "demo.mp4",
            "fps": 12,
            "layout": "thwc",
        }
    ]


def test_replay_demo_fails_before_runtime_creation_when_scenario_invalid() -> None:
    adapter = _FakeDemoAdapter(scenario_valid=False)
    output_factory_calls = 0

    def output_factory(output_spec: object) -> OutputTarget:
        nonlocal output_factory_calls
        del output_spec
        output_factory_calls += 1
        return NullOutputTarget()

    spec = DemoSpec(
        model_id="fake-demo",
        scenario="missing-scenario",
        input_mode="replay",
        output=NullOutputSpec(),
    )

    with pytest.raises(ValueError, match="invalid scenario"):
        run_replay_demo(
            spec=spec,
            adapter=adapter,
            output_target_factory=output_factory,
        )

    assert adapter.prepare_scenario_calls == [spec]
    assert not adapter.create_runtime_called
    assert output_factory_calls == 0


def test_demo_adapter_declares_supported_modes() -> None:
    adapter = _FakeDemoAdapter(
        input_modes=("replay",),
        output_modes=("null", "mp4", "webrtc"),
    )

    assert adapter.supported_input_modes() == ("replay",)
    assert adapter.supported_output_modes() == ("null", "mp4", "webrtc")

    with pytest.raises(ValueError, match="input_mode='keyboard-driving'"):
        run_replay_demo(
            spec=DemoSpec(
                model_id="fake-demo",
                scenario="valid-scenario",
                input_mode="keyboard-driving",
                output=NullOutputSpec(),
            ),
            adapter=adapter,
        )

    assert adapter.prepare_scenario_calls == []
    assert not adapter.create_runtime_called


def test_webrtc_demo_uses_existing_session_manager_with_adapter_runtime() -> None:
    adapter = _FakeDemoAdapter()
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            fps=24,
            video_width=16,
            video_height=8,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter)

    assert isinstance(demo.session_manager, BaseWebRTCSessionManager)
    assert demo.runtime is adapter.webrtc_runtime
    assert demo.session_manager._runtime is adapter.webrtc_runtime
    assert demo.session_manager.runtime_config.video_width == 16
    assert demo.session_manager.runtime_config.video_height == 8
    assert demo.session_manager.fps == 24
    assert demo.session_manager._model_name() == "fake-demo"
    assert demo.app is None
    assert demo.host == "0.0.0.0"
    assert demo.port == 8082
    assert adapter.create_webrtc_runtime_calls == [spec]
    assert not adapter.create_runtime_called


def test_webrtc_demo_builds_app_from_extension_routes() -> None:
    app_extension = WebRTCAppExtension(
        web_resource=files("flashdreams.runtime.demo"),
        preload_name="Fake WebRTC",
        routes=(
            json_get_route(
                "/api/fake/model-info",
                lambda manager: {"model": manager._model_name()},
            ),
        ),
    )
    adapter = _FakeDemoAdapter(app_extension=app_extension)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            fps=24,
            video_width=16,
            video_height=8,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter, create_app=True)

    assert demo.app is not None
    route_paths = {resource.canonical for resource in demo.app.router.resources()}
    assert "/api/fake/model-info" in route_paths
    assert adapter.create_webrtc_app_extension_calls == [
        {
            "spec": spec,
            "session_manager": demo.session_manager,
            "request_session_url": "http://127.0.0.1:8082/request_session",
        }
    ]
    demo.app[PACKAGE_RESOURCE_STACK_KEY].close()


def test_webrtc_demo_app_extension_keeps_configure_app_callback() -> None:
    async def fake_route(_: web.Request) -> web.StreamResponse:
        return web.json_response({"ok": True})

    def configure_app(app: web.Application) -> None:
        app.router.add_get("/api/fake/configured", fake_route)

    app_extension = WebRTCAppExtension(
        web_resource=files("flashdreams.runtime.demo"),
        configure_app=configure_app,
    )
    adapter = _FakeDemoAdapter(app_extension=app_extension)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            fps=24,
            video_width=16,
            video_height=8,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter, create_app=True)

    assert demo.app is not None
    route_paths = {resource.canonical for resource in demo.app.router.resources()}
    assert "/api/fake/configured" in route_paths
    demo.app[PACKAGE_RESOURCE_STACK_KEY].close()


@pytest.mark.asyncio
async def test_webrtc_demo_manager_options_configure_shared_manager() -> None:
    class FakeRuntimeError(RuntimeError):
        pass

    reset_calls: list[tuple[Any, Any]] = []
    pending_inputs: list[object] = [object()]
    clear_calls = 0
    peer_hooks: list[Any] = []
    offers: list[str] = []
    answers: list[str] = []

    async def reset_runtime(runtime: Any, session_input: Any) -> None:
        reset_calls.append((runtime, session_input))

    def clear_pending() -> None:
        nonlocal clear_calls
        clear_calls += 1

    manager_options = WebRTCManagerOptions(
        model_name="fake-model-v2",
        busy_message="fake session busy",
        warmup_label="Fake Warmup",
        runtime_error_types=(FakeRuntimeError,),
        close_session_on_generation_error=True,
        supported_keys=frozenset({"w"}),
        peek_pending_session_input=lambda: pending_inputs[0],
        clear_pending_session_input=clear_pending,
        reset_runtime_for_session=reset_runtime,
        chunk_done_extra=lambda runtime, runtime_config: {
            "runtime": runtime.marker,
            "width": runtime_config.video_width,
        },
        register_extra_peer_handlers=peer_hooks.append,
        on_offer_received=offers.append,
        on_answer_created=answers.append,
    )
    adapter = _FakeDemoAdapter(manager_options=manager_options)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(
            fps=24,
            video_width=16,
            video_height=8,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter)
    manager = demo.session_manager

    assert isinstance(manager, SharedDemoWebRTCSessionManager)
    assert manager._model_name() == "fake-model-v2"
    assert manager._busy_message == "fake session busy"
    assert manager._warmup_label == "Fake Warmup"
    assert manager._runtime_error_types == (FakeRuntimeError,)
    assert manager._close_session_on_generation_error is True
    assert manager._peek_pending_session_input() is pending_inputs[0]
    manager._clear_pending_session_input()
    assert clear_calls == 1
    await manager._reset_runtime_for_session(pending_inputs[0])
    assert reset_calls == [(demo.runtime, pending_inputs[0])]
    assert manager._chunk_done_extra() == {"runtime": "fake-webrtc", "width": 16}

    peer = object()
    manager._register_extra_peer_handlers(peer)
    manager._on_offer_received("offer-sdp")
    manager._on_answer_created("answer-sdp")
    assert peer_hooks == [peer]
    assert offers == ["offer-sdp"]
    assert answers == ["answer-sdp"]

    resampler = manager._make_resampler(start_v=1.0)
    resampler.on_edge(arrival_t=0.5, event="keydown", key="q")
    segments, _ = resampler.sample_chunk(num_frames=1)
    assert segments[0][2] == frozenset()

    resampler = manager._make_resampler(start_v=1.0)
    resampler.on_edge(arrival_t=0.5, event="keydown", key="w")
    segments, _ = resampler.sample_chunk(num_frames=1)
    assert segments[0][2] == frozenset({"w"})


@pytest.mark.asyncio
async def test_json_get_route_builds_payload_from_shared_manager() -> None:
    class _Manager:
        def _model_name(self) -> str:
            return "fake-demo"

    route = json_get_route(
        "/api/fake/model-info",
        lambda manager: {"model": manager._model_name()},
    )
    request = make_mocked_request("GET", "/api/fake/model-info")

    response = await route.handler(request, _Manager())

    assert _json_response_payload(response) == {"model": "fake-demo"}


def test_webrtc_route_normalizes_method_and_requires_absolute_path() -> None:
    route = WebRTCRoute(
        method=" get ",
        path="/api/fake",
        handler=lambda _request, _manager: web.json_response({}),
    )

    assert route.method == "GET"
    assert route.path == "/api/fake"

    with pytest.raises(ValueError, match="path must start"):
        WebRTCRoute(
            method="GET",
            path="api/fake",
            handler=lambda _request, _manager: web.json_response({}),
        )


def test_pending_session_input_state_stores_valid_input_and_rejects_busy() -> None:
    class _Manager:
        active = False

        def has_active_session(self) -> bool:
            return self.active

    validated: list[str] = []
    manager = _Manager()
    state = PendingSessionInputState(
        busy_message="fake session busy",
        input_type=str,
        validate_input=validated.append,
    )

    state.set(manager, "accepted")

    assert state.peek() == "accepted"
    assert validated == ["accepted"]
    state.clear()
    assert state.peek() is None

    with pytest.raises(TypeError, match="Expected str"):
        state.set(manager, object())

    manager.active = True
    with pytest.raises(SessionBusyError, match="fake session busy"):
        state.set(manager, "next")


@pytest.mark.asyncio
async def test_session_input_route_applies_pending_input_and_maps_errors() -> None:
    class _Manager:
        active = False

        def __init__(self) -> None:
            self.state = PendingSessionInputState(
                busy_message="fake session busy",
                input_type=str,
                validate_input=self._validate,
            )

        def has_active_session(self) -> bool:
            return self.active

        def set_pending_session_input(self, session_input: str) -> None:
            self.state.set(self, session_input)

        def _validate(self, session_input: str) -> None:
            if session_input == "invalid":
                raise ValueError("invalid input")

    manager = _Manager()
    request = make_mocked_request("POST", "/api/fake/session/input")
    route = session_input_route(
        "/api/fake/session/input",
        parse_input=lambda _request, _manager: "accepted",
        build_response=lambda session_input, _manager: {
            "accepted": session_input,
        },
    )

    response = await route.handler(request, manager)

    assert _json_response_payload(response) == {"accepted": "accepted"}
    assert manager.state.peek() == "accepted"

    invalid_route = session_input_route(
        "/api/fake/session/input",
        parse_input=lambda _request, _manager: "invalid",
    )
    with pytest.raises(web.HTTPBadRequest, match="invalid input"):
        await invalid_route.handler(request, manager)

    wrong_type_route = session_input_route(
        "/api/fake/session/input",
        parse_input=lambda _request, _manager: object(),
    )
    with pytest.raises(web.HTTPBadRequest, match="Expected str"):
        await wrong_type_route.handler(request, manager)

    manager.active = True
    with pytest.raises(web.HTTPConflict, match="fake session busy"):
        await route.handler(request, manager)


def test_webrtc_app_extension_rejects_ambiguous_static_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either web_resource or web_dir"):
        WebRTCAppExtension(web_resource=object(), web_dir=tmp_path)


def test_webrtc_app_extension_requires_static_source_before_serving() -> None:
    adapter = _FakeDemoAdapter(
        app_extension=WebRTCAppExtension(configure_app=lambda app: None)
    )
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(
            fps=24,
            video_width=16,
            video_height=8,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
    )

    with pytest.raises(ValueError, match="requires web_resource or web_dir"):
        build_webrtc_demo(spec=spec, adapter=adapter, create_app=True)


class _ChunkIndexMapping:
    mapping_schema = InputMappingSchema(
        name="chunk-index",
        produces_global_conditioning=(InputField(name="prompt"),),
        produces_step=(InputField(name="chunk_index"),),
    )

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        del canonical_schema, inference_input_schema

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del canonical_inputs
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={"chunk_index": request.step_index},
            metadata=inference_input.metadata,
        )


class _FakeDemoAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(
        self,
        *,
        scenario_valid: bool = True,
        video_output: bool = False,
        input_modes: tuple[str, ...] = ("replay", "keyboard-driving"),
        output_modes: tuple[str, ...] = ("null", "mp4", "webrtc"),
        manager_options: WebRTCManagerOptions | None = None,
        app_extension: WebRTCAppExtension | None = None,
    ) -> None:
        self._scenario_valid = scenario_valid
        self._video_output = video_output
        self._input_modes = input_modes
        self._output_modes = output_modes
        self._manager_options = manager_options
        self._app_extension = app_extension
        self.mapping = _ChunkIndexMapping()
        self.prepared_scenario = PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"},
            ),
            user_inputs=UserInputs(),
            source_schema=UserInputSchema(),
            canonicalizer=InputCanonicalizer(),
            mapping=self.mapping,
        )
        self.prepare_scenario_calls: list[DemoSpec] = []
        self.create_runtime_called = False
        self.runtime: _FakeRuntime | None = None
        self.webrtc_runtime: _FakeWebRTCRuntime | None = None
        self.create_webrtc_runtime_calls: list[DemoSpec] = []
        self.create_webrtc_manager_options_calls: list[dict[str, Any]] = []
        self.create_webrtc_app_extension_calls: list[dict[str, Any]] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return self._input_modes

    def supported_output_modes(self) -> tuple[str, ...]:
        return self._output_modes

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.create_runtime_called = True
        self.runtime = _FakeRuntime(
            inference_input_schema=self.inference_input_schema,
            video_output=self._video_output,
        )
        return self.runtime

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        self.prepare_scenario_calls.append(spec)
        if not self._scenario_valid:
            raise ValueError("invalid scenario")
        return self.prepared_scenario

    def create_webrtc_runtime(self, spec: DemoSpec) -> "_FakeWebRTCRuntime":
        self.create_webrtc_runtime_calls.append(spec)
        self.webrtc_runtime = _FakeWebRTCRuntime()
        return self.webrtc_runtime

    def create_webrtc_manager_options(
        self,
        *,
        spec: DemoSpec,
        runtime: "_FakeWebRTCRuntime",
        runtime_config: Any,
    ) -> WebRTCManagerOptions:
        self.create_webrtc_manager_options_calls.append(
            {
                "spec": spec,
                "runtime": runtime,
                "runtime_config": runtime_config,
            }
        )
        return self._manager_options or WebRTCManagerOptions()

    def create_webrtc_app_extension(
        self,
        *,
        spec: DemoSpec,
        session_manager: BaseWebRTCSessionManager[Any, Any],
        request_session_url: str,
    ) -> WebRTCAppExtension | None:
        self.create_webrtc_app_extension_calls.append(
            {
                "spec": spec,
                "session_manager": session_manager,
                "request_session_url": request_session_url,
            }
        )
        return self._app_extension


class _FakeRuntime:
    def __init__(
        self,
        *,
        inference_input_schema: InferenceInputSchema,
        video_output: bool,
    ) -> None:
        self._inference_input_schema = inference_input_schema
        self._video_output = video_output
        self.session: _FakeSession | None = None
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self._inference_input_schema.require_global_conditioning(inputs)
        self.session = _FakeSession(
            inference_input_schema=self._inference_input_schema,
            video_output=self._video_output,
        )
        return self.session

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(
        self,
        *,
        inference_input_schema: InferenceInputSchema,
        video_output: bool,
    ) -> None:
        self._inference_input_schema = inference_input_schema
        self._video_output = video_output
        self.step_index = 0
        self.closed = False

    def next_step_request(self) -> StepRequest | None:
        if self.step_index >= 2:
            return None
        return StepRequest(
            step_index=self.step_index,
            user_input_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self._inference_input_schema.require_step(inputs)
        output: object
        if self._video_output:
            output = VideoStepResult.from_video_chunk(
                chunk_index=self.step_index,
                video_chunk=torch.full(
                    (1, 1, 1, 3, 2, 2),
                    self.step_index,
                    dtype=torch.float32,
                ),
                layout="bvtchw",
            )
        else:
            output = f"chunk-{self.step_index}"
        result = StepResult(
            step_index=self.step_index,
            output=output,
            frame_count=1,
            output_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
        )
        self.step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.step_index = 0

    def close(self) -> None:
        self.closed = True


class _RecordingOutputTarget:
    def open(self) -> None:
        return None

    def write(self, result: StepResult) -> None:
        del result

    def close(self) -> Sequence[OutputArtifact]:
        return ()


class _FakeWebRTCRuntime:
    marker = "fake-webrtc"

    async def initialize(self) -> None:
        return None

    async def reset_for_new_session(self) -> None:
        return None

    def peek_steady_chunk_num_frames(self) -> int:
        return 1

    def peek_next_chunk_num_frames(self) -> int:
        return 1

    async def generate_chunk(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> Any:
        del segments, frame_times
        return None

    async def close(self) -> None:
        return None

    def send_exit_signal(self) -> None:
        return None

    def wait_for_termination(self) -> None:
        return None
