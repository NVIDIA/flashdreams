# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from flashdreams.demo import (
    Application,
    ApplicationSession,
    CallbackIOHandlerServer,
    DemoAdapterApplication,
    DemoApplication,
    FrameOutputSink,
    InferenceSessionApplicationAdapter,
    IOHandler,
    IOHandlerServer,
    ReplayIOHandler,
    Runner,
    RuntimeOutputSinkFrameAdapter,
    WebRTCIOHandlerServer,
    create_native_window_io_handler,
    create_replay_io_handler,
    create_webrtc_io_handler,
)
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputField,
    InputMapping,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    BatchSessionDriver,
    DemoSpec,
    InMemorySessionMetricsRecorder,
    NullOutputSink,
    NullOutputSpec,
    OutputDecision,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    RunContext,
    RunModeCapabilities,
    RunResult,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
    StepPipeline,
    UserInputWindow,
)
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequirements

pytestmark = pytest.mark.ci_cpu


def test_public_demo_contracts_are_importable() -> None:
    assert Application.__name__ == "Application"
    assert ApplicationSession.__name__ == "ApplicationSession"
    assert DemoApplication.__name__ == "DemoApplication"
    assert IOHandler.__name__ == "IOHandler"
    assert IOHandlerServer.__name__ == "IOHandlerServer"
    assert FrameOutputSink.__name__ == "FrameOutputSink"
    assert create_replay_io_handler.__name__ == "create_replay_io_handler"
    assert create_native_window_io_handler.__name__ == (
        "create_native_window_io_handler"
    )
    assert create_webrtc_io_handler.__name__ == "create_webrtc_io_handler"


def test_inference_session_adapter_satisfies_application_session() -> None:
    session = _FakeSession()
    adapter = InferenceSessionApplicationAdapter(session)

    assert isinstance(adapter, ApplicationSession)
    adapter.init()
    requirements = adapter.next_step_requirements()

    assert requirements == StepRequirements(
        step_index=0,
        inference_input_schema=session.inference_input_schema,
    )
    result = adapter.step(InferenceInput(step={"chunk_index": 0}))
    assert result.step_index == 0
    assert adapter.session_info() == SessionInfo(output_layout="thwc")
    adapter.reset()
    adapter.close()
    assert session.initialized
    assert session.reset_called
    assert session.closed


def test_demo_adapter_application_satisfies_application() -> None:
    adapter = _FakeDemoAdapter()
    demo = DemoAdapterApplication(
        adapter=adapter,
        spec=DemoSpec(
            model_id="fake-demo",
            input_mode="replay",
            output=NullOutputSpec(),
        ),
    )

    assert isinstance(demo, Application)
    demo.init(())
    session = demo.create_session()

    assert isinstance(session, ApplicationSession)
    assert session.next_step_requirements() == StepRequirements(
        step_index=0,
        inference_input_schema=_FakeSession.inference_input_schema,
    )
    demo.close()
    assert adapter.runtime.closed


def test_demo_adapter_application_closes_all_created_runtimes() -> None:
    adapter = _FakeDemoAdapter()
    demo = DemoAdapterApplication(
        adapter=adapter,
        spec=DemoSpec(
            model_id="fake-demo",
            input_mode="replay",
            output=NullOutputSpec(),
        ),
    )

    demo.init(())
    demo.create_session()
    demo.create_session()
    demo.close()
    demo.close()

    assert len(adapter.runtimes) == 2
    assert [runtime.closed for runtime in adapter.runtimes] == [True, True]


def test_demo_adapter_application_rejects_unsupported_launch_args() -> None:
    demo = DemoAdapterApplication(
        adapter=_FakeDemoAdapter(),
        spec=DemoSpec(
            model_id="fake-demo",
            input_mode="replay",
            output=NullOutputSpec(),
        ),
    )

    with pytest.raises(ValueError, match="does not support launch arguments"):
        demo.init(("--device", "cuda:0"))


def test_io_handler_protocol_keeps_input_conversion_outside_io() -> None:
    handler = _FakeIOHandler()

    assert isinstance(handler, IOHandler)
    window = handler.next_window(StepRequirements(step_index=3))

    assert window.start_s == 3.0
    assert window.end_s == 4.0
    assert handler.get_user_input_state("keyboard", "key_w") is False


def test_runtime_output_sink_frame_adapter_satisfies_frame_output_sink() -> None:
    output = NullOutputSink(store_results=True)
    output.open(SessionInfo())
    adapter = RuntimeOutputSinkFrameAdapter(output)

    assert isinstance(adapter, FrameOutputSink)
    adapter.handle_output(
        0.0,
        StepResult(
            step_index=0,
            output="chunk",
            frame_count=1,
            output_window=TimeWindow(start_s=0.0, end_s=1.0),
        ),
    )

    assert output.output_count == 1


def test_runner_run_drives_public_app_through_shared_runtime_path() -> None:
    caller_thread_id = threading.get_ident()
    app = _RunnerFakeApplication(total_steps=2)
    io_handler = _RecordingIOHandler()

    result = Runner(
        io_handler=io_handler,
        app=app,
        launch_args=("--quality", "fast"),
        model_id="fake-runner",
    ).run()

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert app.launch_args == ("--quality", "fast")
    assert app.init_thread_id == caller_thread_id
    assert app.session is not None
    assert app.session.init_thread_id != caller_thread_id
    assert app.session.step_thread_ids == (app.session.init_thread_id,) * 2
    assert app.session.closed
    assert app.closed
    assert io_handler.opened_with == [SessionInfo(output_layout="thwc")]
    assert io_handler.requested_steps == [0, 1]
    assert io_handler.begin_generations == [0]
    assert io_handler.emitted_steps == [0, 1]
    assert io_handler.closed


def test_runner_closes_public_app_when_host_is_external() -> None:
    app = _RunnerFakeApplication(total_steps=1)
    host = RuntimeHost(_ExternalApplicationRuntime(app))

    try:
        result = Runner(
            io_handler=_RecordingIOHandler(),
            app=app,
            host=host,
            model_id="fake-runner",
        ).run()
    finally:
        host.close()

    assert result.status == "completed"
    assert app.closed


def test_runner_closes_public_app_when_context_cleanup_fails() -> None:
    app = _RunnerFakeApplication(total_steps=1)

    with pytest.raises(RuntimeError, match="run metrics close failed"):
        Runner(
            io_handler=_RecordingIOHandler(),
            app=app,
            metrics=_FailingCloseMetricsRecorder(),
            model_id="fake-runner",
        ).run()

    assert app.closed


def test_runner_preserves_primary_error_when_cleanup_fails() -> None:
    app = _RunnerFakeApplication(total_steps=1, fail_init=True)

    with pytest.raises(RuntimeError, match="fake init failed") as exc_info:
        Runner(
            io_handler=_RecordingIOHandler(),
            app=app,
            metrics=_FailingCloseMetricsRecorder(),
            model_id="fake-runner",
        ).run()

    notes = getattr(exc_info.value, "__notes__", ())
    assert any("run metrics close failed" in note for note in notes)
    assert app.closed


def test_runner_preserves_failed_result_when_cleanup_fails() -> None:
    app = _RunnerFakeApplication(total_steps=1, fail_step=0)

    result = Runner(
        io_handler=_RecordingIOHandler(),
        app=app,
        metrics=_FailingCloseMetricsRecorder(),
        model_id="fake-runner",
    ).run()

    assert result.status == "failed"
    assert result.error is not None
    assert str(result.error) == "fake step failed"
    assert app.closed


@pytest.mark.asyncio
async def test_runner_run_async_delegates_to_async_session_helper() -> None:
    app = _RunnerFakeApplication(total_steps=1)
    io_handler = _RecordingIOHandler()
    run_mode = _AsyncRecordingRunMode(io_handler)

    result = await Runner(
        io_handler=io_handler,
        app=app,
        run_mode=run_mode,
        model_id="fake-runner",
    ).run_async()

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 1
    assert run_mode.driver.called
    assert io_handler.emitted_steps == [0]


@pytest.mark.asyncio
async def test_runner_run_async_closes_public_app_when_context_cleanup_fails() -> None:
    app = _RunnerFakeApplication(total_steps=1)

    with pytest.raises(RuntimeError, match="run metrics close failed"):
        await Runner(
            io_handler=_RecordingIOHandler(),
            app=app,
            metrics=_FailingCloseMetricsRecorder(),
            run_mode=_AsyncRecordingRunMode(_RecordingIOHandler()),
            model_id="fake-runner",
        ).run_async()

    assert app.closed


@pytest.mark.asyncio
async def test_runner_run_async_preserves_primary_error_when_cleanup_fails() -> None:
    app = _RunnerFakeApplication(total_steps=1, fail_init=True)

    with pytest.raises(RuntimeError, match="fake init failed") as exc_info:
        await Runner(
            io_handler=_RecordingIOHandler(),
            app=app,
            metrics=_FailingCloseMetricsRecorder(),
            run_mode=_AsyncRecordingRunMode(_RecordingIOHandler()),
            model_id="fake-runner",
        ).run_async()

    notes = getattr(exc_info.value, "__notes__", ())
    assert any("run metrics close failed" in note for note in notes)
    assert app.closed


@pytest.mark.asyncio
async def test_runner_run_async_preserves_failed_result_when_cleanup_fails() -> None:
    app = _RunnerFakeApplication(total_steps=1, fail_step=0)

    result = await Runner(
        io_handler=_RecordingIOHandler(),
        app=app,
        metrics=_FailingCloseMetricsRecorder(),
        run_mode=_AsyncRecordingRunMode(_RecordingIOHandler()),
        model_id="fake-runner",
    ).run_async()

    assert result.status == "failed"
    assert result.error is not None
    assert str(result.error) == "fake step failed"
    assert app.closed


def test_replay_io_factory_runs_through_public_runner() -> None:
    output_sink = NullOutputSink(store_results=True)
    io_handler = create_replay_io_handler(output_sink=output_sink)
    app = _RunnerFakeApplication(total_steps=2)

    assert isinstance(io_handler, ReplayIOHandler)
    result = Runner(
        io_handler=io_handler,
        app=app,
        model_id="fake-replay-factory",
    ).run()

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert output_sink.output_count == 2


def test_replay_io_factory_forwards_to_metric_tail() -> None:
    metric_tail = _RecordingFrameOutputSink()
    io_handler = create_replay_io_handler(metric_output_sink=metric_tail)

    result = Runner(
        io_handler=io_handler,
        app=_RunnerFakeApplication(total_steps=1),
        model_id="fake-replay-factory",
    ).run()

    assert result.status == "completed"
    assert metric_tail.records == [(0.0, 0)]


def test_native_window_factory_exposes_io_handler_shape() -> None:
    io_handler = create_native_window_io_handler((1280, 720))

    assert isinstance(io_handler, IOHandler)
    assert io_handler.run_mode.name == "native-window"
    with pytest.raises(ValueError, match="viewport dimensions"):
        create_native_window_io_handler((0, 720))


def test_webrtc_io_factory_returns_server_not_ready_handler() -> None:
    handler = _RecordingIOHandler()
    server = create_webrtc_io_handler(
        "127.0.0.1",
        8080,
        (1280, 720),
        handlers=(handler,),
    )
    calls: list[IOHandler] = []

    def run_session(io_handler: IOHandler) -> RunResult:
        calls.append(io_handler)
        return RunResult(status="completed")

    assert isinstance(server, IOHandlerServer)
    assert isinstance(server, WebRTCIOHandlerServer)
    result = server.serve(run_session)

    assert result.status == "completed"
    assert calls == [handler]
    assert not isinstance(server, IOHandler)


def test_demo_application_replay_selects_factory_and_runner() -> None:
    app = _ReplayDemoApplication()

    app.main(["replay"])

    assert app.adapter.runtimes
    runtime = app.adapter.runtimes[0]
    assert runtime.session.closed
    assert runtime.closed


def test_demo_application_server_selection_does_not_build_replay_app() -> None:
    app = _ServerDemoApplication()

    app.main(["webrtc"])

    assert app.server_called
    assert not app.replay_adapter_called


class _ReplayDemoApplication(DemoApplication):
    def __init__(self) -> None:
        self.adapter = _FakeDemoAdapter()

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=("replay",))
        return parser.parse_args(argv)

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        assert args.command == "replay"
        return DemoSpec(
            model_id="fake-demo",
            input_mode="replay",
            output=NullOutputSpec(),
        )

    def replay_adapter(self) -> "_FakeDemoAdapter":
        return self.adapter


class _ServerDemoApplication(DemoApplication):
    def __init__(self) -> None:
        self.server_called = False
        self.replay_adapter_called = False

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=("webrtc",))
        return parser.parse_args(argv)

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        del args
        raise AssertionError("server selection should not build a replay spec")

    def replay_adapter(self) -> "_FakeDemoAdapter":
        self.replay_adapter_called = True
        return _FakeDemoAdapter()

    def create_io_handler(
        self,
        args: argparse.Namespace,
    ) -> IOHandler | IOHandlerServer:
        assert args.command == "webrtc"

        def serve() -> RunResult:
            self.server_called = True
            return RunResult(status="completed")

        return CallbackIOHandlerServer(serve)


class _FakeSession:
    inference_input_schema = InferenceInputSchema(
        step_fields=(InputField(name="chunk_index"),)
    )

    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.reset_called = False
        self.step_index = 0

    def init(self) -> None:
        self.initialized = True

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="thwc")

    def next_step_request(self) -> StepRequest | None:
        if self.step_index >= 1:
            return None
        return StepRequest(
            step_index=self.step_index,
            inference_input_schema=self.inference_input_schema,
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self.inference_input_schema.require_step(inputs)
        result = StepResult(
            step_index=self.step_index,
            output="chunk",
            frame_count=1,
            output_window=TimeWindow(start_s=0.0, end_s=1.0),
        )
        self.step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.reset_called = True

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        assert inputs.global_conditioning["prompt"] == "demo"
        return self.session

    def close(self) -> None:
        self.closed = True


class _FakeDemoAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self) -> None:
        self.runtime = _FakeRuntime()
        self.runtimes: list[_FakeRuntime] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        assert config.model_id == self.model_id

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.runtime = _FakeRuntime()
        self.runtimes.append(self.runtime)
        return self.runtime

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        assert spec.model_id == self.model_id
        return PreparedScenario(
            initial_inputs=InferenceInput(global_conditioning={"prompt": "demo"}),
            user_inputs=UserInputs(),
            source_schema=UserInputSchema(),
        )

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> "_FakeModelInputProvider":
        assert spec.model_id == self.model_id
        return _FakeModelInputProvider(
            scenario=scenario,
            inference_input_schema=self.inference_input_schema,
        )


class _FakeModelInputProvider:
    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        inference_input_schema: InferenceInputSchema,
    ) -> None:
        self._scenario = scenario
        self.capabilities = ProviderCapabilities(
            supports_recorded_input=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=inference_input_schema,
        )
        self.closed = False

    def prepare_initial_input(self) -> InferenceInput:
        return self._scenario.initial_inputs

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del user_window
        return PreparedStep(
            inference_input=InferenceInput(step={"chunk_index": request.step_index})
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.closed = True


class _FakeIOHandler:
    def open(self, session_info: SessionInfo) -> None:
        del session_info

    def next_window(self, requirements: StepRequirements) -> UserInputWindow:
        start_s = float(requirements.step_index)
        return UserInputWindow(start_s=start_s, end_s=start_s + 1.0)

    def get_user_input_state(self, modality: str, name: str) -> Any:
        assert modality == "keyboard"
        assert name == "key_w"
        return False

    def begin_generation(self, generation: int) -> None:
        assert generation >= 0

    def emit_chunk(self, result: StepResult) -> OutputDecision:
        assert result.step_index >= 0
        return OutputDecision()

    def should_exit(self) -> bool:
        return False

    def close(self) -> Sequence[OutputArtifact]:
        return ()


class _RecordingFrameOutputSink:
    def __init__(self) -> None:
        self.records: list[tuple[float, int]] = []

    def handle_output(self, timestamp_s: float, chunk: StepResult) -> None:
        self.records.append((timestamp_s, chunk.step_index))


class _RunnerFakeApplication:
    model_id = "runner-fake"

    def __init__(
        self,
        *,
        total_steps: int,
        fail_init: bool = False,
        fail_step: int | None = None,
    ) -> None:
        self.total_steps = total_steps
        self.fail_init = fail_init
        self.fail_step = fail_step
        self.launch_args: tuple[str, ...] = ()
        self.init_thread_id: int | None = None
        self.session: _RunnerFakeSession | None = None
        self.closed = False

    def init(self, launch_args: Sequence[str]) -> None:
        self.launch_args = tuple(launch_args)
        self.init_thread_id = threading.get_ident()
        if self.fail_init:
            raise RuntimeError("fake init failed")

    def create_session(self) -> "_RunnerFakeSession":
        self.session = _RunnerFakeSession(
            total_steps=self.total_steps,
            fail_step=self.fail_step,
        )
        return self.session

    def close(self) -> None:
        self.closed = True


class _ExternalApplicationRuntime:
    def __init__(self, app: _RunnerFakeApplication) -> None:
        self._app = app
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        session = self._app.create_session()
        session.init()
        return _ExternalInferenceSession(session)

    def close(self) -> None:
        self.closed = True


class _FailingCloseMetricsRecorder(InMemorySessionMetricsRecorder):
    def close(self) -> Any:
        raise RuntimeError("run metrics close failed")


class _ExternalInferenceSession:
    def __init__(self, session: "_RunnerFakeSession") -> None:
        self._session = session

    def session_info(self) -> SessionInfo:
        return self._session.session_info()

    def next_step_request(self) -> StepRequest | None:
        requirements = self._session.next_step_requirements()
        if requirements is None:
            return None
        return StepRequest(step_index=requirements.step_index)

    def step(self, inputs: InferenceInput) -> StepResult:
        return self._session.step(inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self._session.reset(inputs)

    def close(self) -> None:
        self._session.close()


class _RunnerFakeSession:
    def __init__(self, *, total_steps: int, fail_step: int | None) -> None:
        self.total_steps = total_steps
        self.fail_step = fail_step
        self.step_index = 0
        self.init_thread_id: int | None = None
        self.step_thread_ids: tuple[int, ...] = ()
        self.closed = False

    def init(self) -> None:
        self.init_thread_id = threading.get_ident()

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="thwc")

    def next_step_requirements(self) -> StepRequirements | None:
        if self.step_index >= self.total_steps:
            return None
        return StepRequirements(step_index=self.step_index)

    def step(self, model_input: InferenceInput) -> StepResult:
        assert model_input.step["step_index"] == self.step_index
        assert isinstance(model_input.step["user_window"], UserInputWindow)
        if self.fail_step == self.step_index:
            raise RuntimeError("fake step failed")
        self.step_thread_ids = (*self.step_thread_ids, threading.get_ident())
        result = StepResult(
            step_index=self.step_index,
            output=f"runner-chunk-{self.step_index}",
            frame_count=1,
            output_window=TimeWindow(
                start_s=float(self.step_index),
                end_s=float(self.step_index + 1),
            ),
        )
        self.step_index += 1
        return result

    def reset(self, model_input: InferenceInput | None = None) -> None:
        del model_input
        self.step_index = 0

    def close(self) -> None:
        self.closed = True


class _RecordingIOHandler:
    def __init__(self) -> None:
        self.opened_with: list[SessionInfo] = []
        self.requested_steps: list[int] = []
        self.begin_generations: list[int] = []
        self.emitted_steps: list[int] = []
        self.closed = False

    def open(self, session_info: SessionInfo) -> None:
        self.opened_with.append(session_info)

    def next_window(self, requirements: StepRequirements) -> UserInputWindow:
        self.requested_steps.append(requirements.step_index)
        start_s = float(requirements.step_index)
        return UserInputWindow(start_s=start_s, end_s=start_s + 1.0)

    def get_user_input_state(self, modality: str, name: str) -> Any:
        del modality, name
        return None

    def begin_generation(self, generation: int) -> None:
        self.begin_generations.append(generation)

    def emit_chunk(self, result: StepResult) -> OutputDecision:
        self.emitted_steps.append(result.step_index)
        return OutputDecision()

    def should_exit(self) -> bool:
        return False

    def close(self) -> Sequence[OutputArtifact]:
        self.closed = True
        return ()


@dataclass(slots=True)
class _AsyncRecordingRunMode:
    io_handler: _RecordingIOHandler
    name: str = "async-public-runner"
    capabilities: RunModeCapabilities = field(
        default_factory=lambda: RunModeCapabilities(supports_artifacts=True)
    )
    driver: "_AsyncBatchDriver" = field(default_factory=lambda: _AsyncBatchDriver())

    def validate_run(self, *, spec: DemoSpec, adapter: Any) -> None:
        del spec, adapter

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: Any,
        adapter: Any,
        provider: Any,
    ) -> None:
        del spec, scenario, adapter, provider

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: Any,
        host: Any,
        model_warmup_plan: Any,
    ) -> RunContext:
        del spec, adapter
        return RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy,
            ),
            model_warmup_plan=model_warmup_plan,
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: Any,
        provider: Any,
        adapter: Any,
    ) -> SessionEdges:
        del spec, scenario, provider, adapter
        return SessionEdges(
            input_source=_AsyncRunModeInputSource(self.io_handler),
            output_sink=_AsyncRunModeOutputSink(self.io_handler),
            cleanup_tasks=context.cleanup_tasks,
        )

    def select_driver(self) -> "_AsyncBatchDriver":
        return self.driver


class _AsyncBatchDriver:
    def __init__(self) -> None:
        self.called = False

    async def run_one_session(
        self,
        *,
        host: Any,
        provider: Any,
        session_edges: SessionEdges,
        pipeline: StepPipeline,
    ) -> RunResult:
        self.called = True
        return BatchSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=session_edges,
            pipeline=pipeline,
        )


class _AsyncRunModeInputSource:
    is_finite = False
    is_deterministic = False
    user_input_schema = UserInputSchema()

    def __init__(self, io_handler: _RecordingIOHandler) -> None:
        self._io_handler = io_handler

    def is_finished(self) -> bool:
        return self._io_handler.should_exit()

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        return self._io_handler.next_window(request)


class _AsyncRunModeOutputSink:
    produces_artifacts = True

    def __init__(self, io_handler: _RecordingIOHandler) -> None:
        self._io_handler = io_handler
        self._begun = False

    def open(self, session_info: SessionInfo) -> None:
        self._io_handler.open(session_info)

    def begin_generation(self, generation: int) -> None:
        self._begun = True
        self._io_handler.begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        if not self._begun:
            self.begin_generation(0)
        return self._io_handler.emit_chunk(result)

    def close(self) -> Sequence[OutputArtifact]:
        return self._io_handler.close()
