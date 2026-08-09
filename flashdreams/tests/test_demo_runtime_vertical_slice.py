# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import pytest

from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputMapping,
    OutputArtifact,
    StepRequest,
    StepResult,
    UserInputs,
)
from flashdreams.runtime.demo import (
    BatchSessionDriver,
    ControlDecision,
    DemoSpec,
    DriverInvariantError,
    ErrorAction,
    InMemorySessionMetricsRecorder,
    NullOutputSpec,
    OutputDecision,
    PreparedScenario,
    PreparedStep,
    RunContext,
    RunResult,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
    StepPipeline,
    UserInputWindow,
    run_demo_session,
)

pytestmark = pytest.mark.ci_cpu


def test_step_pipeline_passes_provider_input_to_session_and_sink() -> None:
    provider = _FakeVideoModelInputProvider()
    session = _FakeVideoSession(num_steps=1)
    output = _RecordingOutputSink()
    output.open(SessionInfo(output_layout="fake-video", steady_output_frame_count=1))
    metrics = InMemorySessionMetricsRecorder()
    request = StepRequest(step_index=0)
    user_window = _window(0)

    outcome = StepPipeline().execute_step(
        request=request,
        user_window=user_window,
        provider=provider,
        session=session,
        output=output,
        metrics=metrics,
    )

    assert outcome == _empty_step_outcome()
    assert session.step_inputs == provider.prepared_step_inputs
    assert [result.output for result in output.results] == ["frame-0"]
    assert metrics.step_count == 1


def test_batch_driver_runs_fake_video_demo_through_runtime_host() -> None:
    session = _FakeVideoSession(num_steps=2)
    runtime = _FakeVideoRuntime(session=session)
    host = _RecordingRuntimeHost(runtime)
    provider = _FakeVideoModelInputProvider()
    output = _RecordingOutputSink()
    metrics = InMemorySessionMetricsRecorder()
    edges = SessionEdges(
        input_source=_FakeBatchInputSource(num_windows=2),
        output_sink=output,
        metrics=metrics,
    )

    result = BatchSessionDriver().run_one_session(
        host=host,
        provider=provider,
        session_edges=edges,
        pipeline=StepPipeline(),
    )

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert runtime.start_session_inputs == [provider.initial_input]
    assert [dict(inputs.step) for inputs in session.step_inputs] == [
        {"request_step": 0, "window": (0.0, 1.0)},
        {"request_step": 1, "window": (1.0, 2.0)},
    ]
    assert [result.output for result in output.results] == ["frame-0", "frame-1"]
    assert output.opened_with == SessionInfo(
        output_layout="fake-video",
        steady_output_frame_count=1,
    )
    assert session.close_count == 1
    assert provider.close_count == 1
    assert host.calls.count("execute_step") == 2
    assert "prepare_initial_input" in host.calls
    assert "start_session" in host.calls
    assert "prepare_step" not in host.calls
    assert "step" not in host.calls


def test_run_demo_session_builds_edges_and_records_session_once() -> None:
    session = _FakeVideoSession(num_steps=1)
    runtime = _FakeVideoRuntime(session=session)
    run_metrics = InMemorySessionMetricsRecorder()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _FakeVideoModelInputProvider()
    adapter = _FakeDemoAdapter(provider=provider)
    output = _RecordingOutputSink()
    factory_calls: list[tuple[DemoSpec, PreparedScenario]] = []
    run_mode = _FakeRunMode(
        input_source=_FakeBatchInputSource(num_windows=1),
        output_sink_factory=lambda spec, scenario: _record_output_factory_call(
            factory_calls,
            spec,
            scenario,
            output,
        ),
    )
    spec = _spec()
    scenario = _scenario()

    result = run_demo_session(
        context=context,
        spec=spec,
        scenario=scenario,
        adapter=adapter,
        run_mode=run_mode,
        pipeline=StepPipeline(),
    )

    assert result.status == "completed"
    assert adapter.provider_calls == [(spec, scenario)]
    assert factory_calls == [(spec, scenario)]
    assert run_metrics.sessions == [result]
    assert len(run_metrics.sessions) == 1
    new_reservation = context.admission.try_reserve()
    assert new_reservation is not None
    new_reservation.release()


def test_busy_admission_returns_rejected_and_records_once() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    admission = SingleSessionAdmissionPolicy()
    held = admission.try_reserve()
    assert held is not None
    run_metrics = InMemorySessionMetricsRecorder()
    context = _run_context(runtime, admission=admission, run_metrics=run_metrics)
    adapter = _FakeDemoAdapter(provider=_FakeVideoModelInputProvider())

    result = run_demo_session(
        context=context,
        spec=_spec(),
        scenario=_scenario(),
        adapter=adapter,
        run_mode=_FakeRunMode(input_source=_FakeBatchInputSource(num_windows=1)),
        pipeline=StepPipeline(),
    )

    held.release()
    assert result == RunResult.rejected(reason="busy")
    assert run_metrics.sessions == [result]
    assert adapter.provider_calls == []
    assert runtime.start_session_inputs == []


def test_setup_failure_returns_failed_before_runtime_session_creation() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    provider = _FakeVideoModelInputProvider(
        fail_initial=ValueError("invalid provider compatibility")
    )
    metrics = InMemorySessionMetricsRecorder()

    result = BatchSessionDriver().run_one_session(
        host=RuntimeHost(runtime),
        provider=provider,
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=_RecordingOutputSink(),
            metrics=metrics,
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "failed"
    assert isinstance(result.error, ValueError)
    assert result.reason == "invalid provider compatibility"
    assert runtime.start_session_inputs == []
    assert provider.close_count == 1
    assert metrics.errors == ["invalid provider compatibility"]


def test_run_demo_session_closes_provider_when_validation_fails() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    run_metrics = InMemorySessionMetricsRecorder()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _FakeVideoModelInputProvider()

    result = run_demo_session(
        context=context,
        spec=_spec(),
        scenario=_scenario(),
        adapter=_FakeDemoAdapter(provider=provider),
        run_mode=_FakeRunMode(
            input_source=_FakeBatchInputSource(num_windows=1),
            validate_error=ValueError("provider incompatible"),
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "failed"
    assert result.reason == "provider incompatible"
    assert provider.close_count == 1
    assert runtime.start_session_inputs == []
    assert run_metrics.sessions == [result]


def test_setup_failure_can_return_skipped_but_not_completed() -> None:
    skipped = BatchSessionDriver().run_one_session(
        host=RuntimeHost(_FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))),
        provider=_FakeVideoModelInputProvider(fail_initial=RuntimeError("skip me")),
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=_RecordingOutputSink(),
            error_policy=_SetupPolicy(result_status="skipped"),
        ),
        pipeline=StepPipeline(),
    )
    assert skipped.status == "skipped"
    assert skipped.error is None

    provider = _FakeVideoModelInputProvider(fail_initial=RuntimeError("bad policy"))
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = SessionEdges(
        input_source=_FakeBatchInputSource(num_windows=1),
        output_sink=output,
        metrics=metrics,
        error_policy=_SetupPolicy(result_status="completed"),
        transport=transport,
    )

    with pytest.raises(DriverInvariantError, match="Setup failures"):
        BatchSessionDriver().run_one_session(
            host=RuntimeHost(_FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))),
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )

    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert provider.close_count == 1


def test_run_demo_session_closes_edges_when_driver_invariant_escapes() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    run_metrics = InMemorySessionMetricsRecorder()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _FakeVideoModelInputProvider(
        fail_initial=RuntimeError("bad setup policy")
    )
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    session_metrics = InMemorySessionMetricsRecorder()

    with pytest.raises(DriverInvariantError, match="Setup failures"):
        run_demo_session(
            context=context,
            spec=_spec(),
            scenario=_scenario(),
            adapter=_FakeDemoAdapter(provider=provider),
            run_mode=_FakeRunMode(
                input_source=_FakeBatchInputSource(num_windows=1),
                output_sink=output,
                metrics=session_metrics,
                transport=transport,
                error_policy=_SetupPolicy(result_status="completed"),
            ),
            pipeline=StepPipeline(),
        )

    assert output.close_count == 1
    assert transport.close_count == 1
    assert session_metrics.closed
    assert provider.close_count == 1
    assert len(run_metrics.sessions) == 1
    assert run_metrics.sessions[0].status == "failed"
    assert isinstance(run_metrics.sessions[0].error, DriverInvariantError)


def test_input_source_finished_error_returns_failed_not_completed() -> None:
    metrics = InMemorySessionMetricsRecorder()

    result = BatchSessionDriver().run_one_session(
        host=RuntimeHost(_FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))),
        provider=_FakeVideoModelInputProvider(),
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(
                num_windows=1,
                fail_is_finished=RuntimeError("input source failed"),
            ),
            output_sink=_RecordingOutputSink(),
            metrics=metrics,
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "failed"
    assert result.reason == "input source failed"
    assert metrics.errors == ["input source failed"]


def test_step_failure_returns_failed_from_driver() -> None:
    session = _FakeVideoSession(num_steps=1, fail_step=0)
    output = _RecordingOutputSink()

    result = BatchSessionDriver().run_one_session(
        host=RuntimeHost(_FakeVideoRuntime(session=session)),
        provider=_FakeVideoModelInputProvider(),
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=output,
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "failed"
    assert isinstance(result.error, RuntimeError)
    assert result.reason == "step failed"
    assert output.results == []
    assert session.close_count == 1


def test_session_edges_close_result_is_idempotent_and_first_result_wins() -> None:
    output = _RecordingOutputSink(
        artifacts=(OutputArtifact(kind="test/artifact", uri="memory://artifact"),)
    )
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = SessionEdges(
        input_source=_FakeBatchInputSource(num_windows=0),
        output_sink=output,
        metrics=metrics,
        transport=transport,
    )
    first_error = RuntimeError("first")

    first = edges.close_result(
        status="failed",
        reason="first",
        error=first_error,
    )
    second = edges.close_result(status="completed")

    assert second is first
    assert first.status == "failed"
    assert first.reason == "first"
    assert first.error is first_error
    assert tuple(first.artifacts) == (
        OutputArtifact(kind="test/artifact", uri="memory://artifact"),
    )
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed


def test_run_result_rejected_is_the_only_convenience_constructor() -> None:
    constructors = {
        name
        for name, value in RunResult.__dict__.items()
        if isinstance(value, classmethod)
    }

    assert constructors == {"rejected"}
    assert RunResult.rejected(reason="busy").status == "rejected"


def _empty_step_outcome() -> Any:
    from flashdreams.runtime.demo import StepOutcome

    return StepOutcome(output=OutputDecision(), control=ControlDecision())


def _window(index: int) -> UserInputWindow:
    start_s = float(index)
    return UserInputWindow(
        start_s=start_s,
        end_s=start_s + 1.0,
        frame_times=(start_s + 1.0,),
        inputs=UserInputs(),
    )


def _spec() -> DemoSpec:
    return DemoSpec(
        model_id="fake-video-demo",
        input_mode="replay",
        output=NullOutputSpec(),
        config=InferenceConfig(model_id="fake-video-demo"),
    )


def _scenario() -> PreparedScenario:
    return PreparedScenario(initial_inputs=InferenceInput())


def _run_context(
    runtime: _FakeVideoRuntime,
    *,
    admission: SingleSessionAdmissionPolicy | None = None,
    run_metrics: InMemorySessionMetricsRecorder | None = None,
) -> RunContext:
    host = RuntimeHost(runtime)
    return RunContext(
        host=host,
        run_metrics=run_metrics or InMemorySessionMetricsRecorder(),
        admission=admission
        or SingleSessionAdmissionPolicy(health_check=lambda: host.is_healthy),
    )


def _record_output_factory_call(
    calls: list[tuple[DemoSpec, PreparedScenario]],
    spec: DemoSpec,
    scenario: PreparedScenario,
    output: "_RecordingOutputSink",
) -> "_RecordingOutputSink":
    calls.append((spec, scenario))
    return output


class _FakeVideoModelInputProvider:
    def __init__(self, *, fail_initial: Exception | None = None) -> None:
        self.fail_initial = fail_initial
        self.initial_input = InferenceInput(
            global_conditioning={"prompt": "fake video prompt"}
        )
        self.prepared_step_inputs: list[InferenceInput] = []
        self.reset_inputs: list[InferenceInput | None] = []
        self.close_count = 0

    def prepare_initial_input(self) -> InferenceInput:
        if self.fail_initial is not None:
            raise self.fail_initial
        return self.initial_input

    def prepare_step(
        self,
        *,
        request: StepRequest,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        inference_input = InferenceInput(
            step={
                "request_step": request.step_index,
                "window": (user_window.start_s, user_window.end_s),
            }
        )
        self.prepared_step_inputs.append(inference_input)
        return PreparedStep(inference_input=inference_input)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self.reset_inputs.append(inputs)

    def close(self) -> None:
        self.close_count += 1


class _FakeBatchInputSource:
    is_finite = True
    is_deterministic = True

    def __init__(
        self,
        *,
        num_windows: int,
        fail_is_finished: Exception | None = None,
    ) -> None:
        self.windows = [_window(index) for index in range(num_windows)]
        self.fail_is_finished = fail_is_finished
        self.next_window_requests: list[StepRequest] = []
        self.index = 0

    def is_finished(self) -> bool:
        if self.fail_is_finished is not None:
            raise self.fail_is_finished
        return self.index >= len(self.windows)

    def next_window(self, request: StepRequest) -> UserInputWindow:
        self.next_window_requests.append(request)
        window = self.windows[self.index]
        self.index += 1
        return window


class _FakeVideoRuntime:
    def __init__(self, *, session: "_FakeVideoSession") -> None:
        self.session = session
        self.start_session_inputs: list[InferenceInput] = []
        self.close_count = 0

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self.start_session_inputs.append(inputs)
        return self.session

    def close(self) -> None:
        self.close_count += 1


class _FakeVideoSession:
    def __init__(
        self,
        *,
        num_steps: int,
        fail_step: int | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.fail_step = fail_step
        self.next_request_index = 0
        self.step_inputs: list[InferenceInput] = []
        self.close_count = 0

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="fake-video", steady_output_frame_count=1)

    def next_step_request(self) -> StepRequest | None:
        if self.next_request_index >= self.num_steps:
            return None
        request = StepRequest(step_index=self.next_request_index)
        self.next_request_index += 1
        return request

    def step(self, inputs: InferenceInput) -> StepResult:
        step_index = len(self.step_inputs)
        if self.fail_step == step_index:
            raise RuntimeError("step failed")
        self.step_inputs.append(inputs)
        return StepResult(
            step_index=step_index,
            output=f"frame-{step_index}",
            frame_count=1,
            metrics={"model_step_s": 0.01},
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.next_request_index = 0
        self.step_inputs.clear()

    def close(self) -> None:
        self.close_count += 1


class _RecordingRuntimeHost(RuntimeHost):
    def __init__(self, runtime: _FakeVideoRuntime) -> None:
        super().__init__(runtime)
        self.calls: list[str] = []

    def call(self, func: Callable[..., Any], /, *args: object, **kwargs: object) -> Any:
        self.calls.append(getattr(func, "__name__", type(func).__name__))
        return super().call(func, *args, **kwargs)


class _RecordingOutputSink:
    produces_artifacts = True

    def __init__(
        self,
        *,
        artifacts: Sequence[OutputArtifact] = (),
        decision: OutputDecision | None = None,
    ) -> None:
        self.artifacts = tuple(artifacts)
        self.decision = decision or OutputDecision()
        self.opened_with: SessionInfo | None = None
        self.results: list[StepResult] = []
        self.close_count = 0

    def open(self, session_info: SessionInfo) -> None:
        self.opened_with = session_info

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        self.results.append(result)
        return self.decision

    def close(self) -> Sequence[OutputArtifact]:
        self.close_count += 1
        return self.artifacts


class _RecordingTransport:
    def __init__(self) -> None:
        self.close_count = 0

    def is_active(self) -> bool:
        return self.close_count == 0

    def close(self) -> None:
        self.close_count += 1


class _SetupPolicy:
    def __init__(
        self,
        *,
        result_status: Literal["completed", "failed", "skipped"],
    ) -> None:
        self.result_status = result_status

    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status=self.result_status)

    def handle(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed")


class _FakeDemoAdapter:
    model_id = "fake-video-demo"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, provider: _FakeVideoModelInputProvider) -> None:
        self.provider = provider
        self.provider_calls: list[tuple[DemoSpec, PreparedScenario]] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        del config
        raise NotImplementedError("FakeVideoDemo uses an explicit RuntimeHost.")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        del spec
        return _scenario()

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> _FakeVideoModelInputProvider:
        self.provider_calls.append((spec, scenario))
        return self.provider


class _FakeRunMode:
    def __init__(
        self,
        *,
        input_source: _FakeBatchInputSource,
        output_sink: _RecordingOutputSink | None = None,
        output_sink_factory: (
            Callable[[DemoSpec, PreparedScenario], _RecordingOutputSink] | None
        ) = None,
        metrics: InMemorySessionMetricsRecorder | None = None,
        transport: _RecordingTransport | None = None,
        error_policy: _SetupPolicy | None = None,
        validate_error: Exception | None = None,
    ) -> None:
        self.input_source = input_source
        self.output_sink = output_sink or _RecordingOutputSink()
        self.output_sink_factory = output_sink_factory
        self.metrics = metrics or InMemorySessionMetricsRecorder()
        self.transport = transport
        self.error_policy = error_policy
        self.validate_error = validate_error

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: Any,
        provider: Any,
    ) -> None:
        del spec, scenario, adapter, provider
        if self.validate_error is not None:
            raise self.validate_error

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: Any,
        adapter: Any,
    ) -> SessionEdges:
        del context, provider, adapter
        output_sink = (
            self.output_sink_factory(spec, scenario)
            if self.output_sink_factory is not None
            else self.output_sink
        )
        return SessionEdges(
            input_source=self.input_source,
            output_sink=output_sink,
            metrics=self.metrics,
            error_policy=self.error_policy or _SetupPolicy(result_status="failed"),
            transport=self.transport or _RecordingTransport(),
        )

    def select_driver(self) -> BatchSessionDriver:
        return BatchSessionDriver()
