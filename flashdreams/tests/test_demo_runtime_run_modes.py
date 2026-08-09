# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

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
    StepRequirements,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    AsyncSessionDriver,
    BenchmarkErrorPolicy,
    DemoSpec,
    DriverInvariantError,
    InMemorySessionMetricsRecorder,
    ModelWarmupPlan,
    Mp4ErrorPolicy,
    Mp4OutputSpec,
    NativeWindowErrorPolicy,
    NullErrorPolicy,
    NullOutputSpec,
    OutputDecision,
    PreparedScenario,
    ProviderCapabilities,
    RunContext,
    RunModeCapabilities,
    RunResult,
    RuntimeHost,
    SessionDriver,
    SessionEdges,
    SessionInfo,
    StepPipeline,
    UserInputWindow,
    WebRTCErrorPolicy,
    WebRTCOutputSpec,
    run_demo_session,
    run_demo_session_async,
)

pytestmark = pytest.mark.ci_cpu


def test_fake_mp4_run_mode_calls_session_helper_once(tmp_path: Path) -> None:
    spec = DemoSpec(
        model_id="fake-demo",
        input_mode="replay",
        output=Mp4OutputSpec(path=tmp_path / "fake.mp4", fps=12),
    )
    adapter = _FakeAdapter()
    mode = _FakeRunMode(name="mp4", driver=_ClosingSyncDriver())
    runtime = _UnusedRuntime()
    helper_calls: list[DemoSpec] = []

    result = _run_fake_single_session_mode(
        spec=spec,
        adapter=adapter,
        mode=mode,
        runtime=runtime,
        helper=lambda **kwargs: _record_sync_helper(helper_calls, **kwargs),
    )

    assert result.status == "completed"
    assert helper_calls == [spec]
    assert len(mode.created_edges) == 1
    assert mode.created_edges[0].is_closed
    assert mode.created_edges[0].cleanup_tasks is mode.require_context().cleanup_tasks
    assert adapter.providers[0].close_count == 1
    assert mode.admission.reservations[0].release_count == 1


def test_fake_benchmark_run_mode_calls_helper_once_per_scenario() -> None:
    specs = [
        DemoSpec(
            model_id="fake-demo",
            input_mode="replay",
            output=NullOutputSpec(),
            scenario=f"scenario-{index}",
        )
        for index in range(2)
    ]
    adapter = _FakeAdapter()
    mode = _FakeRunMode(name="benchmark", driver=_ClosingSyncDriver())
    helper_calls: list[DemoSpec] = []

    results = _run_fake_benchmark_mode(
        specs=specs,
        adapter=adapter,
        mode=mode,
        runtime=_UnusedRuntime(),
        helper=lambda **kwargs: _record_sync_helper(helper_calls, **kwargs),
    )

    assert [result.status for result in results] == ["completed", "completed"]
    assert helper_calls == specs
    assert adapter.prepare_scenario_calls == specs
    assert len(adapter.providers) == 2
    assert len({id(provider) for provider in adapter.providers}) == 2
    assert all(provider.close_count == 1 for provider in adapter.providers)
    assert len(mode.created_edges) == 2
    assert len({id(edges) for edges in mode.created_edges}) == 2
    assert all(edges.is_closed for edges in mode.created_edges)
    assert all(
        reservation.release_count == 1 for reservation in mode.admission.reservations
    )


@pytest.mark.asyncio
async def test_fake_webrtc_offer_reserves_before_prepare_or_negotiation() -> None:
    spec = DemoSpec(
        model_id="fake-demo",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(port=8081),
    )
    events: list[str] = []
    blocking_io = _BlockingIOService(events)
    webrtc = _FakeWebRTCService(events)
    adapter = _FakeAdapter(events=events)
    mode = _FakeRunMode(
        name="webrtc",
        driver=_ClosingAsyncDriver(),
        admission=_RecordingAdmission(events=events),
        services={"blocking_io": blocking_io, "webrtc": webrtc},
    )
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_UnusedRuntime()),
        model_warmup_plan=ModelWarmupPlan(),
    )
    helper_calls: list[DemoSpec] = []

    answer = await _handle_fake_webrtc_offer(
        context=context,
        spec=spec,
        adapter=adapter,
        mode=mode,
        helper=lambda **kwargs: _record_async_helper(helper_calls, **kwargs),
        events=events,
    )

    assert answer == "answer"
    assert events.index("admission.reserve") < events.index("blocking_io.run")
    assert events.index("blocking_io.run") < events.index("webrtc.answer")
    assert blocking_io.run_count == 1
    assert helper_calls == [spec]
    assert mode.admission.reservations[0].release_count == 1
    assert adapter.providers[0].close_count == 1
    assert mode.created_edges[0].is_closed


@pytest.mark.asyncio
async def test_async_session_cancellation_shields_pre_edge_provider_cleanup() -> None:
    spec = DemoSpec(
        model_id="fake-demo",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(port=8081),
    )
    provider = _BlockingCloseProvider()
    adapter = _BlockingCloseAdapter(provider=provider)
    mode = _CancelBeforeEdgesRunMode(name="webrtc", driver=_ClosingAsyncDriver())
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_UnusedRuntime()),
        model_warmup_plan=ModelWarmupPlan(),
    )
    scenario = adapter.prepare_scenario(spec)
    task = asyncio.create_task(
        run_demo_session_async(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    )

    close_started = await asyncio.to_thread(provider.close_started.wait, 1.0)
    assert close_started
    task.cancel()
    provider.release_close.set()
    try:
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        provider.release_close.set()
        context.host.close()

    assert result.status == "cancelled"
    assert result.reason == "cancelled during session assembly"
    assert provider.close_count == 1
    assert mode.created_edges == []
    assert mode.admission.reservations[0].release_count == 1
    run_metrics = cast(InMemorySessionMetricsRecorder, context.run_metrics)
    assert run_metrics.sessions == [result]


def test_run_demo_session_rejects_reused_closed_session_edges() -> None:
    spec = DemoSpec(
        model_id="fake-demo",
        input_mode="replay",
        output=NullOutputSpec(),
    )
    adapter = _FakeAdapter()
    mode = _ReusingRunMode(name="mp4", driver=_ClosingSyncDriver())
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(_UnusedRuntime()),
        model_warmup_plan=ModelWarmupPlan(),
    )
    scenario = adapter.prepare_scenario(spec)

    first = run_demo_session(
        context=context,
        spec=spec,
        scenario=scenario,
        adapter=adapter,
        run_mode=mode,
        pipeline=StepPipeline(),
    )
    with pytest.raises(DriverInvariantError, match="must not be reused"):
        run_demo_session(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )

    assert first.status == "completed"
    run_metrics = cast(InMemorySessionMetricsRecorder, context.run_metrics)
    assert len(run_metrics.sessions) == 1
    assert run_metrics.sessions[0] is first
    assert adapter.providers[1].close_count == 1


@pytest.mark.asyncio
async def test_run_context_close_async_drains_cleanup_tasks() -> None:
    metrics = InMemorySessionMetricsRecorder()
    context = RunContext(
        host=RuntimeHost(_UnusedRuntime()),
        run_metrics=metrics,
        admission=_RecordingAdmission(events=[]),
    )
    task = asyncio.create_task(_finished_cleanup_result())
    context.cleanup_tasks.add(task)

    with pytest.raises(RuntimeError, match="Pending session cleanup tasks"):
        context.close()

    summary = await context.close_async()

    assert not context.cleanup_tasks
    assert task.done()
    assert summary.metrics.counters["sessions"] == 0
    assert metrics.closed


def test_error_policy_implementations_keep_setup_failures_terminal() -> None:
    exc = RuntimeError("setup failed")
    policies = (
        Mp4ErrorPolicy(),
        BenchmarkErrorPolicy(),
        WebRTCErrorPolicy(recoverable_exception_types=(RuntimeError,)),
        NativeWindowErrorPolicy(),
        NullErrorPolicy(),
    )

    for policy in policies:
        action = policy.handle_setup_error(exc)
        assert action.result_status == "failed"
        assert action.close_session
        assert not action.drop_chunk


def test_benchmark_error_policy_marks_failed_scenario_continuable() -> None:
    action = BenchmarkErrorPolicy().handle(RuntimeError("scenario failed"))

    assert action.result_status == "failed"
    assert action.close_session
    assert action.continue_next_scenario
    assert not action.drop_chunk


def test_webrtc_error_policy_can_drop_recoverable_step_errors() -> None:
    action = WebRTCErrorPolicy(
        recoverable_exception_types=(RuntimeError,),
    ).handle(RuntimeError("output queue full"))

    assert action.result_status == "failed"
    assert not action.close_session
    assert action.drop_chunk
    assert not action.continue_next_scenario


def _run_fake_single_session_mode(
    *,
    spec: DemoSpec,
    adapter: "_FakeAdapter",
    mode: "_FakeRunMode",
    runtime: "_UnusedRuntime",
    helper: Callable[..., RunResult],
) -> RunResult:
    mode.validate_run(spec=spec, adapter=adapter)
    scenario = adapter.prepare_scenario(spec)
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(runtime),
        model_warmup_plan=ModelWarmupPlan(),
    )
    mode.warmup_context(
        context=context,
        spec=spec,
        scenario=scenario,
        adapter=adapter,
    )
    return helper(
        context=context,
        spec=spec,
        scenario=scenario,
        adapter=adapter,
        run_mode=mode,
        pipeline=StepPipeline(),
    )


def _run_fake_benchmark_mode(
    *,
    specs: Sequence[DemoSpec],
    adapter: "_FakeAdapter",
    mode: "_FakeRunMode",
    runtime: "_UnusedRuntime",
    helper: Callable[..., RunResult],
) -> list[RunResult]:
    mode.validate_run(spec=specs[0], adapter=adapter)
    context = mode.create_run_context(
        spec=specs[0],
        adapter=adapter,
        host=RuntimeHost(runtime),
        model_warmup_plan=ModelWarmupPlan(),
    )
    results: list[RunResult] = []
    for spec in specs:
        scenario = adapter.prepare_scenario(spec)
        results.append(
            helper(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=mode,
                pipeline=StepPipeline(),
            )
        )
    return results


async def _handle_fake_webrtc_offer(
    *,
    context: RunContext,
    spec: DemoSpec,
    adapter: "_FakeAdapter",
    mode: "_FakeRunMode",
    helper: Callable[..., Coroutine[Any, Any, RunResult]],
    events: list[str],
) -> str:
    events.append("handler.start")
    reservation = context.admission.try_reserve()
    if reservation is None:
        return "busy"

    task: asyncio.Task[RunResult] | None = None
    try:
        blocking_io = cast(_BlockingIOService, context.services["blocking_io"])
        scenario = await blocking_io.run(adapter.prepare_scenario, spec)
        task = asyncio.create_task(
            helper(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=mode,
                pipeline=StepPipeline(),
                reservation=reservation,
            )
        )
        webrtc = cast(_FakeWebRTCService, context.services["webrtc"])
        return await webrtc.answer(task)
    except Exception:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        reservation.release()
        raise


def _record_sync_helper(calls: list[DemoSpec], **kwargs: Any) -> RunResult:
    calls.append(kwargs["spec"])
    return run_demo_session(**kwargs)


async def _record_async_helper(calls: list[DemoSpec], **kwargs: Any) -> RunResult:
    calls.append(kwargs["spec"])
    return await run_demo_session_async(**kwargs)


async def _finished_cleanup_result() -> RunResult:
    await asyncio.sleep(0)
    return RunResult.rejected(reason="test cleanup")


class _FakeAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, events: list[str] | None = None) -> None:
        self.events = events
        self.prepare_scenario_calls: list[DemoSpec] = []
        self.providers: list[_FakeProvider] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay", "keyboard-driving")

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null", "mp4", "webrtc")

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return _UnusedRuntime()

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if self.events is not None:
            self.events.append("adapter.prepare_scenario")
        self.prepare_scenario_calls.append(spec)
        return PreparedScenario(initial_inputs=InferenceInput())

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> "_FakeProvider":
        del spec, scenario
        provider = _FakeProvider()
        self.providers.append(provider)
        return provider


class _BlockingCloseAdapter(_FakeAdapter):
    def __init__(self, *, provider: "_BlockingCloseProvider") -> None:
        super().__init__()
        self.provider = provider

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> "_BlockingCloseProvider":
        del spec, scenario
        self.providers.append(self.provider)
        return self.provider


class _FakeProvider:
    capabilities = ProviderCapabilities(
        supports_recorded_input=True,
        supports_reset=True,
        deterministic_given_inputs=True,
    )

    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _BlockingCloseProvider(_FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def close(self) -> None:
        self.close_started.set()
        if not self.release_close.wait(timeout=1.0):
            raise RuntimeError("timed out waiting to release provider close")
        super().close()


class _FakeRunMode:
    def __init__(
        self,
        *,
        name: str,
        driver: SessionDriver | AsyncSessionDriver,
        admission: "_RecordingAdmission | None" = None,
        services: Mapping[str, object] | None = None,
    ) -> None:
        self.name = name
        self.driver = driver
        self.created_edges: list[SessionEdges] = []
        self.validate_run_count = 0
        self.warmup_count = 0
        self.capabilities = RunModeCapabilities(requires_finite_input=True)
        self.admission = admission or _RecordingAdmission(events=[])
        self.services = services or {}
        self.context: RunContext | None = None

    def require_context(self) -> RunContext:
        if self.context is None:
            raise AssertionError("Run context was not created.")
        return self.context

    def validate_run(self, *, spec: DemoSpec, adapter: Any) -> None:
        del spec, adapter
        self.validate_run_count += 1

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
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        del spec, adapter
        self.context = RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=self.admission,
            model_warmup_plan=model_warmup_plan,
            services=self.services,
        )
        return self.context

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
        edges = SessionEdges(
            input_source=_FinishedInputSource(),
            output_sink=_RecordingOutputSink(),
            cleanup_tasks=context.cleanup_tasks,
            metrics=InMemorySessionMetricsRecorder(),
            transport=_RecordingTransport(),
        )
        self.created_edges.append(edges)
        return edges

    def select_driver(self) -> SessionDriver | AsyncSessionDriver:
        return self.driver

    def warmup_context(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: Any,
        adapter: Any,
    ) -> None:
        del context, spec, scenario, adapter
        self.warmup_count += 1


class _ReusingRunMode(_FakeRunMode):
    def __init__(
        self,
        *,
        name: str,
        driver: SessionDriver | AsyncSessionDriver,
    ) -> None:
        super().__init__(name=name, driver=driver)
        self._edges: SessionEdges | None = None

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: Any,
        provider: Any,
        adapter: Any,
    ) -> SessionEdges:
        if self._edges is None:
            self._edges = super().create_session_edges(
                context=context,
                spec=spec,
                scenario=scenario,
                provider=provider,
                adapter=adapter,
            )
        return self._edges


class _CancelBeforeEdgesRunMode(_FakeRunMode):
    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: Any,
        adapter: Any,
        provider: Any,
    ) -> None:
        del spec, scenario, adapter, provider
        raise asyncio.CancelledError


class _ClosingSyncDriver:
    def run_one_session(
        self,
        *,
        host: RuntimeHost,
        provider: Any,
        session_edges: SessionEdges,
        pipeline: StepPipeline,
    ) -> RunResult:
        del host, pipeline
        provider.close()
        return session_edges.close_result(status="completed")


class _ClosingAsyncDriver:
    async def run_one_session(
        self,
        *,
        host: RuntimeHost,
        provider: Any,
        session_edges: SessionEdges,
        pipeline: StepPipeline,
    ) -> RunResult:
        del pipeline
        await host.call_async(provider.close)
        return session_edges.close_result(status="completed")


class _RecordingAdmission:
    def __init__(self, *, events: list[str]) -> None:
        self.events = events
        self.reservations: list[_RecordingReservation] = []

    def try_reserve(self) -> "_RecordingReservation":
        self.events.append("admission.reserve")
        reservation = _RecordingReservation()
        self.reservations.append(reservation)
        return reservation


class _RecordingReservation:
    def __init__(self) -> None:
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            return
        self.release_count += 1


class _FinishedInputSource:
    is_finite = True
    is_deterministic = True
    user_input_schema = UserInputSchema()

    def is_finished(self) -> bool:
        return True

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        del request
        return UserInputWindow(start_s=0.0, end_s=0.0)


class _RecordingOutputSink:
    produces_artifacts = False

    def __init__(self) -> None:
        self.close_count = 0

    def open(self, session_info: SessionInfo) -> None:
        del session_info

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: Any) -> OutputDecision:
        del result
        return OutputDecision()

    def close(self) -> Sequence[Any]:
        self.close_count += 1
        return ()


class _RecordingTransport:
    def close(self) -> None:
        return

    def is_active(self) -> bool:
        return True


class _BlockingIOService:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.run_count = 0

    async def run(self, func: Callable[..., Any], *args: object) -> Any:
        self.run_count += 1
        self.events.append("blocking_io.run")
        await asyncio.sleep(0)
        return func(*args)


class _FakeWebRTCService:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def answer(self, task: asyncio.Task[RunResult]) -> str:
        self.events.append("webrtc.answer")
        result = await task
        assert result.status == "completed"
        return "answer"


class _UnusedRuntime:
    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        raise AssertionError("The fake Phase 4 drivers do not start sessions.")

    def close(self) -> None:
        return
