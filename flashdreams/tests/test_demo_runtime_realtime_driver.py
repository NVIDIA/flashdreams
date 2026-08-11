# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import pytest

from flashdreams.runtime import (
    InferenceInput,
    InferenceRuntime,
    StepRequest,
    StepRequirements,
    StepResult,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    ActivationResult,
    DriverInvariantError,
    ErrorAction,
    InMemorySessionMetricsRecorder,
    OutputDecision,
    PreparedStep,
    ProviderCapabilities,
    RealtimeSessionDriver,
    RealtimeWindowResult,
    RunContext,
    RunResult,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
    StepPipeline,
    UserInputWindow,
    shielded_session_cleanup,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.asyncio
async def test_realtime_driver_non_activation_returns_not_activated() -> None:
    runtime = _FakeRealtimeRuntime(session=_FakeRealtimeSession(num_steps=1))
    host = RuntimeHost(runtime)
    provider = _FakeRealtimeProvider()
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(
        input_source=_RealtimeInputSource(),
        output=output,
        transport=transport,
        metrics=metrics,
        activation=_ActivationPolicy(ActivationResult(activated=False, reason="idle")),
    )

    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()

    assert result.status == "not_activated"
    assert result.reason == "idle"
    assert runtime.start_session_inputs == []
    assert provider.prepare_initial_count == 0
    assert provider.close_count == 1
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed


@pytest.mark.asyncio
async def test_realtime_driver_transport_close_before_first_step_is_not_activated() -> (
    None
):
    session = _FakeRealtimeSession(num_steps=1)
    runtime = _FakeRealtimeRuntime(session=session)
    host = RuntimeHost(runtime)
    transport = _RecordingTransport()
    edges = _edges(
        input_source=_RealtimeInputSource(transport_to_close=transport),
        transport=transport,
    )

    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=_FakeRealtimeProvider(),
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()

    assert result.status == "not_activated"
    assert result.reason == "transport closed before first step"
    assert session.step_inputs == []


@pytest.mark.asyncio
async def test_repeated_cancellation_during_cleanup_still_closes_edges() -> None:
    entered_window = asyncio.Event()
    session = _FakeRealtimeSession(num_steps=1)
    runtime = _FakeRealtimeRuntime(session=session)
    host = RuntimeHost(runtime)
    provider = _FakeRealtimeProvider(close_delay_s=0.05)
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(
        input_source=_RealtimeInputSource(
            entered=entered_window,
            wait_forever=True,
        ),
        output=output,
        transport=transport,
        metrics=metrics,
    )
    task = asyncio.create_task(
        RealtimeSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    )
    await entered_window.wait()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    result = await task
    host.close()

    assert result.status == "cancelled"
    assert result.reason == "cancelled"
    assert session.close_count == 1
    assert provider.close_count == 1
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert not edges.cleanup_tasks


@pytest.mark.asyncio
async def test_cancelled_realtime_driver_inside_timeout_returns_result() -> None:
    timeout_context = getattr(asyncio, "timeout", None)
    if timeout_context is None:
        pytest.skip("asyncio.timeout is unavailable on this Python version.")
    entered_window = asyncio.Event()
    runtime = _FakeRealtimeRuntime(session=_FakeRealtimeSession(num_steps=1))
    host = RuntimeHost(runtime)
    edges = _edges(
        input_source=_RealtimeInputSource(
            entered=entered_window,
            wait_forever=True,
        )
    )

    async def timeout_after_window_entry(timeout: Any) -> None:
        await entered_window.wait()
        timeout.reschedule(asyncio.get_running_loop().time())

    try:
        async with timeout_context(None) as timeout:
            timeout_task = asyncio.create_task(timeout_after_window_entry(timeout))
            result = await RealtimeSessionDriver().run_one_session(
                host=host,
                provider=_FakeRealtimeProvider(),
                session_edges=edges,
                pipeline=StepPipeline(),
            )
            timeout_task.cancel()
            await asyncio.gather(timeout_task, return_exceptions=True)
    finally:
        host.close()

    assert entered_window.is_set()
    assert result.status == "cancelled"


@pytest.mark.asyncio
async def test_realtime_driver_invariant_finalizes_edges_before_reraising() -> None:
    runtime = _FakeRealtimeRuntime(session=_FakeRealtimeSession(num_steps=1))
    host = RuntimeHost(runtime)
    provider = _FakeRealtimeProvider(fail_initial=RuntimeError("bad setup policy"))
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(
        output=output,
        transport=transport,
        metrics=metrics,
        error_policy=_SetupPolicy(result_status="completed"),
    )

    try:
        with pytest.raises(DriverInvariantError, match="Setup failures") as raised:
            await RealtimeSessionDriver().run_one_session(
                host=host,
                provider=provider,
                session_edges=edges,
                pipeline=StepPipeline(),
            )
    finally:
        host.close()

    result = edges.close_result()
    assert result.status == "failed"
    assert result.error is raised.value
    assert provider.close_count == 1
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed


@pytest.mark.asyncio
async def test_realtime_driver_setup_invariant_reraises_without_error_policy() -> None:
    runtime = _FakeRealtimeRuntime(session=_FakeRealtimeSession(num_steps=1))
    host = RuntimeHost(runtime)
    provider = _FakeRealtimeProvider(
        fail_initial=DriverInvariantError("setup invariant")
    )
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(
        output=output,
        transport=transport,
        metrics=metrics,
        error_policy=_SetupPolicy(result_status="failed"),
    )

    try:
        with pytest.raises(DriverInvariantError, match="setup invariant"):
            await RealtimeSessionDriver().run_one_session(
                host=host,
                provider=provider,
                session_edges=edges,
                pipeline=StepPipeline(),
            )
    finally:
        host.close()

    assert provider.close_count == 1
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert metrics.errors == []


@pytest.mark.asyncio
async def test_realtime_step_invariant_reraises_without_error_policy() -> None:
    runtime = _FakeRealtimeRuntime(session=_FakeRealtimeSession(num_steps=1))
    host = RuntimeHost(runtime)
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(
        output=output,
        transport=transport,
        metrics=metrics,
        error_policy=_DropOutputErrorPolicy(),
    )

    try:
        with pytest.raises(DriverInvariantError, match="step invariant"):
            await RealtimeSessionDriver().run_one_session(
                host=host,
                provider=_FakeRealtimeProvider(),
                session_edges=edges,
                pipeline=_InvariantPipeline(),
            )
    finally:
        host.close()

    assert metrics.errors == []
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed


@pytest.mark.asyncio
async def test_run_context_close_async_drains_registered_cleanup_task() -> None:
    runtime = _FakeRealtimeRuntime(session=_FakeRealtimeSession(num_steps=1))
    host = RuntimeHost(runtime)
    context = RunContext(
        host=host,
        run_metrics=InMemorySessionMetricsRecorder(),
        admission=SingleSessionAdmissionPolicy(),
    )
    edges = _edges(cleanup_tasks=context.cleanup_tasks)
    cleanup_task = asyncio.create_task(
        shielded_session_cleanup(
            host=host,
            session=runtime.session,
            provider=_FakeRealtimeProvider(close_delay_s=0.02),
            session_edges=edges,
            status="cancelled",
            reason="test",
            error=None,
        )
    )
    await asyncio.sleep(0)

    summary = await context.close_async()
    cleanup_result = await cleanup_task

    assert cleanup_task.done()
    assert cleanup_result.status == "cancelled"
    assert not context.cleanup_tasks
    assert edges.is_closed
    assert summary.metrics.counters["sessions"] == 0


@pytest.mark.asyncio
async def test_shielded_cleanup_never_raises_and_returns_result_on_close_errors() -> (
    None
):
    runtime = _FakeRealtimeRuntime(
        session=_FakeRealtimeSession(num_steps=1, fail_close=RuntimeError("session"))
    )
    host = RuntimeHost(runtime)
    provider = _FakeRealtimeProvider(fail_close=RuntimeError("provider"))
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(metrics=metrics)

    try:
        result = await shielded_session_cleanup(
            host=host,
            session=runtime.session,
            provider=provider,
            session_edges=edges,
            status="failed",
            reason="test failure",
            error=RuntimeError("original"),
        )
        assert not host.is_healthy
        assert host.unhealthy_reason == "model-affine cleanup failed"
    finally:
        host.close()

    assert result.status == "failed"
    assert result.reason == "test failure"
    assert metrics.closed
    assert metrics.cleanup_errors == ["session", "provider"]


@pytest.mark.asyncio
async def test_shielded_cleanup_timeout_bounds_shutdown() -> None:
    host = _NeverReturningHost()
    session = _FakeRealtimeSession(num_steps=1)
    provider = _FakeRealtimeProvider()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(metrics=metrics)

    result = await shielded_session_cleanup(
        host=cast(RuntimeHost, host),
        session=session,
        provider=provider,
        session_edges=edges,
        status="cancelled",
        reason="timeout test",
        error=None,
        timeout_s=0.001,
    )

    assert result.status == "cancelled"
    assert host.unhealthy_reason == "model-affine cleanup timed out"
    assert host.close_targets == [session, provider]
    assert provider.close_count == 0
    assert metrics.cleanup_errors == []
    assert len(metrics.orphaned_cleanup_errors) == 1
    assert metrics.closed


@pytest.mark.asyncio
async def test_shielded_cleanup_dispatch_failure_marks_host_unhealthy() -> None:
    host = _RejectingHost(RuntimeError("worker rejected cleanup"))
    session = _FakeRealtimeSession(num_steps=1)
    provider = _FakeRealtimeProvider()
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(metrics=metrics)

    result = await shielded_session_cleanup(
        host=cast(RuntimeHost, host),
        session=session,
        provider=provider,
        session_edges=edges,
        status="cancelled",
        reason="dispatch failure",
        error=None,
        timeout_s=0.001,
    )

    assert result.status == "cancelled"
    assert host.unhealthy_reason == "model-affine cleanup failed"
    assert host.cleanup_dispatch_count == 1
    assert session.close_count == 0
    assert provider.close_count == 0
    assert metrics.cleanup_errors == ["worker rejected cleanup"]
    assert metrics.orphaned_cleanup_errors == []
    assert metrics.closed


@pytest.mark.asyncio
async def test_realtime_driver_applies_backpressure_through_clock() -> None:
    session = _FakeRealtimeSession(num_steps=2)
    runtime = _FakeRealtimeRuntime(session=session)
    host = RuntimeHost(runtime)
    clock = _RecordingRealtimeClock()
    metrics = InMemorySessionMetricsRecorder()
    output = _RecordingOutputSink(
        decisions=(
            OutputDecision(backpressure_s=0.25),
            OutputDecision(should_stop=True),
        )
    )
    edges = _edges(clock=clock, output=output, metrics=metrics)

    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=_FakeRealtimeProvider(),
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()

    assert result.status == "completed"
    assert clock.backpressure == [0.25]
    assert metrics.catch_up_count == 2
    assert len(output.results) == 2


@pytest.mark.asyncio
async def test_realtime_driver_calls_step_pipeline_on_runtime_host() -> None:
    session = _FakeRealtimeSession(num_steps=1)
    runtime = _FakeRealtimeRuntime(session=session)
    host = _RecordingRuntimeHost(runtime)
    edges = _edges(
        output=_RecordingOutputSink(decisions=(OutputDecision(should_stop=True),))
    )

    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=_FakeRealtimeProvider(),
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()

    assert result.status == "completed"
    assert "execute_step" in host.async_calls
    assert "prepare_step" not in host.async_calls
    assert "step" not in host.async_calls


@pytest.mark.asyncio
async def test_slow_fake_model_step_does_not_block_event_loop() -> None:
    session = _FakeRealtimeSession(num_steps=1, step_delay_s=0.05)
    runtime = _FakeRealtimeRuntime(session=session)
    host = RuntimeHost(runtime)
    edges = _edges(
        output=_RecordingOutputSink(decisions=(OutputDecision(should_stop=True),))
    )
    ticks = 0
    finished = False

    async def heartbeat() -> None:
        nonlocal ticks
        while not finished:
            ticks += 1
            await asyncio.sleep(0.005)

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=_FakeRealtimeProvider(),
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        finished = True
        await heartbeat_task
        host.close()

    assert result.status == "completed"
    assert ticks >= 2


@pytest.mark.asyncio
async def test_realtime_driver_fatal_model_error_returns_failed() -> None:
    session = _FakeRealtimeSession(num_steps=1, fail_step=0)
    runtime = _FakeRealtimeRuntime(session=session)
    host = RuntimeHost(runtime)
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(metrics=metrics)

    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=_FakeRealtimeProvider(),
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()

    assert result.status == "failed"
    assert result.reason == "step failed"
    assert isinstance(result.error, RuntimeError)
    assert session.close_count == 1
    assert metrics.errors == ["step failed"]


@pytest.mark.asyncio
async def test_realtime_driver_can_drop_recoverable_output_error() -> None:
    session = _FakeRealtimeSession(num_steps=2)
    runtime = _FakeRealtimeRuntime(session=session)
    host = RuntimeHost(runtime)
    output = _RecordingOutputSink(
        fail_first_write=RuntimeError("output queue full"),
        decisions=(OutputDecision(should_stop=True),),
    )
    metrics = InMemorySessionMetricsRecorder()
    edges = _edges(
        output=output,
        metrics=metrics,
        error_policy=_DropOutputErrorPolicy(),
    )

    try:
        result = await RealtimeSessionDriver().run_one_session(
            host=host,
            provider=_FakeRealtimeProvider(),
            session_edges=edges,
            pipeline=StepPipeline(),
        )
    finally:
        host.close()

    assert result.status == "completed"
    assert metrics.errors == ["output queue full"]
    assert [step.step_index for step in output.results] == [1]
    assert len(session.step_inputs) == 2


def _edges(
    *,
    input_source: "_RealtimeInputSource | None" = None,
    output: "_RecordingOutputSink | None" = None,
    transport: "_RecordingTransport | None" = None,
    metrics: InMemorySessionMetricsRecorder | None = None,
    activation: "_ActivationPolicy | None" = None,
    clock: "_RecordingRealtimeClock | None" = None,
    cleanup_tasks: set[asyncio.Task[RunResult]] | None = None,
    error_policy: Any | None = None,
) -> SessionEdges:
    return SessionEdges(
        input_source=input_source or _RealtimeInputSource(),
        output_sink=output
        or _RecordingOutputSink(decisions=(OutputDecision(should_stop=True),)),
        cleanup_tasks=cleanup_tasks or set(),
        metrics=metrics or InMemorySessionMetricsRecorder(),
        error_policy=error_policy or _DefaultTestErrorPolicy(),
        transport=transport or _RecordingTransport(),
        clock=clock or _RecordingRealtimeClock(),
        activation=activation or _ActivationPolicy(ActivationResult(activated=True)),
    )


def _window(index: int) -> UserInputWindow:
    start_s = float(index)
    return UserInputWindow(
        start_s=start_s,
        end_s=start_s + 1.0,
        frame_times=(start_s + 1.0,),
    )


class _ActivationPolicy:
    timeout_s: float | None = None

    def __init__(self, result: ActivationResult) -> None:
        self.result = result
        self.calls = 0

    async def wait_until_active(self, clock: Any) -> ActivationResult:
        del clock
        self.calls += 1
        await asyncio.sleep(0)
        return self.result


class _RecordingRealtimeClock:
    is_realtime = True
    is_deterministic = False

    def __init__(self) -> None:
        self.backpressure: list[float] = []
        self.anchors: list[float] = []

    def now(self) -> float:
        return 0.0

    def anchor(self, wall_time_s: float) -> None:
        self.anchors.append(wall_time_s)

    async def wait_until_window_end(self, end_s: float) -> None:
        del end_s

    async def apply_backpressure(self, requested_s: float) -> None:
        self.backpressure.append(requested_s)
        await asyncio.sleep(0)

    def catch_up(self, **kwargs: Any) -> object:
        del kwargs
        return object()


class _RealtimeInputSource:
    is_finite = False
    is_deterministic = False
    user_input_schema = UserInputSchema()

    def __init__(
        self,
        *,
        entered: asyncio.Event | None = None,
        wait_forever: bool = False,
        transport_to_close: "_RecordingTransport | None" = None,
    ) -> None:
        self.entered = entered
        self.wait_forever = wait_forever
        self.transport_to_close = transport_to_close
        self.requests: list[StepRequirements] = []

    def is_finished(self) -> bool:
        return False

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: Any,
    ) -> RealtimeWindowResult:
        del clock
        self.requests.append(request)
        if self.entered is not None:
            self.entered.set()
        if self.wait_forever:
            await asyncio.Event().wait()
        if self.transport_to_close is not None:
            self.transport_to_close.close()
        return RealtimeWindowResult(window=_window(request.step_index))


class _FakeRealtimeProvider:
    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        supports_reset=True,
        deterministic_given_inputs=False,
    )

    def __init__(
        self,
        *,
        fail_initial: Exception | None = None,
        fail_close: Exception | None = None,
        close_delay_s: float = 0.0,
    ) -> None:
        self.fail_initial = fail_initial
        self.fail_close = fail_close
        self.close_delay_s = close_delay_s
        self.prepare_initial_count = 0
        self.close_count = 0
        self.reset_inputs: list[InferenceInput | None] = []

    def prepare_initial_input(self) -> InferenceInput:
        if self.fail_initial is not None:
            raise self.fail_initial
        self.prepare_initial_count += 1
        return InferenceInput(global_conditioning={"prompt": "realtime"})

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        return PreparedStep(
            inference_input=InferenceInput(
                step={
                    "request_step": request.step_index,
                    "window": (user_window.start_s, user_window.end_s),
                }
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self.reset_inputs.append(inputs)

    def close(self) -> None:
        if self.close_delay_s:
            time.sleep(self.close_delay_s)
        self.close_count += 1
        if self.fail_close is not None:
            raise self.fail_close


class _FakeRealtimeRuntime:
    def __init__(self, *, session: "_FakeRealtimeSession") -> None:
        self.session = session
        self.start_session_inputs: list[InferenceInput] = []
        self.close_count = 0

    def start_session(self, inputs: InferenceInput) -> "_FakeRealtimeSession":
        self.start_session_inputs.append(inputs)
        return self.session

    def close(self) -> None:
        self.close_count += 1


class _FakeRealtimeSession:
    def __init__(
        self,
        *,
        num_steps: int,
        fail_step: int | None = None,
        fail_close: Exception | None = None,
        step_delay_s: float = 0.0,
    ) -> None:
        self.num_steps = num_steps
        self.fail_step = fail_step
        self.fail_close = fail_close
        self.step_delay_s = step_delay_s
        self.next_request_index = 0
        self.step_inputs: list[InferenceInput] = []
        self.close_count = 0

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="fake-realtime", steady_output_frame_count=1)

    def next_step_requirements(self) -> StepRequirements | None:
        if self.next_request_index >= self.num_steps:
            return None
        request = StepRequirements(step_index=self.next_request_index)
        self.next_request_index += 1
        return request

    def next_step_request(self) -> StepRequest | None:
        raise AssertionError("demo driver should request StepRequirements")

    def step(self, inputs: InferenceInput) -> StepResult:
        step_index = len(self.step_inputs)
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        if self.fail_step == step_index:
            raise RuntimeError("step failed")
        self.step_inputs.append(inputs)
        return StepResult(
            step_index=step_index,
            output=f"frame-{step_index}",
            frame_count=1,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.next_request_index = 0
        self.step_inputs.clear()

    def close(self) -> None:
        self.close_count += 1
        if self.fail_close is not None:
            raise self.fail_close


class _RecordingRuntimeHost(RuntimeHost):
    def __init__(self, runtime: InferenceRuntime) -> None:
        super().__init__(runtime)
        self.async_calls: list[str] = []

    async def call_async(
        self,
        func: Callable[..., Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> Any:
        self.async_calls.append(getattr(func, "__name__", type(func).__name__))
        return await super().call_async(func, *args, **kwargs)


class _InvariantPipeline(StepPipeline):
    def execute_step(self, **kwargs: object) -> Any:
        del kwargs
        raise DriverInvariantError("step invariant")


class _RecordingOutputSink:
    produces_artifacts = False

    def __init__(
        self,
        *,
        decisions: Sequence[OutputDecision] = (),
        fail_first_write: Exception | None = None,
    ) -> None:
        self.decisions = list(decisions)
        self.fail_first_write = fail_first_write
        self.opened_with: SessionInfo | None = None
        self.generations: list[int] = []
        self.results: list[StepResult] = []
        self.close_count = 0
        self.write_attempts = 0

    def open(self, session_info: SessionInfo) -> None:
        self.opened_with = session_info

    def begin_generation(self, generation: int) -> None:
        self.generations.append(generation)

    def write(self, result: StepResult) -> OutputDecision:
        self.write_attempts += 1
        if self.fail_first_write is not None:
            exc = self.fail_first_write
            self.fail_first_write = None
            raise exc
        self.results.append(result)
        if self.decisions:
            return self.decisions.pop(0)
        return OutputDecision()

    def close(self) -> Sequence[Any]:
        self.close_count += 1
        return ()


class _RecordingTransport:
    def __init__(self) -> None:
        self.active = True
        self.close_count = 0

    def is_active(self) -> bool:
        return self.active

    def close(self) -> None:
        self.active = False
        self.close_count += 1


class _DefaultTestErrorPolicy:
    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed")

    def handle(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed")


class _SetupPolicy(_DefaultTestErrorPolicy):
    def __init__(
        self,
        *,
        result_status: Literal["completed", "failed", "skipped"],
    ) -> None:
        self.result_status = result_status

    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status=self.result_status)


class _DropOutputErrorPolicy(_DefaultTestErrorPolicy):
    def handle(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(
            close_session=False,
            drop_chunk=True,
            result_status="failed",
        )


class _NeverReturningHost:
    def __init__(self) -> None:
        self.unhealthy_reason: str | None = None
        self.close_targets: list[Any] = []

    async def call_async(
        self,
        func: Callable[..., Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> Any:
        del func, kwargs
        for arg in args:
            if callable(arg):
                close = cast(Callable[[], None], arg)
                self.close_targets.append(getattr(close, "__self__", close))
        await asyncio.Event().wait()

    def mark_unhealthy(
        self,
        reason: str = "marked unhealthy",
        error: Exception | None = None,
    ) -> None:
        del error
        self.unhealthy_reason = reason


class _RejectingHost:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.cleanup_dispatch_count = 0
        self.unhealthy_reason: str | None = None

    async def call_async(
        self,
        func: Callable[..., Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> Any:
        del func, args, kwargs
        self.cleanup_dispatch_count += 1
        raise self.exc

    def mark_unhealthy(
        self,
        reason: str = "marked unhealthy",
        error: Exception | None = None,
    ) -> None:
        del error
        self.unhealthy_reason = reason
