# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import pytest

import flashdreams.runtime.demo.drivers as drivers_module
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
    StepRequirements,
    StepResult,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    BatchSessionDriver,
    ControlDecision,
    DemoSpec,
    DriverInvariantError,
    ErrorAction,
    InMemorySessionMetricsRecorder,
    ModelWarmupPlan,
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
    run_demo_session,
    run_demo_session_async,
)

pytestmark = pytest.mark.ci_cpu


def test_step_pipeline_passes_provider_input_to_session_and_sink() -> None:
    provider = _FakeVideoModelInputProvider()
    session = _FakeVideoSession(num_steps=1)
    output = _RecordingOutputSink()
    output.open(SessionInfo(output_layout="fake-video", steady_output_frame_count=1))
    metrics = InMemorySessionMetricsRecorder()
    request = StepRequirements(step_index=0)
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
        cleanup_tasks=set(),
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


def test_batch_driver_cleanup_failure_marks_host_unhealthy() -> None:
    session = _FakeVideoSession(num_steps=1)
    runtime = _FakeVideoRuntime(session=session)
    host = RuntimeHost(runtime)
    provider = _FakeVideoModelInputProvider(
        fail_close=RuntimeError("provider close failed")
    )
    metrics = InMemorySessionMetricsRecorder()

    try:
        result = BatchSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=SessionEdges(
                input_source=_FakeBatchInputSource(num_windows=1),
                output_sink=_RecordingOutputSink(),
                cleanup_tasks=set(),
                metrics=metrics,
            ),
            pipeline=StepPipeline(),
        )

        assert result.status == "completed"
        assert not host.is_healthy
        assert host.unhealthy_reason == "model-affine cleanup failed"
        assert provider.close_count == 1
        assert metrics.cleanup_errors == ["provider close failed"]
    finally:
        host.close()


def test_batch_driver_slices_windows_from_step_requirements() -> None:
    session = _FakeVideoSession(num_steps=2, input_frame_counts=(3, 2))
    runtime = _FakeVideoRuntime(session=session)
    host = _RecordingRuntimeHost(runtime)
    provider = _FakeVideoModelInputProvider()
    input_source = _SlicingBatchInputSource(fps=2.0, num_windows=2)

    result = BatchSessionDriver().run_one_session(
        host=host,
        provider=provider,
        session_edges=SessionEdges(
            input_source=input_source,
            output_sink=_RecordingOutputSink(),
            cleanup_tasks=set(),
            metrics=InMemorySessionMetricsRecorder(),
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "completed"
    assert [request.step_index for request in input_source.next_window_requests] == [
        0,
        1,
    ]
    assert [
        request.input_frame_count for request in input_source.next_window_requests
    ] == [3, 2]
    assert input_source.windows == [
        _window_with_frame_times(start_s=0.0, frame_times=(0.0, 0.5, 1.0)),
        _window_with_frame_times(start_s=1.5, frame_times=(1.5, 2.0)),
    ]
    assert [dict(inputs.step) for inputs in session.step_inputs] == [
        {"request_step": 0, "window": (0.0, 1.5)},
        {"request_step": 1, "window": (1.5, 2.5)},
    ]


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
            cleanup_tasks=set(),
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


def test_output_sink_open_failure_returns_failed_before_step_loop() -> None:
    session = _FakeVideoSession(num_steps=1)
    metrics = InMemorySessionMetricsRecorder()
    output = _RecordingOutputSink(fail_open=RuntimeError("open failed"))

    result = BatchSessionDriver().run_one_session(
        host=RuntimeHost(_FakeVideoRuntime(session=session)),
        provider=_FakeVideoModelInputProvider(),
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=output,
            cleanup_tasks=set(),
            metrics=metrics,
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "failed"
    assert result.reason == "open failed"
    assert metrics.errors == ["open failed"]
    assert session.step_inputs == []
    assert output.results == []
    assert output.close_count == 1


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
    assert run_metrics.session_errors == ["provider incompatible"]
    snapshot = run_metrics.close()
    assert snapshot.counters["sessions"] == 1
    assert snapshot.counters["session_errors"] == 1
    assert snapshot.session_statuses == ("failed",)


def test_run_demo_session_keeps_failure_when_run_cleanup_metrics_fail() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    run_metrics = _FailingCleanupMetrics()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _FakeVideoModelInputProvider(
        fail_close=RuntimeError("provider close failed")
    )

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
    assert not context.host.is_healthy
    assert context.host.unhealthy_reason == "model-affine cleanup failed"
    assert run_metrics.cleanup_error_attempts == 1
    assert run_metrics.sessions == [result]
    assert runtime.start_session_inputs == []


@pytest.mark.asyncio
async def test_run_demo_session_async_keeps_failure_when_run_cleanup_metrics_fail() -> (
    None
):
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    run_metrics = _FailingCleanupMetrics()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _FakeVideoModelInputProvider(
        fail_close=RuntimeError("provider close failed")
    )

    result = await run_demo_session_async(
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
    assert not context.host.is_healthy
    assert context.host.unhealthy_reason == "model-affine cleanup failed"
    assert run_metrics.cleanup_error_attempts == 1
    assert run_metrics.sessions == [result]
    assert runtime.start_session_inputs == []


@pytest.mark.asyncio
async def test_run_demo_session_async_invariant_cancellation_finalizes_edges() -> None:
    close_entered = threading.Event()
    release_close = threading.Event()
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    run_metrics = InMemorySessionMetricsRecorder()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _BlockingCloseVideoModelInputProvider(
        close_entered=close_entered,
        release_close=release_close,
    )
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    session_metrics = InMemorySessionMetricsRecorder()
    select_error = DriverInvariantError("select driver invariant")
    task = asyncio.create_task(
        run_demo_session_async(
            context=context,
            spec=_spec(),
            scenario=_scenario(),
            adapter=_FakeDemoAdapter(provider=provider),
            run_mode=_FakeRunMode(
                input_source=_FakeBatchInputSource(num_windows=1),
                output_sink=output,
                metrics=session_metrics,
                transport=transport,
                select_error=select_error,
            ),
            pipeline=StepPipeline(),
        )
    )

    try:
        assert await asyncio.to_thread(close_entered.wait, 2.0)
        task.cancel()
        await asyncio.sleep(0)
        release_close.set()
        with pytest.raises(
            DriverInvariantError, match="select driver invariant"
        ) as raised:
            await task
    finally:
        release_close.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        context.host.close()

    assert raised.value is select_error
    assert provider.close_count == 1
    assert output.close_count == 1
    assert transport.close_count == 1
    assert session_metrics.closed
    assert len(run_metrics.sessions) == 1
    recorded = cast(RunResult, run_metrics.sessions[0])
    assert recorded.status == "failed"
    assert recorded.error is select_error
    assert runtime.start_session_inputs == []


@pytest.mark.asyncio
async def test_run_demo_session_async_cancels_before_driver_owns_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    run_metrics = InMemorySessionMetricsRecorder()
    context = _run_context(runtime, run_metrics=run_metrics)
    provider = _FakeVideoModelInputProvider()
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    session_metrics = InMemorySessionMetricsRecorder()
    driver_boundary_reached = asyncio.Event()
    release_driver = asyncio.Event()

    async def fake_run_async_driver(
        *,
        driver: object,
        host: RuntimeHost,
        provider: Any,
        session_edges: SessionEdges,
        pipeline: StepPipeline,
    ) -> RunResult:
        del driver, host, provider, session_edges, pipeline
        driver_boundary_reached.set()
        await release_driver.wait()
        return RunResult(status="completed")

    monkeypatch.setattr(
        drivers_module,
        "_run_async_driver",
        fake_run_async_driver,
    )
    task = asyncio.create_task(
        run_demo_session_async(
            context=context,
            spec=_spec(),
            scenario=_scenario(),
            adapter=_FakeDemoAdapter(provider=provider),
            run_mode=_FakeRunMode(
                input_source=_FakeBatchInputSource(num_windows=1),
                output_sink=output,
                metrics=session_metrics,
                transport=transport,
            ),
            pipeline=StepPipeline(),
        )
    )

    try:
        await asyncio.wait_for(driver_boundary_reached.wait(), timeout=2.0)
        task.cancel()
        result = await task
    finally:
        release_driver.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        context.host.close()

    assert result.status == "cancelled"
    assert result.reason == "cancelled during session assembly"
    assert provider.close_count == 1
    assert output.close_count == 1
    assert transport.close_count == 1
    assert session_metrics.closed
    assert run_metrics.sessions == [result]
    assert runtime.start_session_inputs == []


def test_setup_failure_can_return_skipped_but_not_completed() -> None:
    skipped = BatchSessionDriver().run_one_session(
        host=RuntimeHost(_FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))),
        provider=_FakeVideoModelInputProvider(fail_initial=RuntimeError("skip me")),
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=_RecordingOutputSink(),
            cleanup_tasks=set(),
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
        cleanup_tasks=set(),
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


def test_batch_driver_setup_invariant_reraises_without_error_policy() -> None:
    provider = _FakeVideoModelInputProvider(
        fail_initial=DriverInvariantError("setup invariant")
    )
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = SessionEdges(
        input_source=_FakeBatchInputSource(num_windows=1),
        output_sink=output,
        cleanup_tasks=set(),
        metrics=metrics,
        error_policy=_SetupPolicy(result_status="failed"),
        transport=transport,
    )

    with pytest.raises(DriverInvariantError, match="setup invariant"):
        BatchSessionDriver().run_one_session(
            host=RuntimeHost(_FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))),
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )

    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert metrics.errors == []
    assert provider.close_count == 1


def test_batch_driver_invariant_finalizes_edges_when_host_closed() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    host = RuntimeHost(runtime)
    host.close()
    provider = _FakeVideoModelInputProvider()
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()
    edges = SessionEdges(
        input_source=_FakeBatchInputSource(num_windows=1),
        output_sink=output,
        cleanup_tasks=set(),
        metrics=metrics,
        error_policy=_SetupPolicy(result_status="completed"),
        transport=transport,
    )

    with pytest.raises(DriverInvariantError, match="Setup failures"):
        BatchSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )

    assert edges.is_closed
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert metrics.cleanup_errors == ["runtime host is closed"]
    assert provider.close_count == 0


def test_batch_driver_ordinary_cleanup_finalizes_edges_when_host_closed() -> None:
    session = _FakeVideoSession(num_steps=1)
    runtime = _FakeVideoRuntime(session=session)
    host = _ClosingAfterStepRuntimeHost(runtime)
    provider = _FakeVideoModelInputProvider()
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = InMemorySessionMetricsRecorder()

    result = BatchSessionDriver().run_one_session(
        host=host,
        provider=provider,
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=output,
            cleanup_tasks=set(),
            metrics=metrics,
            transport=transport,
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 1
    assert result.metrics.counters["cleanup_errors"] == 2
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert metrics.cleanup_errors == [
        "runtime host is closed",
        "runtime host is closed",
    ]
    assert session.close_count == 0
    assert provider.close_count == 0


def test_batch_driver_invariant_finalizes_edges_when_cleanup_metrics_fail() -> None:
    runtime = _FakeVideoRuntime(session=_FakeVideoSession(num_steps=1))
    host = RuntimeHost(runtime)
    host.close()
    provider = _FakeVideoModelInputProvider()
    output = _RecordingOutputSink()
    transport = _RecordingTransport()
    metrics = _FailingCleanupMetrics()
    edges = SessionEdges(
        input_source=_FakeBatchInputSource(num_windows=1),
        output_sink=output,
        cleanup_tasks=set(),
        metrics=metrics,
        error_policy=_SetupPolicy(result_status="completed"),
        transport=transport,
    )

    with pytest.raises(DriverInvariantError, match="Setup failures") as raised:
        BatchSessionDriver().run_one_session(
            host=host,
            provider=provider,
            session_edges=edges,
            pipeline=StepPipeline(),
        )

    result = edges.close_result()
    assert result.status == "failed"
    assert result.error is raised.value
    assert output.close_count == 1
    assert transport.close_count == 1
    assert metrics.closed
    assert metrics.cleanup_error_attempts == 1
    assert provider.close_count == 0


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
    recorded = cast(RunResult, run_metrics.sessions[0])
    assert recorded.status == "failed"
    assert isinstance(recorded.error, DriverInvariantError)


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
            cleanup_tasks=set(),
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
            cleanup_tasks=set(),
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
        cleanup_tasks=set(),
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


def test_output_sink_close_failure_records_cleanup_error_without_losing_result() -> (
    None
):
    session = _FakeVideoSession(num_steps=1)
    metrics = InMemorySessionMetricsRecorder()
    output = _RecordingOutputSink(fail_close=RuntimeError("close failed"))

    result = BatchSessionDriver().run_one_session(
        host=RuntimeHost(_FakeVideoRuntime(session=session)),
        provider=_FakeVideoModelInputProvider(),
        session_edges=SessionEdges(
            input_source=_FakeBatchInputSource(num_windows=1),
            output_sink=output,
            cleanup_tasks=set(),
            metrics=metrics,
        ),
        pipeline=StepPipeline(),
    )

    assert result.status == "completed"
    assert result.reason is None
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 1
    assert result.metrics.counters["cleanup_errors"] == 1
    assert result.metrics.errors == ("close failed",)
    assert output.close_count == 1


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


def _window_with_frame_times(
    *,
    start_s: float,
    frame_times: Sequence[float],
) -> UserInputWindow:
    return UserInputWindow(
        start_s=start_s,
        end_s=start_s + len(frame_times) * 0.5,
        frame_times=frame_times,
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
    capabilities = ProviderCapabilities(
        supports_recorded_input=True,
        supports_reset=True,
        deterministic_given_inputs=True,
    )

    def __init__(
        self,
        *,
        fail_initial: Exception | None = None,
        fail_close: Exception | None = None,
    ) -> None:
        self.fail_initial = fail_initial
        self.fail_close = fail_close
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
        request: StepRequirements,
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
        if self.fail_close is not None:
            raise self.fail_close


class _BlockingCloseVideoModelInputProvider(_FakeVideoModelInputProvider):
    def __init__(
        self,
        *,
        close_entered: threading.Event,
        release_close: threading.Event,
    ) -> None:
        super().__init__()
        self.close_entered = close_entered
        self.release_close = release_close

    def close(self) -> None:
        self.close_count += 1
        self.close_entered.set()
        assert self.release_close.wait(timeout=2.0)


class _FakeBatchInputSource:
    is_finite = True
    is_deterministic = True
    user_input_schema = UserInputSchema()

    def __init__(
        self,
        *,
        num_windows: int,
        fail_is_finished: Exception | None = None,
    ) -> None:
        self.windows = [_window(index) for index in range(num_windows)]
        self.fail_is_finished = fail_is_finished
        self.next_window_requests: list[StepRequirements] = []
        self.index = 0

    def is_finished(self) -> bool:
        if self.fail_is_finished is not None:
            raise self.fail_is_finished
        return self.index >= len(self.windows)

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        self.next_window_requests.append(request)
        window = self.windows[self.index]
        self.index += 1
        return window


class _SlicingBatchInputSource:
    is_finite = True
    is_deterministic = True
    user_input_schema = UserInputSchema()

    def __init__(self, *, fps: float, num_windows: int) -> None:
        self.fps = fps
        self.num_windows = num_windows
        self.next_window_requests: list[StepRequirements] = []
        self.windows: list[UserInputWindow] = []
        self.window_index = 0
        self.next_frame_index = 0

    def is_finished(self) -> bool:
        return self.window_index >= self.num_windows

    def next_window(self, request: StepRequirements) -> UserInputWindow:
        self.next_window_requests.append(request)
        start_frame = self.next_frame_index
        self.next_frame_index += request.input_frame_count
        self.window_index += 1
        frame_times = tuple(
            frame_index / self.fps
            for frame_index in range(start_frame, self.next_frame_index)
        )
        window = _window_with_frame_times(
            start_s=start_frame / self.fps,
            frame_times=frame_times,
        )
        self.windows.append(window)
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
        input_frame_counts: Sequence[int] | None = None,
        fail_step: int | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.input_frame_counts = tuple(input_frame_counts or (1,) * num_steps)
        self.fail_step = fail_step
        self.next_request_index = 0
        self.step_inputs: list[InferenceInput] = []
        self.close_count = 0

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="fake-video", steady_output_frame_count=1)

    def next_step_requirements(self) -> StepRequirements | None:
        if self.next_request_index >= self.num_steps:
            return None
        request = StepRequirements(
            step_index=self.next_request_index,
            input_frame_count=self.input_frame_counts[self.next_request_index],
        )
        self.next_request_index += 1
        return request

    def next_step_request(self) -> StepRequest | None:
        raise AssertionError("demo driver should request StepRequirements")

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


class _ClosingAfterStepRuntimeHost(_RecordingRuntimeHost):
    def call(self, func: Callable[..., Any], /, *args: object, **kwargs: object) -> Any:
        result = super().call(func, *args, **kwargs)
        if getattr(func, "__name__", type(func).__name__) == "execute_step":
            self.close()
        return result


class _RecordingOutputSink:
    produces_artifacts = True

    def __init__(
        self,
        *,
        artifacts: Sequence[OutputArtifact] = (),
        decision: OutputDecision | None = None,
        fail_open: Exception | None = None,
        fail_close: Exception | None = None,
    ) -> None:
        self.artifacts = tuple(artifacts)
        self.decision = decision or OutputDecision()
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.opened_with: SessionInfo | None = None
        self.results: list[StepResult] = []
        self.close_count = 0

    def open(self, session_info: SessionInfo) -> None:
        if self.fail_open is not None:
            raise self.fail_open
        self.opened_with = session_info

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        self.results.append(result)
        return self.decision

    def close(self) -> Sequence[OutputArtifact]:
        self.close_count += 1
        if self.fail_close is not None:
            raise self.fail_close
        return self.artifacts


class _RecordingTransport:
    def __init__(self) -> None:
        self.close_count = 0

    def is_active(self) -> bool:
        return self.close_count == 0

    def close(self) -> None:
        self.close_count += 1


class _FailingCleanupMetrics(InMemorySessionMetricsRecorder):
    cleanup_error_attempts: int

    def __init__(self) -> None:
        super().__init__()
        self.cleanup_error_attempts = 0

    def record_cleanup_error(self, exc: Exception) -> None:
        del exc
        self.cleanup_error_attempts += 1
        raise RuntimeError("cleanup metrics failed")


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
    name = "fake"

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
        select_error: Exception | None = None,
    ) -> None:
        self.input_source = input_source
        self.output_sink = output_sink or _RecordingOutputSink()
        self.output_sink_factory = output_sink_factory
        self.metrics = metrics or InMemorySessionMetricsRecorder()
        self.transport = transport
        self.error_policy = error_policy
        self.validate_error = validate_error
        self.select_error = select_error
        self.capabilities = RunModeCapabilities(
            requires_finite_input=True,
            supports_artifacts=True,
        )

    def validate_run(
        self,
        *,
        spec: DemoSpec,
        adapter: Any,
    ) -> None:
        del spec, adapter

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: Any,
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        del spec, adapter
        return RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy
            ),
            model_warmup_plan=model_warmup_plan,
        )

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
        del provider, adapter
        output_sink = (
            self.output_sink_factory(spec, scenario)
            if self.output_sink_factory is not None
            else self.output_sink
        )
        return SessionEdges(
            input_source=self.input_source,
            output_sink=output_sink,
            cleanup_tasks=context.cleanup_tasks,
            metrics=self.metrics,
            error_policy=self.error_policy or _SetupPolicy(result_status="failed"),
            transport=self.transport or _RecordingTransport(),
        )

    def select_driver(self) -> BatchSessionDriver:
        if self.select_error is not None:
            raise self.select_error
        return BatchSessionDriver()
