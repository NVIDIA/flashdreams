# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session drivers and helpers for demo runtime vertical slices."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast

from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import (
    StepRequest,
    StepRequirements,
    step_requirements_from_request,
)

from .host import RuntimeHost
from .outputs import SessionInfo
from .pipeline import StepPipeline
from .run_modes import (
    DriverStatus,
    RunContext,
    RunMode,
    RunResult,
    SessionEdges,
    SessionReservation,
)
from .session_inputs import BatchInputSource, ModelInputProvider
from .spec import DemoAdapter, DemoSpec, PreparedScenario
from .timing import ActivationPolicy, RealtimeClock
from .validation import resolve_run_capabilities, validate_resolved_run

CLEANUP_TIMEOUT_S = 30.0
_MODEL_CLEANUP_FAILED_REASON = "model-affine cleanup failed"
_MODEL_CLEANUP_TIMED_OUT_REASON = "model-affine cleanup timed out"


class DriverInvariantError(RuntimeError):
    """A driver invariant was violated; this is a driver bug, not a run result."""


class BatchSessionDriver:
    """Minimal finite-session driver for Phase 2 fake-model coverage."""

    def run_one_session(
        self,
        *,
        host: RuntimeHost,
        provider: ModelInputProvider,
        session_edges: SessionEdges,
        pipeline: StepPipeline,
    ) -> RunResult:
        session: InferenceSession | None = None
        final_status: DriverStatus = "completed"
        final_reason: str | None = None
        final_error: Exception | None = None
        invariant_closed = False
        setup_ok = False
        try:
            try:
                initial_input = host.call(provider.prepare_initial_input)
                session = host.call(host.start_session, initial_input)
                session_info = host.call(_session_info, session)
                session_edges.output_sink.open(session_info)
                setup_ok = True
            except DriverInvariantError:
                raise
            except Exception as exc:
                action = session_edges.error_policy.handle_setup_error(exc)
                if action.drop_chunk or action.result_status == "completed":
                    raise DriverInvariantError(
                        "Setup failures must resolve to failed or skipped."
                    ) from exc
                session_edges.metrics.record_error(exc, action)
                final_status = action.result_status
                final_reason = str(exc)
                final_error = exc if action.result_status == "failed" else None

            input_source = cast(BatchInputSource, session_edges.input_source)
            while setup_ok:
                if session is None:
                    raise DriverInvariantError("setup_ok was set without a session.")
                try:
                    if session_edges.input_source.is_finished():
                        break
                    request = _next_step_requirements(host=host, session=session)
                    if request is None:
                        break
                    user_window = input_source.next_window(request)
                    outcome = host.call(
                        pipeline.execute_step,
                        request=request,
                        user_window=user_window,
                        provider=provider,
                        session=session,
                        output=session_edges.output_sink,
                        metrics=session_edges.metrics,
                    )
                    if outcome.control.reset:
                        host.call(session.reset, outcome.control.reset_input)
                        if not outcome.control.provider_already_reset:
                            host.call(provider.reset, outcome.control.reset_input)
                        continue
                    if outcome.control.close_session:
                        break
                    if outcome.output.should_stop:
                        break
                except DriverInvariantError:
                    raise
                except Exception as exc:
                    action = session_edges.error_policy.handle(exc)
                    session_edges.metrics.record_error(exc, action)
                    if action.drop_chunk:
                        continue
                    final_status = action.result_status
                    final_reason = str(exc)
                    final_error = exc if action.result_status == "failed" else None
                    break
        except DriverInvariantError as exc:
            if session is not None:
                _close_on_host_best_effort(
                    host=host,
                    close=session.close,
                    session_edges=session_edges,
                )
            _close_on_host_best_effort(
                host=host,
                close=provider.close,
                session_edges=session_edges,
            )
            session_edges.close_result(
                status="failed",
                reason=str(exc),
                error=exc,
            )
            invariant_closed = True
            raise
        except Exception as exc:
            final_status = "failed"
            final_reason = str(exc)
            final_error = exc
        finally:
            if not invariant_closed:
                if session is not None:
                    _close_on_host_best_effort(
                        host=host,
                        close=session.close,
                        session_edges=session_edges,
                    )
                _close_on_host_best_effort(
                    host=host,
                    close=provider.close,
                    session_edges=session_edges,
                )

        return session_edges.close_result(
            status=final_status,
            reason=final_reason,
            error=final_error,
        )


class RealtimeSessionDriver:
    """Async realtime session driver built on shared Phase 5 primitives."""

    cleanup_timeout_s: float

    def __init__(self, *, cleanup_timeout_s: float = CLEANUP_TIMEOUT_S) -> None:
        if cleanup_timeout_s <= 0:
            raise ValueError("cleanup_timeout_s must be > 0.")
        self.cleanup_timeout_s = float(cleanup_timeout_s)

    async def run_one_session(
        self,
        *,
        host: RuntimeHost,
        provider: ModelInputProvider,
        session_edges: SessionEdges,
        pipeline: StepPipeline,
    ) -> RunResult:
        session: InferenceSession | None = None
        final_status: DriverStatus = "completed"
        final_reason: str | None = None
        final_error: Exception | None = None
        setup_ok = False
        generation = 0
        first_step_started = False
        invariant_error: DriverInvariantError | None = None
        try:
            activation, clock = _realtime_activation_and_clock(session_edges)
            input_source = _realtime_input_source(session_edges)
            activation_result = await activation.wait_until_active(clock)
            if not activation_result.activated:
                final_status = "not_activated"
                final_reason = activation_result.reason
            elif not session_edges.transport.is_active():
                final_status = "not_activated"
                final_reason = "transport closed before first step"
            else:
                try:
                    initial_input = await host.call_async(
                        provider.prepare_initial_input
                    )
                    session = await host.call_async(host.start_session, initial_input)
                    session_info = await host.call_async(_session_info, session)
                    session_edges.output_sink.open(session_info)
                    session_edges.output_sink.begin_generation(generation)
                    setup_ok = True
                except DriverInvariantError:
                    raise
                except Exception as exc:
                    action = session_edges.error_policy.handle_setup_error(exc)
                    if action.drop_chunk or action.result_status == "completed":
                        raise DriverInvariantError(
                            "Setup failures must resolve to failed or skipped."
                        ) from exc
                    session_edges.metrics.record_error(exc, action)
                    final_status = action.result_status
                    final_reason = str(exc)
                    final_error = exc if action.result_status == "failed" else None

            while setup_ok:
                if session is None:
                    raise DriverInvariantError("setup_ok was set without a session.")
                if not session_edges.transport.is_active():
                    if not first_step_started:
                        final_status = "not_activated"
                        final_reason = "transport closed before first step"
                    break
                try:
                    request = await _next_step_requirements_async(
                        host=host,
                        session=session,
                    )
                    if request is None:
                        break
                    window_result = await input_source.next_realtime_window(
                        request=request,
                        clock=clock,
                    )
                    session_edges.metrics.record_catch_up(window_result.catch_up)
                    if (
                        not session_edges.transport.is_active()
                        and not first_step_started
                    ):
                        final_status = "not_activated"
                        final_reason = "transport closed before first step"
                        break
                    outcome = await host.call_async(
                        pipeline.execute_step,
                        request=request,
                        user_window=window_result.window,
                        provider=provider,
                        session=session,
                        output=session_edges.output_sink,
                        metrics=session_edges.metrics,
                    )
                    first_step_started = True
                    if outcome.control.reset:
                        await host.call_async(
                            session.reset,
                            outcome.control.reset_input,
                        )
                        if not outcome.control.provider_already_reset:
                            await host.call_async(
                                provider.reset,
                                outcome.control.reset_input,
                            )
                        generation += 1
                        session_edges.output_sink.begin_generation(generation)
                        continue
                    if outcome.control.close_session:
                        break
                    if outcome.output.should_stop:
                        break
                    if outcome.output.backpressure_s > 0:
                        await clock.apply_backpressure(outcome.output.backpressure_s)
                except DriverInvariantError:
                    raise
                except Exception as exc:
                    action = session_edges.error_policy.handle(exc)
                    session_edges.metrics.record_error(exc, action)
                    if action.close_session:
                        final_status = action.result_status
                        final_reason = str(exc)
                        final_error = exc if action.result_status == "failed" else None
                        break
                    if action.drop_chunk:
                        continue
                    final_status = "failed"
                    final_reason = str(exc)
                    final_error = exc
                    break
        except asyncio.CancelledError:
            uncancel_current_task()
            final_status = "cancelled"
            final_reason = (
                "cancelled before first step" if session is None else "cancelled"
            )
            final_error = None
        except DriverInvariantError as exc:
            invariant_error = exc
            final_status = "failed"
            final_reason = str(exc)
            final_error = exc
        except Exception as exc:
            final_status = "failed"
            final_reason = str(exc)
            final_error = exc

        result = await shielded_session_cleanup(
            host=host,
            session=session,
            provider=provider,
            session_edges=session_edges,
            status=final_status,
            reason=final_reason,
            error=final_error,
            timeout_s=self.cleanup_timeout_s,
        )
        if invariant_error is not None:
            raise invariant_error
        return result


def _mark_host_cleanup_failed(host: RuntimeHost, exc: Exception | None = None) -> None:
    host.mark_unhealthy(_MODEL_CLEANUP_FAILED_REASON, exc)


def run_demo_session(
    *,
    context: RunContext,
    spec: DemoSpec,
    scenario: PreparedScenario,
    adapter: DemoAdapter,
    run_mode: RunMode,
    pipeline: StepPipeline,
    reservation: SessionReservation | None = None,
) -> RunResult:
    """Run one prepared demo session through a selected run mode."""
    if reservation is None:
        reservation = context.admission.try_reserve()
    if reservation is None:
        result = RunResult.rejected(reason="busy")
        context.run_metrics.record_session(result)
        return result

    provider: Any | None = None
    session_edges: SessionEdges | None = None
    driver_started = False
    try:
        create_provider = getattr(adapter, "create_model_input_provider")
        provider = context.host.call(create_provider, spec, scenario)
        run_mode.validate_session(
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            provider=provider,
        )
        session_edges = run_mode.create_session_edges(
            context=context,
            spec=spec,
            scenario=scenario,
            provider=provider,
            adapter=adapter,
        )
        resolved_capabilities = resolve_run_capabilities(
            spec=spec,
            provider=provider,
            session_edges=session_edges,
        )
        validate_resolved_run(
            spec=spec,
            adapter=adapter,
            provider=provider,
            run_mode=run_mode,
            session_edges=session_edges,
            resolved=resolved_capabilities,
        )
        if session_edges.is_closed:
            raise DriverInvariantError(
                "RunMode returned already closed SessionEdges; session edges "
                "must not be reused."
            )
        driver = run_mode.select_driver()
        driver_started = True
        result = _run_sync_driver(
            driver=driver,
            host=context.host,
            provider=provider,
            session_edges=session_edges,
            pipeline=pipeline,
        )
        context.run_metrics.record_session(result)
        return result
    except DriverInvariantError as exc:
        _record_run_session_error(context, exc)
        if provider is not None and not driver_started:
            _close_partial_provider_sync(
                context=context,
                provider=provider,
                session_edges=session_edges,
            )
        if session_edges is not None and (
            driver_started or not session_edges.is_closed
        ):
            result = session_edges.close_result(
                status="failed",
                reason=str(exc),
                error=exc,
            )
            context.run_metrics.record_session(result)
        raise
    except Exception as exc:
        _record_run_session_error(context, exc)
        if provider is not None and not driver_started:
            _close_partial_provider_sync(
                context=context,
                provider=provider,
                session_edges=session_edges,
            )
        if session_edges is not None and (
            driver_started or not session_edges.is_closed
        ):
            result = session_edges.close_result(
                status="failed",
                reason=str(exc),
                error=exc,
            )
        else:
            result = RunResult(status="failed", reason=str(exc), error=exc)
        context.run_metrics.record_session(result)
        return result
    finally:
        reservation.release()


async def run_demo_session_async(
    *,
    context: RunContext,
    spec: DemoSpec,
    scenario: PreparedScenario,
    adapter: DemoAdapter,
    run_mode: RunMode,
    pipeline: StepPipeline,
    reservation: SessionReservation | None = None,
) -> RunResult:
    """Run one prepared async/realtime demo session through a selected run mode."""
    if reservation is None:
        reservation = context.admission.try_reserve()
    if reservation is None:
        result = RunResult.rejected(reason="busy")
        context.run_metrics.record_session(result)
        return result

    provider: Any | None = None
    session_edges: SessionEdges | None = None
    try:
        try:
            create_provider = getattr(adapter, "create_model_input_provider")
            provider = await context.host.call_async(create_provider, spec, scenario)
            run_mode.validate_session(
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                provider=provider,
            )
            session_edges = run_mode.create_session_edges(
                context=context,
                spec=spec,
                scenario=scenario,
                provider=provider,
                adapter=adapter,
            )
            resolved_capabilities = resolve_run_capabilities(
                spec=spec,
                provider=provider,
                session_edges=session_edges,
            )
            validate_resolved_run(
                spec=spec,
                adapter=adapter,
                provider=provider,
                run_mode=run_mode,
                session_edges=session_edges,
                resolved=resolved_capabilities,
            )
            if session_edges.is_closed:
                raise DriverInvariantError(
                    "RunMode returned already closed SessionEdges; session edges "
                    "must not be reused."
                )
            driver = run_mode.select_driver()
            result = await _run_async_driver(
                driver=driver,
                host=context.host,
                provider=provider,
                session_edges=session_edges,
                pipeline=pipeline,
            )
            context.run_metrics.record_session(result)
            return result
        except asyncio.CancelledError:
            uncancel_current_task()
            result = await _close_partial_session_async(
                context=context,
                provider=provider,
                session_edges=session_edges,
                status="cancelled",
                reason="cancelled during session assembly",
                error=None,
                close_provider=_needs_partial_provider_cleanup(session_edges),
            )
            context.run_metrics.record_session(result)
            return result
        except DriverInvariantError as exc:
            _record_run_session_error(context, exc)
            should_record_session = session_edges is not None
            result = await _close_partial_session_async(
                context=context,
                provider=provider,
                session_edges=session_edges,
                status="failed",
                reason=str(exc),
                error=exc,
                close_provider=_needs_partial_provider_cleanup(session_edges),
            )
            if should_record_session:
                context.run_metrics.record_session(result)
            raise
        except Exception as exc:
            _record_run_session_error(context, exc)
            result = await _close_partial_session_async(
                context=context,
                provider=provider,
                session_edges=session_edges,
                status="failed",
                reason=str(exc),
                error=exc,
                close_provider=_needs_partial_provider_cleanup(session_edges),
            )
            context.run_metrics.record_session(result)
            return result
    finally:
        reservation.release()


def _session_info(session: InferenceSession) -> SessionInfo:
    session_info = getattr(session, "session_info", None)
    if not callable(session_info):
        return SessionInfo()
    value = session_info()
    if not isinstance(value, SessionInfo):
        raise TypeError(
            "session.session_info() must return SessionInfo, "
            f"got {type(value).__name__}."
        )
    return value


def _next_step_requirements(
    *,
    host: RuntimeHost,
    session: InferenceSession,
) -> StepRequirements | None:
    next_requirements = getattr(session, "next_step_requirements", None)
    if callable(next_requirements):
        return _coerce_step_requirements(host.call(next_requirements))

    next_request = getattr(session, "next_step_request", None)
    if not callable(next_request):
        raise TypeError(
            "InferenceSession must provide next_step_requirements() or "
            "legacy next_step_request()."
        )
    return _coerce_step_requirements(host.call(next_request))


async def _next_step_requirements_async(
    *,
    host: RuntimeHost,
    session: InferenceSession,
) -> StepRequirements | None:
    next_requirements = getattr(session, "next_step_requirements", None)
    if callable(next_requirements):
        return _coerce_step_requirements(await host.call_async(next_requirements))

    next_request = getattr(session, "next_step_request", None)
    if not callable(next_request):
        raise TypeError(
            "InferenceSession must provide next_step_requirements() or "
            "legacy next_step_request()."
        )
    return _coerce_step_requirements(await host.call_async(next_request))


def _coerce_step_requirements(value: object) -> StepRequirements | None:
    if value is None:
        return None
    if isinstance(value, StepRequirements):
        return value
    if isinstance(value, StepRequest):
        return step_requirements_from_request(value)
    raise TypeError(
        "Session next-step method must return StepRequirements, legacy "
        f"StepRequest, or None; got {type(value).__name__}."
    )


def _close_safely(close: Any, session_edges: SessionEdges) -> bool:
    try:
        close()
    except Exception as exc:
        session_edges.record_cleanup_error(exc)
        return False
    return True


def _close_on_host_best_effort(
    *,
    host: RuntimeHost,
    close: Any,
    session_edges: SessionEdges,
) -> bool:
    try:
        cleanup_succeeded = host.call(_close_safely, close, session_edges)
    except Exception as exc:
        # If the host/worker is already unavailable, do not fall back to calling
        # model-affine cleanup directly on the caller thread. Record the loss and
        # let close_result finalize output, transport, and metrics.
        session_edges.record_cleanup_error(exc)
        _mark_host_cleanup_failed(host, exc)
        return False
    if not cleanup_succeeded:
        _mark_host_cleanup_failed(host)
        return False
    return True


def _close_partial_provider_sync(
    *,
    context: RunContext,
    provider: Any,
    session_edges: SessionEdges | None,
) -> None:
    if session_edges is not None:
        _close_on_host_best_effort(
            host=context.host,
            close=provider.close,
            session_edges=session_edges,
        )
        return
    try:
        cleanup_succeeded = context.host.call(
            _close_run_provider_safely,
            provider.close,
            context,
        )
    except Exception as exc:
        _record_run_cleanup_error(context, exc)
        _mark_host_cleanup_failed(context.host, exc)
        return
    if not cleanup_succeeded:
        _mark_host_cleanup_failed(context.host)


def _close_run_provider_safely(close: Any, context: RunContext) -> bool:
    try:
        close()
    except Exception as exc:
        _record_run_cleanup_error(context, exc)
        return False
    return True


async def shielded_session_cleanup(
    *,
    host: RuntimeHost,
    session: InferenceSession | None,
    provider: ModelInputProvider,
    session_edges: SessionEdges,
    status: DriverStatus,
    reason: str | None,
    error: Exception | None,
    timeout_s: float = CLEANUP_TIMEOUT_S,
) -> RunResult:
    """Close realtime session resources exactly once without leaking cancellation."""

    if timeout_s <= 0:
        session_edges.record_cleanup_error(ValueError("timeout_s must be > 0."))
        return session_edges.close_result(status=status, reason=reason, error=error)

    async def cleanup() -> RunResult:
        unhealthy_reason = await _close_model_resources_async(
            host=host,
            session=session,
            provider=provider,
            session_edges=session_edges,
            timeout_s=timeout_s,
        )
        if unhealthy_reason is not None:
            host.mark_unhealthy(unhealthy_reason)
        return session_edges.close_result(
            status=status,
            reason=reason,
            error=error,
        )

    try:
        cleanup_task = asyncio.create_task(cleanup())
    except RuntimeError as exc:
        session_edges.record_cleanup_error(exc)
        return session_edges.close_result(status=status, reason=reason, error=error)

    session_edges.cleanup_tasks.add(cleanup_task)
    try:
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                uncancel_current_task()
                continue
            except Exception:
                break
        return _cleanup_result(cleanup_task, session_edges, status, reason, error)
    finally:
        session_edges.cleanup_tasks.discard(cleanup_task)


def _run_sync_driver(
    *,
    driver: object,
    host: RuntimeHost,
    provider: ModelInputProvider,
    session_edges: SessionEdges,
    pipeline: StepPipeline,
) -> RunResult:
    run_one_session = getattr(driver, "run_one_session", None)
    if not callable(run_one_session):
        raise TypeError(
            "RunMode.select_driver() must return an object with run_one_session(...)."
        )
    result = run_one_session(
        host=host,
        provider=provider,
        session_edges=session_edges,
        pipeline=pipeline,
    )
    if inspect.isawaitable(result):
        raise TypeError(
            "run_demo_session(...) requires a synchronous session driver; "
            "use run_demo_session_async(...) for async drivers."
        )
    if not isinstance(result, RunResult):
        raise TypeError(
            "Session driver run_one_session(...) must return RunResult, "
            f"got {type(result).__name__}."
        )
    return result


async def _run_async_driver(
    *,
    driver: object,
    host: RuntimeHost,
    provider: ModelInputProvider,
    session_edges: SessionEdges,
    pipeline: StepPipeline,
) -> RunResult:
    run_one_session = getattr(driver, "run_one_session", None)
    if not callable(run_one_session):
        raise TypeError(
            "RunMode.select_driver() must return an object with run_one_session(...)."
        )
    result = run_one_session(
        host=host,
        provider=provider,
        session_edges=session_edges,
        pipeline=pipeline,
    )
    if not inspect.isawaitable(result):
        raise TypeError("run_demo_session_async(...) requires an async session driver.")
    resolved = await result
    if not isinstance(resolved, RunResult):
        raise TypeError(
            "Async session driver run_one_session(...) must return RunResult, "
            f"got {type(resolved).__name__}."
        )
    return resolved


async def _close_partial_session_async(
    *,
    context: RunContext,
    provider: Any | None,
    session_edges: SessionEdges | None,
    status: DriverStatus,
    reason: str | None,
    error: Exception | None,
    close_provider: bool,
) -> RunResult:
    if provider is not None and close_provider and session_edges is not None:
        return await shielded_session_cleanup(
            host=context.host,
            session=None,
            provider=provider,
            session_edges=session_edges,
            status=status,
            reason=reason,
            error=error,
        )
    if provider is not None and close_provider:
        await _close_provider_async(
            context=context,
            provider=provider,
            session_edges=session_edges,
        )
    if session_edges is not None:
        return session_edges.close_result(status=status, reason=reason, error=error)
    return RunResult(status=status, reason=reason, error=error)


def _needs_partial_provider_cleanup(session_edges: SessionEdges | None) -> bool:
    return session_edges is None or not session_edges.is_closed


async def _close_provider_async(
    *,
    context: RunContext,
    provider: Any,
    session_edges: SessionEdges | None,
) -> None:
    try:
        close_task = asyncio.create_task(context.host.call_async(provider.close))
    except RuntimeError as close_exc:
        _record_provider_cleanup_error(
            context=context,
            session_edges=session_edges,
            exc=close_exc,
        )
        return

    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            uncancel_current_task()
            continue
        except Exception:
            break

    try:
        await close_task
    except asyncio.CancelledError:
        uncancel_current_task()
        _record_provider_cleanup_error(
            context=context,
            session_edges=session_edges,
            exc=RuntimeError("provider cleanup was cancelled"),
        )
    except Exception as close_exc:
        _record_provider_cleanup_error(
            context=context,
            session_edges=session_edges,
            exc=close_exc,
        )


def _record_provider_cleanup_error(
    *,
    context: RunContext,
    session_edges: SessionEdges | None,
    exc: Exception,
) -> None:
    if session_edges is not None:
        session_edges.record_cleanup_error(exc)
    else:
        _record_run_cleanup_error(context, exc)
    # Partial async assembly may only have a provider to close. If that
    # model-affine cleanup fails, quarantine the host instead of admitting a new
    # session onto a worker that may still own model resources.
    _mark_host_cleanup_failed(context.host, exc)


def _record_run_cleanup_error(context: RunContext, exc: Exception) -> None:
    try:
        context.run_metrics.record_cleanup_error(exc)
    except Exception:
        return


def _record_run_session_error(context: RunContext, exc: Exception) -> None:
    try:
        context.run_metrics.record_session_error(exc)
    except Exception:
        return


async def _close_model_resources_async(
    *,
    host: RuntimeHost,
    session: InferenceSession | None,
    provider: ModelInputProvider,
    session_edges: SessionEdges,
    timeout_s: float,
) -> str | None:
    try:
        resources_closed = await asyncio.wait_for(
            host.call_async(
                _close_model_resources_safely,
                session.close if session is not None else None,
                provider.close,
                session_edges,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as exc:
        # Keep provider cleanup ordered behind session cleanup on the model worker.
        # A timed-out session close may still hold CUDA/Triton state, so running
        # provider cleanup on another thread or replacing the worker is unsafe.
        # The caller marks the host unhealthy so future sessions reject instead.
        session_edges.record_orphaned_cleanup(exc)
        return _MODEL_CLEANUP_TIMED_OUT_REASON
    except Exception as exc:
        session_edges.record_cleanup_error(exc)
        return _MODEL_CLEANUP_FAILED_REASON
    if not resources_closed:
        return _MODEL_CLEANUP_FAILED_REASON
    return None


def _close_model_resources_safely(
    session_close: Any | None,
    provider_close: Any,
    session_edges: SessionEdges,
) -> bool:
    resources_closed = True
    # Session and provider close are intentionally ordered on the model worker.
    # If session close hangs, timeout handling records orphaned cleanup and
    # quarantines the host rather than moving provider close to another thread.
    if session_close is not None:
        resources_closed = _close_safely(session_close, session_edges)
    return _close_safely(provider_close, session_edges) and resources_closed


def _cleanup_result(
    cleanup_task: asyncio.Task[RunResult],
    session_edges: SessionEdges,
    status: DriverStatus,
    reason: str | None,
    error: Exception | None,
) -> RunResult:
    if cleanup_task.done() and not cleanup_task.cancelled():
        exc = cleanup_task.exception()
        if exc is None:
            return cleanup_task.result()
        if isinstance(exc, Exception):
            session_edges.record_cleanup_error(exc)
        else:
            session_edges.record_cleanup_error(
                RuntimeError(f"cleanup failed with {type(exc).__name__}")
            )
    return session_edges.close_result(status=status, reason=reason, error=error)


def _realtime_activation_and_clock(
    session_edges: SessionEdges,
) -> tuple[ActivationPolicy, RealtimeClock]:
    activation = session_edges.activation
    if activation is None:
        raise DriverInvariantError(
            "RealtimeSessionDriver requires SessionEdges.activation."
        )
    clock = session_edges.clock
    if not isinstance(clock, RealtimeClock):
        raise DriverInvariantError("RealtimeSessionDriver requires a RealtimeClock.")
    return activation, clock


def _realtime_input_source(session_edges: SessionEdges) -> Any:
    input_source = session_edges.input_source
    next_realtime_window = getattr(input_source, "next_realtime_window", None)
    if not callable(next_realtime_window):
        raise DriverInvariantError(
            "RealtimeSessionDriver requires a RealtimeInputSource."
        )
    return input_source


def uncancel_current_task() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    uncancel = getattr(task, "uncancel", None)
    if not callable(uncancel):
        return
    cancelling = getattr(task, "cancelling", None)
    if not callable(cancelling):
        return
    while cancelling():
        uncancel()


__all__ = [
    "BatchSessionDriver",
    "CLEANUP_TIMEOUT_S",
    "DriverInvariantError",
    "RealtimeSessionDriver",
    "run_demo_session",
    "run_demo_session_async",
    "shielded_session_cleanup",
    "uncancel_current_task",
]
