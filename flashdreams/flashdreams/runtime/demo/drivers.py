# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session drivers and helpers for demo runtime vertical slices."""

from __future__ import annotations

from typing import Any

from flashdreams.runtime.interfaces import InferenceSession

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
from .session_inputs import ModelInputProvider
from .spec import DemoAdapter, DemoSpec, PreparedScenario


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
                try:
                    if session_edges.input_source.is_finished():
                        break
                    request = host.call(session.next_step_request)
                    if request is None:
                        break
                    user_window = session_edges.input_source.next_window(request)
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
                host.call(_close_safely, session.close, session_edges)
            host.call(_close_safely, provider.close, session_edges)
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
                    host.call(_close_safely, session.close, session_edges)
                host.call(_close_safely, provider.close, session_edges)

        return session_edges.close_result(
            status=final_status,
            reason=final_reason,
            error=final_error,
        )


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
    reservation = reservation or context.admission.try_reserve()
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
        driver = run_mode.select_driver()
        if not isinstance(driver, BatchSessionDriver):
            raise TypeError(
                "Phase 2 run_demo_session supports BatchSessionDriver only, "
                f"got {type(driver).__name__}."
            )
        driver_started = True
        result = driver.run_one_session(
            host=context.host,
            provider=provider,
            session_edges=session_edges,
            pipeline=pipeline,
        )
        context.run_metrics.record_session(result)
        return result
    except DriverInvariantError as exc:
        if provider is not None and not driver_started:
            try:
                context.host.call(provider.close)
            except Exception as close_exc:
                if session_edges is not None:
                    session_edges.metrics.record_cleanup_error(close_exc)
                else:
                    context.run_metrics.record_cleanup_error(close_exc)
        if session_edges is not None:
            result = session_edges.close_result(
                status="failed",
                reason=str(exc),
                error=exc,
            )
            context.run_metrics.record_session(result)
        raise
    except Exception as exc:
        if provider is not None and not driver_started:
            try:
                context.host.call(provider.close)
            except Exception as close_exc:
                if session_edges is not None:
                    session_edges.metrics.record_cleanup_error(close_exc)
                else:
                    context.run_metrics.record_cleanup_error(close_exc)
        if session_edges is not None:
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


def _close_safely(close: Any, session_edges: SessionEdges) -> None:
    try:
        close()
    except Exception as exc:
        session_edges.metrics.record_cleanup_error(exc)


__all__ = [
    "BatchSessionDriver",
    "DriverInvariantError",
    "run_demo_session",
]
