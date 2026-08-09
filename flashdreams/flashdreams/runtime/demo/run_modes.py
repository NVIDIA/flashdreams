# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run/session result and policy helpers for demo session drivers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.output import OutputArtifact

from .host import ModelWarmupPlan
from .outputs import OutputDecision, OutputSink

if TYPE_CHECKING:
    from .host import RuntimeHost
    from .pipeline import StepPipeline
    from .session_inputs import InputSource, ModelInputProvider
    from .spec import DemoAdapter, DemoSpec, PreparedScenario

SessionStatus = Literal[
    "completed",
    "failed",
    "skipped",
    "cancelled",
    "rejected",
    "not_activated",
]

DriverStatus = Literal[
    "completed",
    "failed",
    "skipped",
    "cancelled",
    "not_activated",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class MetricsSnapshot:
    """Closed session or run metrics summary."""

    counters: Mapping[str, int | float] = field(default_factory=dict)
    timings: Mapping[str, Sequence[float]] = field(default_factory=dict)
    session_statuses: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "counters", freeze_mapping(self.counters))
        object.__setattr__(
            self,
            "timings",
            freeze_mapping(
                {key: tuple(values) for key, values in self.timings.items()}
            ),
        )
        object.__setattr__(self, "session_statuses", tuple(self.session_statuses))
        object.__setattr__(self, "errors", tuple(self.errors))


@dataclass(frozen=True, kw_only=True, slots=True)
class RunResult:
    """Outcome of one demo session."""

    __hash__ = None

    status: SessionStatus
    artifacts: Sequence[OutputArtifact] = ()
    metrics: MetricsSnapshot | None = None
    reason: str | None = None
    error: Exception | None = None

    @classmethod
    def rejected(cls, reason: str) -> "RunResult":
        """Admission refused the session. The only no-session result helper."""
        return cls(status="rejected", reason=reason)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True, kw_only=True, slots=True)
class RunSummary:
    """Summary for a run context after one or more sessions."""

    metrics: MetricsSnapshot
    sessions: Sequence[RunResult] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))


@dataclass(frozen=True, kw_only=True, slots=True)
class ErrorAction:
    """Driver policy decision for an operational error."""

    close_session: bool = True
    drop_chunk: bool = False
    continue_next_scenario: bool = False
    result_status: Literal["completed", "failed", "skipped"] = "failed"


class DefaultErrorPolicy:
    """Default policy: operational errors fail the current session."""

    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed")

    def handle(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed")


@runtime_checkable
class ErrorPolicy(Protocol):
    """Maps driver-observed exceptions to session outcomes."""

    def handle_setup_error(self, exc: Exception) -> ErrorAction: ...

    def handle(self, exc: Exception) -> ErrorAction: ...


@runtime_checkable
class SessionMetricsRecorder(Protocol):
    """Metrics callbacks consumed by the Phase 2 drivers and pipeline."""

    def record_step(
        self,
        *,
        request: object,
        user_window: object,
        inference_input: object,
        result: object,
        decision: OutputDecision,
    ) -> None: ...

    def record_control(
        self,
        *,
        request: object,
        user_window: object,
        control: object,
    ) -> None: ...

    def record_error(self, exc: Exception, action: ErrorAction) -> None: ...

    def record_cleanup_error(self, exc: Exception) -> None: ...

    def record_session(self, result: RunResult) -> None: ...

    def close(self) -> MetricsSnapshot: ...


@dataclass(slots=True)
class InMemorySessionMetricsRecorder:
    """Small non-raising metrics recorder for driver tests and fake demos."""

    step_count: int = 0
    control_count: int = 0
    errors: list[str] = field(default_factory=list)
    cleanup_errors: list[str] = field(default_factory=list)
    sessions: list[RunResult] = field(default_factory=list)
    closed: bool = False

    def record_step(
        self,
        *,
        request: object,
        user_window: object,
        inference_input: object,
        result: object,
        decision: OutputDecision,
    ) -> None:
        del request, user_window, inference_input, result, decision
        if not self.closed:
            self.step_count += 1

    def record_control(
        self,
        *,
        request: object,
        user_window: object,
        control: object,
    ) -> None:
        del request, user_window, control
        if not self.closed:
            self.control_count += 1

    def record_error(self, exc: Exception, action: ErrorAction) -> None:
        del action
        if not self.closed:
            self.errors.append(str(exc))

    def record_cleanup_error(self, exc: Exception) -> None:
        if not self.closed:
            self.cleanup_errors.append(str(exc))

    def record_session(self, result: RunResult) -> None:
        if not self.closed:
            self.sessions.append(result)

    def close(self) -> MetricsSnapshot:
        self.closed = True
        return MetricsSnapshot(
            counters={
                "steps": self.step_count,
                "controls": self.control_count,
                "sessions": len(self.sessions),
                "cleanup_errors": len(self.cleanup_errors),
            },
            session_statuses=tuple(result.status for result in self.sessions),
            errors=tuple((*self.errors, *self.cleanup_errors)),
        )


class NoopTransportService:
    """Idempotent placeholder transport for batch sessions."""

    def __init__(self) -> None:
        self.closed = False

    def is_active(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


@runtime_checkable
class TransportService(Protocol):
    """Per-session transport lifecycle hook."""

    def is_active(self) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class SessionReservation(Protocol):
    """Admission reservation for one session."""

    def release(self) -> None: ...


class SingleSessionAdmissionPolicy:
    """Atomic single-session admission policy."""

    def __init__(self, *, health_check: Any | None = None) -> None:
        self._lock = Lock()
        self._reserved = False
        self._health_check = health_check

    def try_reserve(self) -> SessionReservation | None:
        with self._lock:
            if self._reserved or not self._is_healthy():
                return None
            self._reserved = True
            return _SingleSessionReservation(self)

    def _release(self) -> None:
        with self._lock:
            self._reserved = False

    def _is_healthy(self) -> bool:
        if self._health_check is None:
            return True
        return bool(self._health_check())


class _SingleSessionReservation:
    def __init__(self, policy: SingleSessionAdmissionPolicy) -> None:
        self._policy = policy
        self._released = False
        self.release_count = 0

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.release_count += 1
        self._policy._release()


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Atomically reserves session capacity or rejects."""

    def try_reserve(self) -> SessionReservation | None: ...


@runtime_checkable
class SessionDriver(Protocol):
    """Synchronous one-session driver selected by a run mode."""

    def run_one_session(
        self,
        *,
        host: "RuntimeHost",
        provider: "ModelInputProvider",
        session_edges: "SessionEdges",
        pipeline: "StepPipeline",
    ) -> RunResult: ...


@runtime_checkable
class AsyncSessionDriver(Protocol):
    """Async one-session driver selected by realtime run modes."""

    async def run_one_session(
        self,
        *,
        host: "RuntimeHost",
        provider: "ModelInputProvider",
        session_edges: "SessionEdges",
        pipeline: "StepPipeline",
    ) -> RunResult: ...


@dataclass(slots=True)
class RunContext:
    """Run-scoped services shared by one or more demo sessions."""

    host: "RuntimeHost"
    run_metrics: SessionMetricsRecorder
    admission: AdmissionPolicy
    model_warmup_plan: ModelWarmupPlan = field(default_factory=ModelWarmupPlan)
    services: Mapping[str, object] = field(default_factory=dict)
    cleanup_tasks: set[asyncio.Task[RunResult]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.services = freeze_mapping(self.services)

    def close(self) -> RunSummary:
        if self.cleanup_tasks:
            raise RuntimeError(
                "Pending session cleanup tasks; async runs must await close_async()."
            )
        for service in self.services.values():
            close = getattr(service, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    self.run_metrics.record_cleanup_error(exc)
        return RunSummary(
            metrics=self.run_metrics.close(),
            sessions=tuple(getattr(self.run_metrics, "sessions", ())),
        )

    async def close_async(self) -> RunSummary:
        while self.cleanup_tasks:
            pending = tuple(self.cleanup_tasks)
            await asyncio.gather(*pending, return_exceptions=True)
            self.cleanup_tasks.difference_update(pending)
        return self.close()


@dataclass(slots=True)
class SessionEdges:
    """Per-session input/output/policy bundle consumed by drivers."""

    input_source: "InputSource"
    output_sink: OutputSink
    cleanup_tasks: set[asyncio.Task[RunResult]]
    metrics: SessionMetricsRecorder = field(
        default_factory=InMemorySessionMetricsRecorder
    )
    error_policy: ErrorPolicy = field(default_factory=DefaultErrorPolicy)
    transport: TransportService = field(default_factory=NoopTransportService)
    clock: object | None = None
    activation: object | None = None
    _closed_result: RunResult | None = field(default=None, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        """Return whether ``close_result(...)`` has already finalized this session."""
        return self._closed_result is not None

    def close_result(
        self,
        *,
        status: DriverStatus = "completed",
        reason: str | None = None,
        error: Exception | None = None,
    ) -> RunResult:
        """Idempotently close output, transport, and metrics once."""
        if self._closed_result is not None:
            return self._closed_result

        artifacts: Sequence[OutputArtifact] = ()
        try:
            artifacts = tuple(self.output_sink.close())
        except Exception as exc:
            self.metrics.record_cleanup_error(exc)
        try:
            self.transport.close()
        except Exception as exc:
            self.metrics.record_cleanup_error(exc)
        try:
            metrics = self.metrics.close()
        except Exception as exc:
            metrics = MetricsSnapshot(errors=(f"metrics.close failed: {exc}",))
        self._closed_result = RunResult(
            status=status,
            artifacts=artifacts,
            metrics=metrics,
            reason=reason,
            error=error,
        )
        return self._closed_result


@runtime_checkable
class RunMode(Protocol):
    """Run/session construction strategy consumed by shared helpers."""

    name: str

    def validate_run(
        self,
        *,
        spec: "DemoSpec",
        adapter: "DemoAdapter",
    ) -> None: ...

    def validate_session(
        self,
        *,
        spec: "DemoSpec",
        scenario: "PreparedScenario",
        adapter: "DemoAdapter",
        provider: "ModelInputProvider",
    ) -> None: ...

    def create_run_context(
        self,
        *,
        spec: "DemoSpec",
        adapter: "DemoAdapter",
        host: "RuntimeHost",
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext: ...

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: "DemoSpec",
        scenario: "PreparedScenario",
        provider: "ModelInputProvider",
        adapter: "DemoAdapter",
    ) -> SessionEdges: ...

    def select_driver(self) -> SessionDriver | AsyncSessionDriver: ...


@runtime_checkable
class RunModeWarmup(Protocol):
    """Optional run-mode warmup for output or transport services."""

    def warmup_context(
        self,
        *,
        context: RunContext,
        spec: "DemoSpec",
        scenario: "PreparedScenario",
        adapter: "DemoAdapter",
    ) -> None: ...


__all__ = [
    "AdmissionPolicy",
    "AsyncSessionDriver",
    "DefaultErrorPolicy",
    "DriverStatus",
    "ErrorAction",
    "ErrorPolicy",
    "InMemorySessionMetricsRecorder",
    "MetricsSnapshot",
    "NoopTransportService",
    "RunContext",
    "RunMode",
    "RunModeWarmup",
    "RunResult",
    "RunSummary",
    "SessionEdges",
    "SessionDriver",
    "SessionMetricsRecorder",
    "SessionReservation",
    "SessionStatus",
    "SingleSessionAdmissionPolicy",
    "TransportService",
]
