# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run/session result and policy helpers for demo session drivers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from flashdreams.demo.io import OutputSink
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.metrics import (
    InMemoryMetricsRecorder,
    MetricsRecorder,
    MetricsSnapshot,
)
from flashdreams.runtime.output import OutputArtifact

from .host import ModelWarmupPlan, WarmupSessionInputs

if TYPE_CHECKING:
    from .host import RuntimeHost
    from .pipeline import StepPipeline
    from .session_inputs import InputSource, ModelInputProvider
    from .spec import DemoAdapter, DemoSpec, PreparedScenario
    from .timing import ActivationPolicy, DeterministicClock, RealtimeClock

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


class Mp4ErrorPolicy(DefaultErrorPolicy):
    """Abort MP4 sessions on setup or step errors."""


class NullErrorPolicy(DefaultErrorPolicy):
    """Abort headless/null sessions on setup or step errors."""


class LocalWindowErrorPolicy(DefaultErrorPolicy):
    """Abort local-window sessions unless a future UI policy overrides it."""


class BenchmarkErrorPolicy(DefaultErrorPolicy):
    """Close failed scenarios while letting benchmark loops continue."""

    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed", continue_next_scenario=True)

    def handle(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed", continue_next_scenario=True)


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCErrorPolicy:
    """Drop configured recoverable realtime errors, otherwise close the session."""

    recoverable_exception_types: tuple[type[Exception], ...] = ()

    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed")

    def handle(self, exc: Exception) -> ErrorAction:
        if self.recoverable_exception_types and isinstance(
            exc, self.recoverable_exception_types
        ):
            return ErrorAction(
                close_session=False,
                drop_chunk=True,
                result_status="failed",
            )
        return ErrorAction(result_status="failed")


@dataclass(frozen=True, kw_only=True, slots=True)
class RunModeCapabilities:
    """Run-mode requirements and output/transport capabilities."""

    realtime: bool = False
    requires_finite_input: bool = False
    supports_backpressure: bool = False
    supports_interactive_events: bool = False
    supports_artifacts: bool = False


SessionMetricsRecorder = MetricsRecorder
InMemorySessionMetricsRecorder = InMemoryMetricsRecorder


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
    clock: "RealtimeClock | DeterministicClock | None" = None
    activation: "ActivationPolicy | None" = None
    _closed_result: RunResult | None = field(default=None, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        """Return whether ``close_result(...)`` has already finalized this session."""
        return self._closed_result is not None

    def record_cleanup_error(self, exc: Exception) -> None:
        """Record a cleanup error without letting metrics failures block teardown."""
        try:
            self.metrics.record_cleanup_error(exc)
        except Exception:
            return

    def record_orphaned_cleanup(self, exc: Exception) -> None:
        """Record timed-out worker cleanup without blocking teardown."""
        try:
            self.metrics.record_orphaned_cleanup(exc)
        except Exception:
            return

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
            self.record_cleanup_error(exc)
        try:
            self.transport.close()
        except Exception as exc:
            self.record_cleanup_error(exc)
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
    capabilities: RunModeCapabilities

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


def build_model_warmup_plan(
    *,
    host: "RuntimeHost",
    adapter: "DemoAdapter",
    spec: "DemoSpec",
    scenario: "PreparedScenario",
) -> ModelWarmupPlan:
    """Build a host-owned warmup plan through the model-affine worker."""

    create_sessions = getattr(adapter, "create_model_warmup_sessions", None)
    if create_sessions is None:
        return ModelWarmupPlan()
    if not callable(create_sessions):
        raise TypeError(
            "Demo adapter create_model_warmup_sessions attribute must be callable."
        )
    sessions = host.call(create_sessions, spec, scenario)
    return ModelWarmupPlan(sessions=_coerce_warmup_sessions(sessions))


def warmup_run_context(
    *,
    context: RunContext,
    spec: "DemoSpec",
    scenario: "PreparedScenario",
    adapter: "DemoAdapter",
    run_mode: object,
) -> None:
    """Run model warmup, then optional output/transport warmup for a context."""

    context.host.warmup(context.model_warmup_plan)
    warmup_context = getattr(run_mode, "warmup_context", None)
    if warmup_context is None:
        return
    if not callable(warmup_context):
        raise TypeError("RunMode.warmup_context attribute must be callable.")
    warmup_context(
        context=context,
        spec=spec,
        scenario=scenario,
        adapter=adapter,
    )


def _coerce_warmup_sessions(value: object) -> tuple[WarmupSessionInputs, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(
            "Demo adapter create_model_warmup_sessions(...) must return a sequence "
            f"of WarmupSessionInputs, got {type(value).__name__}."
        )
    sessions: list[WarmupSessionInputs] = []
    for session in value:
        if not isinstance(session, WarmupSessionInputs):
            raise TypeError(
                "Demo adapter create_model_warmup_sessions(...) must return only "
                f"WarmupSessionInputs, got {type(session).__name__}."
            )
        sessions.append(session)
    return tuple(sessions)


__all__ = [
    "AdmissionPolicy",
    "AsyncSessionDriver",
    "BenchmarkErrorPolicy",
    "DefaultErrorPolicy",
    "DriverStatus",
    "ErrorAction",
    "ErrorPolicy",
    "InMemorySessionMetricsRecorder",
    "MetricsSnapshot",
    "Mp4ErrorPolicy",
    "LocalWindowErrorPolicy",
    "NoopTransportService",
    "NullErrorPolicy",
    "RunContext",
    "RunMode",
    "RunModeCapabilities",
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
    "WebRTCErrorPolicy",
    "build_model_warmup_plan",
    "warmup_run_context",
]
