# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metrics boundary for inference sessions."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping


@dataclass(frozen=True, kw_only=True, slots=True)
class RuntimeMetricSample:
    """One runtime metric sample.

    Timing samples should use seconds as their canonical unit.
    """

    __hash__ = None

    name: str
    value: float | int
    unit: str = "s"
    step_index: int | None = None
    category: str = "runtime"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("RuntimeMetricSample.name must be non-empty.")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("RuntimeMetricSample.value must be numeric.")
        if not math.isfinite(float(self.value)):
            raise ValueError("RuntimeMetricSample.value must be finite.")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("RuntimeMetricSample.step_index must be >= 0.")
        if not self.unit.strip():
            raise ValueError("RuntimeMetricSample.unit must be non-empty.")
        if self.category == "timing" and self.unit != "s":
            raise ValueError("Timing metric samples must use unit='s'.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


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


@runtime_checkable
class MetricsRecorder(Protocol):
    """Collector for runtime, session, and run metrics."""

    def record(self, sample: RuntimeMetricSample) -> None:
        """Record one metric sample."""
        ...

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one timing sample in seconds."""
        ...

    def record_step(
        self,
        *,
        request: object,
        user_window: object,
        inference_input: object,
        result: object,
        decision: object,
    ) -> None:
        """Record one successful model step."""
        ...

    def record_control(
        self,
        *,
        request: object,
        user_window: object,
        control: object,
    ) -> None:
        """Record one provider-authored control decision."""
        ...

    def record_error(self, exc: Exception, action: object) -> None:
        """Record a driver-observed operational error."""
        ...

    def record_catch_up(self, decision: object) -> None:
        """Record a realtime catch-up decision."""
        ...

    def record_cleanup_error(self, exc: Exception) -> None:
        """Record a cleanup failure without interrupting teardown."""
        ...

    def record_orphaned_cleanup(self, exc: Exception) -> None:
        """Record cleanup that timed out and is still queued."""
        ...

    def record_session(self, result: object) -> None:
        """Record one closed session result."""
        ...

    def record_session_error(self, exc: Exception) -> None:
        """Record diagnostic session assembly failure detail."""
        ...

    def close(self) -> MetricsSnapshot:
        """Finalize metric collection."""
        ...


@dataclass(slots=True)
class InMemoryMetricsRecorder:
    """Simple metrics recorder useful for tests, smoke runs, and adapters."""

    samples: list[RuntimeMetricSample] = field(default_factory=list)
    step_count: int = 0
    control_count: int = 0
    catch_up_count: int = 0
    errors: list[str] = field(default_factory=list)
    cleanup_errors: list[str] = field(default_factory=list)
    orphaned_cleanup_errors: list[str] = field(default_factory=list)
    session_errors: list[str] = field(default_factory=list)
    sessions: list[object] = field(default_factory=list)
    closed: bool = False

    def record(self, sample: RuntimeMetricSample) -> None:
        if self.closed:
            raise RuntimeError("Cannot record metrics after close().")
        self.samples.append(sample)

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.record(
            RuntimeMetricSample(
                name=name,
                value=duration_s,
                unit="s",
                step_index=step_index,
                category="timing",
                metadata={} if metadata is None else metadata,
            )
        )

    def record_step(
        self,
        *,
        request: object,
        user_window: object,
        inference_input: object,
        result: object,
        decision: object,
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

    def record_error(self, exc: Exception, action: object) -> None:
        del action
        if not self.closed:
            self.errors.append(str(exc))

    def record_catch_up(self, decision: object) -> None:
        del decision
        if not self.closed:
            self.catch_up_count += 1

    def record_cleanup_error(self, exc: Exception) -> None:
        if not self.closed:
            self.cleanup_errors.append(str(exc))

    def record_orphaned_cleanup(self, exc: Exception) -> None:
        if not self.closed:
            self.orphaned_cleanup_errors.append(str(exc))

    def record_session(self, result: object) -> None:
        if not self.closed:
            self.sessions.append(result)

    def record_session_error(self, exc: Exception) -> None:
        if not self.closed:
            self.session_errors.append(str(exc))

    def close(self) -> MetricsSnapshot:
        self.closed = True
        return self.snapshot()

    def snapshot(self) -> MetricsSnapshot:
        timings: defaultdict[str, list[float]] = defaultdict(list)
        for sample in self.samples:
            if sample.category == "timing":
                timings[sample.name].append(float(sample.value))
        session_statuses = tuple(
            str(getattr(result, "status", "unknown")) for result in self.sessions
        )
        session_status_counts = Counter(session_statuses)
        return MetricsSnapshot(
            counters={
                "samples": len(self.samples),
                "steps": self.step_count,
                "controls": self.control_count,
                "catch_ups": self.catch_up_count,
                "sessions": len(self.sessions),
                "errors": len(self.errors),
                "cleanup_errors": len(self.cleanup_errors),
                "orphaned_cleanup_errors": len(self.orphaned_cleanup_errors),
                "session_errors": len(self.session_errors),
                **{
                    f"sessions.{status}": count
                    for status, count in sorted(session_status_counts.items())
                },
            },
            timings=timings,
            session_statuses=session_statuses,
            errors=tuple(
                (
                    *self.errors,
                    *self.cleanup_errors,
                    *self.orphaned_cleanup_errors,
                    *self.session_errors,
                )
            ),
        )


class NullMetricsRecorder:
    """Metrics recorder that intentionally drops all samples."""

    def record(self, sample: RuntimeMetricSample) -> None:
        del sample

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del name, duration_s, step_index, metadata

    def record_step(
        self,
        *,
        request: object,
        user_window: object,
        inference_input: object,
        result: object,
        decision: object,
    ) -> None:
        del request, user_window, inference_input, result, decision

    def record_control(
        self,
        *,
        request: object,
        user_window: object,
        control: object,
    ) -> None:
        del request, user_window, control

    def record_error(self, exc: Exception, action: object) -> None:
        del exc, action

    def record_catch_up(self, decision: object) -> None:
        del decision

    def record_cleanup_error(self, exc: Exception) -> None:
        del exc

    def record_orphaned_cleanup(self, exc: Exception) -> None:
        del exc

    def record_session(self, result: object) -> None:
        del result

    def record_session_error(self, exc: Exception) -> None:
        del exc

    def close(self) -> MetricsSnapshot:
        return MetricsSnapshot()


__all__ = [
    "InMemoryMetricsRecorder",
    "MetricsRecorder",
    "MetricsSnapshot",
    "NullMetricsRecorder",
    "RuntimeMetricSample",
]
