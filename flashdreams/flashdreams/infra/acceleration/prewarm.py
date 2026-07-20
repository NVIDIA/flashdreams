# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model/compile prewarm helpers for realtime inference paths."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class PrewarmTimeoutError(TimeoutError):
    """Raised when a synchronous prewarm step exceeds its configured deadline."""

    def __init__(self, label: str, *, timeout_s: float, elapsed_s: float) -> None:
        self.label = label
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s
        super().__init__(
            f"Prewarm step '{label}' exceeded timeout_s={timeout_s:.3f} "
            f"(elapsed_s={elapsed_s:.3f})."
        )


@dataclass(frozen=True)
class PrewarmTiming:
    """Wall-clock timing for one prewarm step."""

    label: str
    start_time: float
    end_time: float

    @property
    def elapsed_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0


@dataclass(frozen=True)
class PrewarmSequenceTiming:
    """Wall-clock timings for a cold-start step followed by steady steps."""

    label: str
    cold_start: PrewarmTiming | None
    steady_state: tuple[PrewarmTiming, ...]

    @property
    def steps(self) -> tuple[PrewarmTiming, ...]:
        if self.cold_start is None:
            return self.steady_state
        return (self.cold_start, *self.steady_state)

    @property
    def elapsed_s(self) -> float:
        steps = self.steps
        if not steps:
            return 0.0
        return steps[-1].end_time - steps[0].start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0


@dataclass(frozen=True)
class PrewarmDeadline:
    """Shared deadline for a multi-step prewarm sequence.

    Synchronous prewarm callbacks cannot be interrupted safely, so deadlines are
    checked before and after each callback.
    """

    label: str
    start_time: float
    timeout_s: float | None = None

    @classmethod
    def start(
        cls,
        *,
        label: str = "prewarm",
        timeout_s: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> "PrewarmDeadline":
        _validate_timeout(timeout_s)
        return cls(label=label, start_time=time_fn(), timeout_s=timeout_s)

    def elapsed_s(
        self,
        *,
        now: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> float:
        if now is None:
            now = time_fn()
        return now - self.start_time

    def remaining_s(
        self,
        *,
        now: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> float | None:
        if self.timeout_s is None:
            return None
        return max(0.0, self.timeout_s - self.elapsed_s(now=now, time_fn=time_fn))

    def raise_if_expired(
        self,
        *,
        now: float | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        if self.timeout_s is None:
            return
        elapsed_s = self.elapsed_s(now=now, time_fn=time_fn)
        if elapsed_s > self.timeout_s:
            raise PrewarmTimeoutError(
                self.label,
                timeout_s=self.timeout_s,
                elapsed_s=elapsed_s,
            )


def run_timed_prewarm(
    step: Callable[[], Any],
    *,
    label: str = "prewarm",
    timeout_s: float | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> PrewarmTiming:
    """Run one synchronous prewarm step and return its wall-clock timing."""
    _validate_timeout(timeout_s)
    start_time = time_fn()
    result = step()
    del result
    end_time = time_fn()
    timing = PrewarmTiming(label=label, start_time=start_time, end_time=end_time)
    _raise_if_timeout_expired(label, timeout_s, timing.elapsed_s)
    return timing


def run_prewarm_sequence(
    *,
    steady_state: Callable[[], Any],
    steady_steps: int,
    cold_start: Callable[[], Any] | None = None,
    label: str = "prewarm",
    timeout_s: float | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> PrewarmSequenceTiming:
    """Run cold-start and steady-state prewarm callbacks under one deadline."""
    if steady_steps < 0:
        raise ValueError(f"steady_steps must be non-negative, got {steady_steps}.")

    deadline = PrewarmDeadline.start(
        label=label,
        timeout_s=timeout_s,
        time_fn=time_fn,
    )
    cold_timing = None
    if cold_start is not None:
        cold_timing = _run_deadlined_step(
            cold_start,
            label=f"{label}.cold_start",
            deadline=deadline,
            time_fn=time_fn,
        )

    steady_timings = tuple(
        _run_deadlined_step(
            steady_state,
            label=f"{label}.steady_state.{step_index}",
            deadline=deadline,
            time_fn=time_fn,
        )
        for step_index in range(steady_steps)
    )
    return PrewarmSequenceTiming(
        label=label,
        cold_start=cold_timing,
        steady_state=steady_timings,
    )


def cuda_graph_prewarm_steps(
    *,
    warmup_iters: int,
    capture_steps: int = 1,
    replay_steps: int = 1,
) -> int:
    """Return steady-state calls needed after cache saturation for graph replay."""
    _validate_non_negative("warmup_iters", warmup_iters)
    _validate_non_negative("capture_steps", capture_steps)
    _validate_non_negative("replay_steps", replay_steps)
    return warmup_iters + capture_steps + replay_steps


def is_warmup_index(index: int, *, warmup_count: int = 1) -> bool:
    """Return whether ``index`` belongs to an excluded warmup prefix."""
    _validate_non_negative("index", index)
    _validate_non_negative("warmup_count", warmup_count)
    return index < warmup_count


def _run_deadlined_step(
    step: Callable[[], Any],
    *,
    label: str,
    deadline: PrewarmDeadline,
    time_fn: Callable[[], float],
) -> PrewarmTiming:
    deadline.raise_if_expired(time_fn=time_fn)
    start_time = time_fn()
    result = step()
    del result
    end_time = time_fn()
    deadline.raise_if_expired(now=end_time, time_fn=time_fn)
    return PrewarmTiming(label=label, start_time=start_time, end_time=end_time)


def _raise_if_timeout_expired(
    label: str,
    timeout_s: float | None,
    elapsed_s: float,
) -> None:
    if timeout_s is not None and elapsed_s > timeout_s:
        raise PrewarmTimeoutError(label, timeout_s=timeout_s, elapsed_s=elapsed_s)


def _validate_timeout(timeout_s: float | None) -> None:
    if timeout_s is None:
        return
    if timeout_s < 0.0:
        raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
