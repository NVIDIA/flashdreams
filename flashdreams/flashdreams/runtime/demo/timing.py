# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime activation, clock, and input-window primitives for demo run modes."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from flashdreams.runtime.inputs import UserInputs, UserInputSchema
from flashdreams.runtime.types import StepRequirements

from .session_inputs import UserInputWindow

CatchUpPolicy = Literal["drop", "fold", "compress"]


@dataclass(frozen=True, kw_only=True, slots=True)
class CatchUpDecision:
    """How a realtime clock bounded stale virtual input time."""

    skipped_s: float = 0.0
    skipped_windows: int = 0
    input_policy: CatchUpPolicy | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.skipped_s) or self.skipped_s < 0.0:
            raise ValueError("CatchUpDecision.skipped_s must be finite and >= 0.")
        if self.skipped_windows < 0:
            raise ValueError("CatchUpDecision.skipped_windows must be >= 0.")
        if self.input_policy not in {None, "drop", "fold", "compress"}:
            raise ValueError(
                f"Unsupported catch-up input_policy={self.input_policy!r}."
            )
        if self.reason is not None and not self.reason.strip():
            raise ValueError("CatchUpDecision.reason must be non-empty when set.")


@dataclass(frozen=True, kw_only=True, slots=True)
class RealtimeWindowResult:
    """Realtime input window plus any catch-up decision that preceded it."""

    window: UserInputWindow
    catch_up: CatchUpDecision = field(default_factory=CatchUpDecision)


@dataclass(frozen=True, kw_only=True, slots=True)
class ActivationResult:
    """Result of waiting for a realtime activation gate."""

    activated: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.reason is not None and not self.reason.strip():
            raise ValueError("ActivationResult.reason must be non-empty when set.")


@runtime_checkable
class DeterministicClock(Protocol):
    """Clock facts for finite deterministic run modes."""

    is_realtime: bool
    is_deterministic: bool


@runtime_checkable
class RealtimeClock(Protocol):
    """Realtime virtual clock used by realtime drivers and input sources."""

    is_realtime: bool
    is_deterministic: bool

    def now(self) -> float: ...

    def anchor(self, wall_time_s: float) -> None: ...

    async def wait_until_window_end(self, end_s: float) -> None: ...

    async def apply_backpressure(self, requested_s: float) -> None: ...

    def catch_up(
        self,
        *,
        request: StepRequirements,
        max_lag_s: float,
        policy: CatchUpPolicy,
    ) -> CatchUpDecision: ...


@runtime_checkable
class ActivationPolicy(Protocol):
    """Wait until a realtime session should start generating."""

    timeout_s: float | None

    async def wait_until_active(
        self,
        clock: RealtimeClock | DeterministicClock,
    ) -> ActivationResult: ...


@runtime_checkable
class ActivationSignal(Protocol):
    """Event-like object accepted by ``SignalActivationPolicy``."""

    def is_set(self) -> bool: ...

    async def wait(self) -> object: ...


@dataclass(slots=True)
class AlwaysActiveActivationPolicy:
    """Activation policy for batch/null modes or already-ready realtime modes."""

    timeout_s: float | None = None
    anchor_clock: bool = False

    async def wait_until_active(
        self,
        clock: RealtimeClock | DeterministicClock,
    ) -> ActivationResult:
        _anchor_if_realtime(clock, anchor=self.anchor_clock)
        return ActivationResult(activated=True)


@dataclass(slots=True)
class SignalActivationPolicy:
    """Activate when any supplied signal fires, with optional timeout."""

    signals: Sequence[ActivationSignal]
    timeout_s: float | None = None
    timeout_reason: str = "activation timed out"
    anchor_clock: bool = True

    def __post_init__(self) -> None:
        if not self.signals:
            raise ValueError("SignalActivationPolicy.signals must be non-empty.")
        self.signals = tuple(self.signals)
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise ValueError("SignalActivationPolicy.timeout_s must be > 0 when set.")
        if not self.timeout_reason.strip():
            raise ValueError("SignalActivationPolicy.timeout_reason must be non-empty.")

    async def wait_until_active(
        self,
        clock: RealtimeClock | DeterministicClock,
    ) -> ActivationResult:
        if any(signal.is_set() for signal in self.signals):
            _anchor_if_realtime(clock, anchor=self.anchor_clock)
            return ActivationResult(activated=True)

        tasks = [asyncio.create_task(signal.wait()) for signal in self.signals]
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return ActivationResult(
                    activated=False,
                    reason=self.timeout_reason,
                )
            for task in done:
                task.result()
            _anchor_if_realtime(clock, anchor=self.anchor_clock)
            return ActivationResult(activated=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class _RealtimeTimeline(Protocol):
    dt: float
    next_chunk_start_v: float

    def reset(self, *, start_v: float) -> None: ...

    def sample_chunk(
        self,
        num_frames: int,
    ) -> Sequence[float]: ...


@dataclass(slots=True)
class RealtimeEventResampler:
    """Transport-neutral realtime window timeline.

    The shared driver only owns virtual time and frame sample locations. Raw
    browser/native events are stored as :class:`UserInputs`; model providers
    decide how to interpret them.
    """

    fps: float
    start_v: float = 0.0
    next_chunk_start_v: float = field(init=False)
    _dt: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        self._dt = 1.0 / float(self.fps)
        self.next_chunk_start_v = float(self.start_v)

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self, *, start_v: float) -> None:
        self.next_chunk_start_v = float(start_v)

    def sample_chunk(self, num_frames: int) -> tuple[float, ...]:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        chunk_start_v = self.next_chunk_start_v
        chunk_end_v = chunk_start_v + num_frames * self._dt
        frame_times = tuple(
            chunk_start_v + (index + 1) * self._dt for index in range(num_frames)
        )
        self.next_chunk_start_v = chunk_end_v
        return frame_times


@dataclass(slots=True)
class ResamplerRealtimeClock:
    """Realtime clock that reuses a resampler's virtual timeline."""

    resampler: _RealtimeTimeline
    now_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep
    is_realtime: bool = True
    is_deterministic: bool = False
    _pending_backpressure_s: float = field(default=0.0, init=False, repr=False)

    @property
    def pending_backpressure_s(self) -> float:
        return self._pending_backpressure_s

    def now(self) -> float:
        return float(self.now_fn())

    def anchor(self, wall_time_s: float) -> None:
        if not math.isfinite(wall_time_s):
            raise ValueError("wall_time_s must be finite.")
        self.resampler.next_chunk_start_v = float(wall_time_s)
        self._pending_backpressure_s = 0.0

    async def wait_until_window_end(self, end_s: float) -> None:
        if not math.isfinite(end_s):
            raise ValueError("end_s must be finite.")
        delay_s = float(end_s) - self.now()
        if delay_s > 0.0:
            await self.sleep_fn(delay_s)

    async def apply_backpressure(self, requested_s: float) -> None:
        if not math.isfinite(requested_s) or requested_s < 0.0:
            raise ValueError("requested_s must be finite and >= 0.")
        self._pending_backpressure_s += float(requested_s)

    def catch_up(
        self,
        *,
        request: StepRequirements,
        max_lag_s: float,
        policy: CatchUpPolicy,
    ) -> CatchUpDecision:
        if policy != "fold":
            raise NotImplementedError(
                f"Catch-up policy {policy!r} has no existing timeline analog yet."
            )
        if not math.isfinite(max_lag_s) or max_lag_s < 0.0:
            raise ValueError("max_lag_s must be finite and >= 0.")

        input_frame_count = input_frame_count_from_request(request)
        chunk_duration_s = input_frame_count * float(self.resampler.dt)
        if chunk_duration_s <= 0.0:
            raise ValueError("Realtime resampler dt must produce a positive window.")

        effective_now_s = self.now() + self._pending_backpressure_s
        self._pending_backpressure_s = 0.0
        current_start_s = float(self.resampler.next_chunk_start_v)
        lag_s = effective_now_s - (current_start_s + chunk_duration_s)
        if lag_s <= max_lag_s:
            return CatchUpDecision()

        latest_start_s = effective_now_s - chunk_duration_s
        if latest_start_s <= current_start_s:
            return CatchUpDecision()

        skipped_s = latest_start_s - current_start_s
        skipped_windows = max(1, math.ceil(skipped_s / chunk_duration_s))
        self.resampler.next_chunk_start_v = latest_start_s
        return CatchUpDecision(
            skipped_s=skipped_s,
            skipped_windows=skipped_windows,
            input_policy=policy,
            reason="lag exceeded max_lag_s",
        )


@dataclass(slots=True)
class RealtimeEventInputSource:
    """Realtime input source backed by raw event windows."""

    resampler: _RealtimeTimeline
    max_lag_s: float | None = None
    catch_up_policy: CatchUpPolicy = "fold"
    is_finite: bool = False
    is_deterministic: bool = False
    user_input_schema: UserInputSchema = field(default_factory=UserInputSchema)

    def __post_init__(self) -> None:
        if self.max_lag_s is not None and (
            not math.isfinite(self.max_lag_s) or self.max_lag_s < 0.0
        ):
            raise ValueError(
                "RealtimeEventInputSource.max_lag_s must be finite and >= 0."
            )
        if self.catch_up_policy != "fold":
            raise NotImplementedError(
                f"Catch-up policy {self.catch_up_policy!r} has no existing "
                "event-window analog yet."
            )

    def is_finished(self) -> bool:
        return False

    def reset(self, *, start_v: float) -> None:
        self.resampler.reset(start_v=start_v)

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: RealtimeClock,
    ) -> RealtimeWindowResult:
        input_frame_count = input_frame_count_from_request(request)
        chunk_duration_s = input_frame_count * self.resampler.dt
        window_end_s = self.resampler.next_chunk_start_v + chunk_duration_s
        await clock.wait_until_window_end(window_end_s)
        catch_up = clock.catch_up(
            request=request,
            max_lag_s=self.max_lag_s
            if self.max_lag_s is not None
            else chunk_duration_s,
            policy=self.catch_up_policy,
        )
        start_s = self.resampler.next_chunk_start_v
        frame_times = self.resampler.sample_chunk(input_frame_count)
        end_s = self.resampler.next_chunk_start_v
        window = UserInputWindow(
            start_s=start_s,
            end_s=end_s,
            frame_times=tuple(frame_times),
            inputs=UserInputs(),
        )
        return RealtimeWindowResult(window=window, catch_up=catch_up)


def input_frame_count_from_request(request: StepRequirements) -> int:
    """Return the positive input frame count declared by a step requirement."""

    value = request.input_frame_count
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("StepRequirements.input_frame_count must be an integer.")
    parsed = value
    if parsed <= 0:
        raise ValueError("StepRequirements.input_frame_count must be > 0.")
    return parsed


def _anchor_if_realtime(
    clock: RealtimeClock | DeterministicClock,
    *,
    anchor: bool,
) -> None:
    if not anchor or not getattr(clock, "is_realtime", False):
        return
    now = getattr(clock, "now", None)
    clock_anchor = getattr(clock, "anchor", None)
    if callable(now) and callable(clock_anchor):
        clock_anchor(float(now()))


__all__ = [
    "ActivationPolicy",
    "ActivationResult",
    "ActivationSignal",
    "AlwaysActiveActivationPolicy",
    "CatchUpDecision",
    "CatchUpPolicy",
    "DeterministicClock",
    "RealtimeEventInputSource",
    "RealtimeEventResampler",
    "RealtimeClock",
    "RealtimeWindowResult",
    "ResamplerRealtimeClock",
    "SignalActivationPolicy",
    "input_frame_count_from_request",
]
