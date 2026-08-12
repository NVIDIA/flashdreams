# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from flashdreams.runtime import StepRequirements, UserInputSchema
from flashdreams.runtime.demo import NullOutputSink, RunResult, SessionEdges
from flashdreams.runtime.demo.timing import (
    CatchUpDecision,
    CatchUpPolicy,
    RealtimeEventInputSource,
    RealtimeEventResampler,
    ResamplerRealtimeClock,
    SignalActivationPolicy,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.asyncio
async def test_signal_activation_waits_for_first_input_and_anchors_clock() -> None:
    event = asyncio.Event()
    resampler = RealtimeEventResampler(fps=30.0, start_v=0.0)
    clock = ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 12.0)
    policy = SignalActivationPolicy(signals=(event,), timeout_s=1.0)

    wait_task = asyncio.create_task(policy.wait_until_active(clock))
    await asyncio.sleep(0)

    assert not wait_task.done()

    event.set()
    result = await wait_task

    assert result.activated
    assert result.reason is None
    assert resampler.next_chunk_start_v == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_activation_timeout_can_close_edges_as_not_activated() -> None:
    event = asyncio.Event()
    resampler = RealtimeEventResampler(fps=30.0, start_v=0.0)
    clock = ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 12.0)
    policy = SignalActivationPolicy(
        signals=(event,),
        timeout_s=0.001,
        timeout_reason="no first input",
    )
    cleanup_tasks: set[asyncio.Task[RunResult]] = set()
    edges = SessionEdges(
        input_source=_OpenRealtimeInputSource(),
        output_sink=NullOutputSink(),
        cleanup_tasks=cleanup_tasks,
        activation=policy,
        clock=clock,
    )

    activation = await policy.wait_until_active(clock)
    result = edges.close_result(
        status="not_activated",
        reason=activation.reason,
    )

    assert not activation.activated
    assert activation.reason == "no first input"
    assert result.status == "not_activated"
    assert result.reason == "no first input"
    assert edges.is_closed
    assert resampler.next_chunk_start_v == pytest.approx(0.0)


def test_resampler_clock_catch_up_bounds_latency() -> None:
    resampler = RealtimeEventResampler(fps=1.0, start_v=0.0)
    clock = ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 5.0)

    decision = clock.catch_up(
        request=_request(input_frame_count=1),
        max_lag_s=1.0,
        policy="fold",
    )

    assert decision == CatchUpDecision(
        skipped_s=4.0,
        skipped_windows=4,
        input_policy="fold",
        reason="lag exceeded max_lag_s",
    )
    assert resampler.next_chunk_start_v == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_realtime_input_source_emits_transport_neutral_window() -> None:
    source_resampler = RealtimeEventResampler(fps=2.0, start_v=0.0)
    sleep = _RecordingSleep()
    clock = ResamplerRealtimeClock(
        resampler=source_resampler,
        now_fn=lambda: 3.0,
        sleep_fn=sleep,
    )
    source = RealtimeEventInputSource(resampler=source_resampler)

    result = await source.next_realtime_window(
        request=_request(input_frame_count=2),
        clock=clock,
    )

    assert sleep.delays == []
    assert result.catch_up == CatchUpDecision(
        skipped_s=2.0,
        skipped_windows=2,
        input_policy="fold",
        reason="lag exceeded max_lag_s",
    )
    assert result.window.start_s == pytest.approx(2.0)
    assert result.window.end_s == pytest.approx(3.0)
    assert result.window.frame_times == pytest.approx((2.5, 3.0))
    assert result.window.inputs.events == ()
    assert result.window.metadata == {}


@pytest.mark.asyncio
async def test_backpressure_is_clock_adjustment_not_blocking_sleep() -> None:
    resampler = RealtimeEventResampler(fps=1.0, start_v=0.0)
    sleep = _RecordingSleep()
    clock = ResamplerRealtimeClock(
        resampler=resampler,
        now_fn=lambda: 2.2,
        sleep_fn=sleep,
    )

    await clock.apply_backpressure(0.3)
    decision = clock.catch_up(
        request=_request(input_frame_count=1),
        max_lag_s=1.0,
        policy="fold",
    )

    assert sleep.delays == []
    assert clock.pending_backpressure_s == pytest.approx(0.0)
    assert decision.skipped_s == pytest.approx(1.5)
    assert decision.skipped_windows == 2
    assert decision.input_policy == "fold"
    assert resampler.next_chunk_start_v == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_window_floor_sleeps_only_when_virtual_time_is_ahead() -> None:
    resampler = RealtimeEventResampler(fps=1.0, start_v=0.0)
    sleep = _RecordingSleep()
    clock = ResamplerRealtimeClock(
        resampler=resampler,
        now_fn=lambda: 1.0,
        sleep_fn=sleep,
    )

    await clock.wait_until_window_end(1.25)
    await clock.wait_until_window_end(0.75)

    assert sleep.delays == [0.25]


@pytest.mark.parametrize("policy", ["drop", "compress"])
def test_realtime_event_source_defers_unsupported_catch_up_policies(
    policy: str,
) -> None:
    resampler = RealtimeEventResampler(fps=1.0, start_v=0.0)
    clock = ResamplerRealtimeClock(resampler=resampler, now_fn=lambda: 5.0)
    unsupported_policy = cast(CatchUpPolicy, policy)

    with pytest.raises(NotImplementedError, match="no existing timeline analog"):
        clock.catch_up(
            request=_request(input_frame_count=1),
            max_lag_s=1.0,
            policy=unsupported_policy,
        )

    with pytest.raises(NotImplementedError, match="event-window analog"):
        RealtimeEventInputSource(
            resampler=resampler,
            catch_up_policy=unsupported_policy,
        )


def _request(*, input_frame_count: int) -> StepRequirements:
    return StepRequirements(
        step_index=0,
        input_frame_count=input_frame_count,
    )


class _RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay_s: float) -> None:
        self.delays.append(delay_s)


class _OpenRealtimeInputSource:
    is_finite = False
    is_deterministic = False
    user_input_schema = UserInputSchema()

    def is_finished(self) -> bool:
        return False

    async def next_realtime_window(
        self,
        *,
        request: StepRequirements,
        clock: Any,
    ) -> object:
        del request, clock
        raise AssertionError("Activation timeout must close before requesting input.")
