# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading

import pytest

from flashdreams.runtime import (
    InferenceInput,
    InferenceSession,
    StepRequest,
    StepResult,
)
from flashdreams.runtime.demo import ModelWarmupPlan, RuntimeHost, WarmupSessionInputs

pytestmark = pytest.mark.ci_cpu


def test_runtime_host_latches_health_and_runs_lifecycle_in_order() -> None:
    runtime = _LifecycleRuntime()
    host = RuntimeHost(runtime)
    error = RuntimeError("runtime wedged")

    host.mark_unhealthy("first failure", error)
    host.mark_unhealthy("second failure")

    assert not host.is_healthy
    assert host.unhealthy_reason == "first failure"
    assert host.unhealthy_error is error

    initial_input = InferenceInput(global_conditioning={"session": "warmup"})
    step_inputs = (
        InferenceInput(step={"step": 0}),
        InferenceInput(step={"step": 1}),
    )
    host.preload()
    host.warmup(
        ModelWarmupPlan(
            sessions=(
                WarmupSessionInputs(
                    initial_input=initial_input,
                    step_inputs=step_inputs,
                ),
            ),
        )
    )
    host.close()
    host.close()

    assert runtime.events == [
        "initialize_distributed",
        "preload",
        ("start_session", initial_input),
        ("step", step_inputs[0]),
        ("step", step_inputs[1]),
        "session.close",
        "runtime.close",
        "close_distributed",
    ]
    assert not host.is_healthy
    with pytest.raises(RuntimeError, match="closed"):
        host.call(lambda: None)


@pytest.mark.asyncio
async def test_runtime_host_call_async_does_not_block_event_loop() -> None:
    runtime = _LifecycleRuntime()
    host = RuntimeHost(runtime)
    loop_thread_id = threading.get_ident()
    started = threading.Event()
    release = threading.Event()
    heartbeat_ticks = 0

    def _slow_model_call() -> int:
        started.set()
        assert release.wait(timeout=2.0)
        return threading.get_ident()

    async def _heartbeat_until_done(task: asyncio.Task[int]) -> None:
        nonlocal heartbeat_ticks
        while not task.done():
            heartbeat_ticks += 1
            await asyncio.sleep(0)

    try:
        model_task = asyncio.create_task(host.call_async(_slow_model_call))
        assert await asyncio.to_thread(started.wait, 1.0)
        heartbeat_task = asyncio.create_task(_heartbeat_until_done(model_task))
        for _ in range(5):
            await asyncio.sleep(0)
        assert heartbeat_ticks > 0

        release.set()
        worker_thread_id = await model_task
        await heartbeat_task
    finally:
        host.close()

    assert worker_thread_id != loop_thread_id


def test_runtime_host_reentrant_sync_and_async_dispatch_raise() -> None:
    host = RuntimeHost(_LifecycleRuntime())

    def _nested_sync_dispatch() -> None:
        host.call(lambda: None)

    def _nested_async_dispatch() -> None:
        async def _dispatch() -> None:
            await host.call_async(lambda: None)

        asyncio.run(_dispatch())

    try:
        with pytest.raises(RuntimeError, match="own thread"):
            host.call(_nested_sync_dispatch)
        with pytest.raises(RuntimeError, match="own thread"):
            host.call(_nested_async_dispatch)
    finally:
        host.close()


def test_non_control_rank_setup_returns_after_worker_loop_without_demo_edges() -> None:
    runtime = _LifecycleRuntime()
    host = RuntimeHost(
        runtime,
        is_control_rank=False,
        worker_loop=runtime.run_worker_loop,
    )
    constructed: list[str] = []

    result = _fake_run_setup(host, constructed)

    assert result == "worker-rank"
    assert constructed == []
    assert runtime.events == [
        "initialize_distributed",
        "preload",
        "run_worker_loop",
        "runtime.close",
        "close_distributed",
    ]


def test_control_rank_setup_reaches_demo_assembly() -> None:
    runtime = _LifecycleRuntime()
    host = RuntimeHost(runtime)
    constructed: list[str] = []

    try:
        result = _fake_run_setup(host, constructed)
    finally:
        host.close()

    assert result == "control-rank"
    assert constructed == ["run_mode", "provider", "input_source", "output_sink"]
    assert runtime.events[:2] == ["initialize_distributed", "preload"]
    assert "run_worker_loop" not in runtime.events


def _fake_run_setup(host: RuntimeHost, constructed: list[str]) -> str:
    host.preload()
    if not host.is_control_rank:
        host.run_worker_loop()
        host.close()
        return "worker-rank"

    constructed.extend(["run_mode", "provider", "input_source", "output_sink"])
    return "control-rank"


class _LifecycleRuntime:
    def __init__(self) -> None:
        self.events: list[object] = []

    def initialize_distributed(self) -> None:
        self.events.append("initialize_distributed")

    def preload(self) -> None:
        self.events.append("preload")

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self.events.append(("start_session", inputs))
        return _LifecycleSession(self.events)

    def run_worker_loop(self) -> None:
        self.events.append("run_worker_loop")

    def close(self) -> None:
        self.events.append("runtime.close")

    def close_distributed(self) -> None:
        self.events.append("close_distributed")


class _LifecycleSession:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self._next_step = 0

    def next_step_request(self) -> StepRequest | None:
        request = StepRequest(step_index=self._next_step)
        self._next_step += 1
        return request

    def step(self, inputs: InferenceInput) -> StepResult:
        self._events.append(("step", inputs))
        return StepResult(step_index=self._next_step, output=None)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._next_step = 0

    def close(self) -> None:
        self._events.append("session.close")
