# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

import pytest

from flashdreams.infra.acceleration.prewarm import (
    PrewarmDeadline,
    PrewarmTimeoutError,
    cuda_graph_prewarm_steps,
    is_warmup_index,
    run_prewarm_sequence,
    run_timed_prewarm,
)

pytestmark = pytest.mark.ci_cpu


def _clock(*times: float) -> Callable[[], float]:
    values = iter(times)
    return lambda: next(values)


def test_run_timed_prewarm_records_elapsed_without_retaining_result() -> None:
    calls: list[str] = []

    def step() -> object:
        calls.append("step")
        return object()

    timing = run_timed_prewarm(
        step,
        label="model",
        time_fn=_clock(10.0, 10.25),
    )

    assert calls == ["step"]
    assert timing.label == "model"
    assert timing.start_time == 10.0
    assert timing.end_time == 10.25
    assert timing.elapsed_ms == pytest.approx(250.0)


def test_run_timed_prewarm_reports_timeout_after_sync_step() -> None:
    with pytest.raises(PrewarmTimeoutError, match="slow"):
        run_timed_prewarm(
            lambda: None,
            label="slow",
            timeout_s=0.1,
            time_fn=_clock(0.0, 0.2),
        )


def test_prewarm_sequence_runs_cold_start_then_steady_steps() -> None:
    calls: list[str] = []

    timing = run_prewarm_sequence(
        cold_start=lambda: calls.append("cold"),
        steady_state=lambda: calls.append("steady"),
        steady_steps=3,
        label="compile",
        time_fn=_clock(0.0, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5),
    )

    assert calls == ["cold", "steady", "steady", "steady"]
    assert timing.label == "compile"
    assert [step.label for step in timing.steps] == [
        "compile.cold_start",
        "compile.steady_state.0",
        "compile.steady_state.1",
        "compile.steady_state.2",
    ]
    assert timing.elapsed_ms == pytest.approx(400.0)


def test_prewarm_sequence_checks_shared_deadline_between_steps() -> None:
    calls: list[str] = []

    with pytest.raises(PrewarmTimeoutError, match="compile"):
        run_prewarm_sequence(
            steady_state=lambda: calls.append("steady"),
            steady_steps=2,
            label="compile",
            timeout_s=0.25,
            time_fn=_clock(0.0, 0.05, 0.1, 0.2, 0.21, 0.22, 0.4),
        )

    assert calls == ["steady", "steady"]


def test_prewarm_deadline_reports_remaining_time() -> None:
    deadline = PrewarmDeadline.start(
        label="startup",
        timeout_s=1.0,
        time_fn=_clock(5.0),
    )

    assert deadline.elapsed_s(now=5.25) == pytest.approx(0.25)
    assert deadline.remaining_s(now=5.25) == pytest.approx(0.75)
    deadline.raise_if_expired(now=5.99)
    with pytest.raises(PrewarmTimeoutError, match="startup"):
        deadline.raise_if_expired(now=6.01)


def test_cuda_graph_prewarm_steps_counts_warmup_capture_and_replay() -> None:
    assert cuda_graph_prewarm_steps(warmup_iters=2) == 4
    assert (
        cuda_graph_prewarm_steps(
            warmup_iters=3,
            capture_steps=2,
            replay_steps=0,
        )
        == 5
    )

    with pytest.raises(ValueError, match="warmup_iters"):
        cuda_graph_prewarm_steps(warmup_iters=-1)


def test_is_warmup_index_matches_excluded_prefix() -> None:
    assert is_warmup_index(0)
    assert not is_warmup_index(1)
    assert is_warmup_index(2, warmup_count=3)
    assert not is_warmup_index(3, warmup_count=3)

    with pytest.raises(ValueError, match="index"):
        is_warmup_index(-1)
