# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for StepResult construction and readiness metadata."""

import inspect
from dataclasses import replace as dataclass_replace
from typing import Any

import pytest
import torch

from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _cpu_result(**kwargs: Any) -> StepResult:
    return StepResult(
        step_index=0,
        output=torch.zeros((1, 3, 2, 2)),
        frame_count=1,
        output_layout=VideoTensorLayout.tchw,
        **kwargs,
    )


def test_constructor_exposes_only_the_public_event_parameter() -> None:
    parameters = inspect.signature(StepResult).parameters

    assert "output_ready_event" in parameters
    assert "_output_ready_event" not in parameters
    assert parameters["output_ready_event"].annotation == torch.cuda.Event | None
    assert not hasattr(_cpu_result(), "output_ready_event")


def test_cpu_output_does_not_synthesize_a_cuda_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("CPU StepResult construction created a CUDA event")

    monkeypatch.setattr(torch.cuda, "Event", fail_if_called)

    assert _cpu_result()._output_ready_event is None
    assert _cpu_result(output_ready_event=None)._output_ready_event is None


def test_default_metrics_are_not_shared() -> None:
    first = _cpu_result()
    second = _cpu_result()

    first.metrics["latency_ms"] = 1.0

    assert second.metrics == {}


def test_safe_replace_preserves_readiness_metadata() -> None:
    result = _cpu_result()

    with pytest.raises(ValueError, match="InitVar 'output_ready_event'"):
        dataclass_replace(result, step_index=1)

    replaced = result.replace(step_index=1)

    assert replaced.step_index == 1
    assert replaced._output_ready_event is result._output_ready_event


def test_rejects_invalid_output_event_type() -> None:
    with pytest.raises(TypeError, match="must be a CUDA event or None"):
        _cpu_result(output_ready_event=object())
