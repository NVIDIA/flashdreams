# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for V2 chunk-lifecycle diagnostics."""

import logging
import threading

import pytest
import torch

from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.session_desc import BackpressureMode
from flashdreams.runtime_v2.session_runner import (
    _close_chunk_trace,
    _open_chunk_trace,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_LOGGER = "flashdreams.runtime_v2.chunk_trace"


def _result(step_index: int, frames: int = 1) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.zeros(frames, 3, 1, 1),
        frame_count=frames,
        output_layout=VideoTensorLayout.tchw,
    )


def test_opt_in_trace_correlates_publish_drop_and_present(caplog) -> None:
    manager = PresentationManager(device=torch.device("cpu"))
    manager.configure(
        backpressure_mode=BackpressureMode.DROP_OLDEST,
        stop=threading.Event(),
        put_timeout=0.01,
        trace_chunk_lifecycle=True,
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        manager.publish(3, [_result(10)])
        assert manager.advance(3)[0]
        manager.publish(3, [_result(11)])
        manager.publish(3, [_result(12)])
        assert manager.advance(3)[0]

    trace = "\n".join(
        record.getMessage() for record in caplog.records if record.name == _LOGGER
    )
    assert "phase=publish_started" in trace
    assert "phase=publish_completed" in trace
    assert "buffered_chunks=" in trace
    assert "chunk_capacity=" in trace
    assert "phase=chunk_dropped" in trace
    assert "step=11" in trace
    assert "reason=queue_full" in trace
    assert "replacement_step=12" in trace
    assert "phase=frame_presented" in trace
    assert "generation=3 step=10 frame=0" in trace
    assert "generation=3 step=12 frame=0" in trace


def test_trace_is_silent_by_default(caplog) -> None:
    manager = PresentationManager(device=torch.device("cpu"))

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        manager.publish(0, [_result(0, frames=1)])
        manager.advance(0)

    assert not [record for record in caplog.records if record.name == _LOGGER]


def test_trace_file_receives_records_and_closes(tmp_path) -> None:
    trace_path = tmp_path / "nested" / "input-trace.log"
    manager = PresentationManager(device=torch.device("cpu"))
    manager.configure(
        backpressure_mode=BackpressureMode.BLOCK,
        stop=threading.Event(),
        put_timeout=0.01,
        trace_chunk_lifecycle=True,
    )

    trace_log = _open_chunk_trace(trace_path)
    try:
        manager.publish(4, [_result(20, frames=1)])
        manager.advance(4)
    finally:
        _close_chunk_trace(trace_log)

    trace = trace_path.read_text(encoding="utf-8")
    assert "phase=publish_started" in trace
    assert "generation=4 step=20" in trace
    assert "phase=frame_presented" in trace
