# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from flashdreams.runtime.metrics import (
    InMemoryMetricsRecorder,
    MetricsSnapshot,
    NullMetricsRecorder,
    RuntimeMetricSample,
)

pytestmark = pytest.mark.ci_cpu


def test_runtime_metric_sample_rejects_bool_values() -> None:
    with pytest.raises(TypeError, match="numeric"):
        RuntimeMetricSample(name="sample", value=True)


def test_in_memory_metrics_recorder_uses_seconds_for_timing() -> None:
    recorder = InMemoryMetricsRecorder()

    recorder.record_timing("model_step", 0.125, step_index=2)

    assert len(recorder.samples) == 1
    sample = recorder.samples[0]
    assert sample.name == "model_step"
    assert sample.value == pytest.approx(0.125)
    assert sample.unit == "s"
    assert sample.category == "timing"
    assert sample.step_index == 2
    snapshot = recorder.close()
    assert isinstance(snapshot, MetricsSnapshot)
    assert recorder.closed
    assert snapshot.counters["samples"] == 1
    assert snapshot.timings["model_step"] == (pytest.approx(0.125),)


def test_in_memory_metrics_recorder_rolls_up_sessions_and_diagnostics() -> None:
    recorder = InMemoryMetricsRecorder()

    recorder.record_session(SimpleNamespace(status="completed"))
    recorder.record_session_error(RuntimeError("assembly failed"))
    recorder.record_error(RuntimeError("step failed"), object())
    recorder.record_catch_up(object())
    recorder.record_cleanup_error(RuntimeError("cleanup failed"))
    recorder.record_orphaned_cleanup(RuntimeError("orphaned cleanup"))
    snapshot = recorder.close()

    assert snapshot.counters["sessions"] == 1
    assert snapshot.counters["sessions.completed"] == 1
    assert snapshot.counters.get("sessions.failed", 0) == 0
    assert snapshot.counters["session_errors"] == 1
    assert snapshot.counters["catch_ups"] == 1
    assert snapshot.session_statuses == ("completed",)
    assert snapshot.errors == (
        "step failed",
        "cleanup failed",
        "orphaned cleanup",
        "assembly failed",
    )


def test_cancelled_session_rollup_does_not_count_as_failed() -> None:
    recorder = InMemoryMetricsRecorder()

    recorder.record_session(SimpleNamespace(status="cancelled"))
    snapshot = recorder.close()

    assert snapshot.counters["sessions"] == 1
    assert snapshot.counters["sessions.cancelled"] == 1
    assert snapshot.counters.get("sessions.failed", 0) == 0
    assert snapshot.session_statuses == ("cancelled",)


def test_null_metrics_recorder_keeps_old_and_new_calls_noop() -> None:
    recorder = NullMetricsRecorder()

    recorder.record(RuntimeMetricSample(name="runtime", value=1.0))
    recorder.record_timing("model_step", 0.125, step_index=2)
    recorder.record_step(
        request=object(),
        user_window=object(),
        inference_input=object(),
        result=object(),
        decision=object(),
    )
    recorder.record_control(
        request=object(),
        user_window=object(),
        control=object(),
    )
    recorder.record_error(RuntimeError("step failed"), object())
    recorder.record_catch_up(object())
    recorder.record_cleanup_error(RuntimeError("cleanup failed"))
    recorder.record_orphaned_cleanup(RuntimeError("orphaned cleanup"))
    recorder.record_session(SimpleNamespace(status="failed"))
    recorder.record_session_error(RuntimeError("assembly failed"))
    snapshot = recorder.close()

    assert isinstance(snapshot, MetricsSnapshot)
    assert snapshot.counters == {}
    assert snapshot.timings == {}
    assert snapshot.session_statuses == ()
    assert snapshot.errors == ()


def test_timing_metric_samples_must_use_seconds() -> None:
    with pytest.raises(ValueError, match="unit='s'"):
        RuntimeMetricSample(
            name="model_step",
            value=12.5,
            unit="ms",
            category="timing",
        )
