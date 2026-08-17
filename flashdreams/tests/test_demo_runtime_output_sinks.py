# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.demo import LocalWindowOutputSink
from flashdreams.runtime import OutputArtifact, StepResult, TimeWindow
from flashdreams.runtime.demo import (
    BenchmarkStatsOutputSink,
    CompositeOutputSink,
    CompositeOutputSinkError,
    DemoSpec,
    IOFactoryOutputSpec,
    LocalWindowOutputSpec,
    Mp4OutputSink,
    Mp4OutputSpec,
    NullOutputSink,
    NullOutputSpec,
    OutputDecision,
    SessionInfo,
    WebRTCOutputSpec,
    build_benchmark_output_sink,
    build_output_sink,
    build_output_target,
)

pytestmark = pytest.mark.ci_cpu


def test_output_decision_validates_presentation_backpressure() -> None:
    assert OutputDecision(backpressure_s=0.1).backpressure_s == pytest.approx(0.1)
    with pytest.raises(ValueError, match="finite and >= 0"):
        OutputDecision(backpressure_s=-0.1)
    with pytest.raises(ValueError, match="finite and >= 0"):
        OutputDecision(backpressure_s=float("inf"))


def test_mp4_output_sink_writes_artifact_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    writer_calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
        layout: str,
        install_hint: str,
    ) -> Path:
        del install_hint
        writer_calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
            }
        )
        return path

    sink = Mp4OutputSink(
        output_path=tmp_path / "out.mp4",
        fps=24,
        writer=fake_writer,
        move_to_cpu=False,
    )
    sink.open(SessionInfo(output_layout="bvtchw", steady_output_frame_count=1))
    sink.begin_generation(0)

    decision = sink.write(
        StepResult.from_video_chunk(
            step_index=2,
            video_chunk=torch.zeros((1, 2, 3, 3, 4, 5)),
            layout="bvtchw",
            metrics={"model_step_s": 0.25},
            output_window=TimeWindow(start_s=1.0, end_s=2.0),
        )
    )
    artifacts = tuple(sink.close())
    second_close = tuple(sink.close())

    assert decision == OutputDecision()
    assert artifacts == second_close
    assert artifacts == (
        OutputArtifact(
            kind="video/mp4",
            uri=str(tmp_path / "out.mp4"),
            metadata={
                "fps": 24,
                "source_layout": "bvtchw",
                "shape": (1, 2, 3, 3, 4, 5),
                "stats_history": (
                    {
                        "step_index": 2,
                        "frames": 3,
                        "model_step_s": 0.25,
                        "output_start_s": 1.0,
                        "output_end_s": 2.0,
                    },
                ),
            },
        ),
    )
    assert writer_calls == [
        {
            "shape": (3, 4, 10, 3),
            "path": tmp_path / "out.mp4",
            "fps": 24,
            "layout": "thwc",
        }
    ]


def test_mp4_output_sink_closes_empty_collector_without_writing(
    tmp_path: Path,
) -> None:
    writer_called = False

    def fake_writer(*args: Any, **kwargs: Any) -> Path:
        nonlocal writer_called
        del args, kwargs
        writer_called = True
        return tmp_path / "out.mp4"

    sink = Mp4OutputSink(
        output_path=tmp_path / "out.mp4",
        fps=24,
        writer=fake_writer,
    )
    sink.open(SessionInfo(output_layout="tchw"))

    assert tuple(sink.close()) == ()
    assert tuple(sink.close()) == ()
    assert not writer_called


def test_output_sink_is_built_from_demo_spec(tmp_path: Path) -> None:
    def fake_writer(*args: Any, **kwargs: Any) -> Path:
        del args, kwargs
        return tmp_path / "demo.mp4"

    spec = DemoSpec(
        model_id="fake-demo",
        input_mode="replay",
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=12),
    )

    mp4_sink = build_output_sink(spec.output, mp4_writer=fake_writer)
    null_sink = build_output_sink(NullOutputSpec(store_results=True))

    assert isinstance(mp4_sink, Mp4OutputSink)
    assert mp4_sink.output_path == tmp_path / "demo.mp4"
    assert mp4_sink.fps == 12
    assert mp4_sink.writer is fake_writer
    assert isinstance(null_sink, NullOutputSink)
    assert null_sink.store_results
    with pytest.raises(ValueError, match="realtime transport sink"):
        build_output_sink(WebRTCOutputSpec())


def test_application_output_specs_have_explicit_dispatch_boundaries(
    tmp_path: Path,
) -> None:
    local_sink = build_output_sink(LocalWindowOutputSpec(title="Application", fps=24.0))
    unresolved_mp4_sink = build_output_sink(
        Mp4OutputSpec(
            path=tmp_path / "application.mp4",
            fps=None,
            output_layout=None,
        )
    )

    assert isinstance(local_sink, LocalWindowOutputSink)
    assert local_sink.title == "Application"
    assert local_sink.fps == 24.0
    assert isinstance(unresolved_mp4_sink, Mp4OutputSink)
    assert unresolved_mp4_sink.fps is None
    assert unresolved_mp4_sink.output_layout is None
    with pytest.raises(TypeError, match="IOFactory output cannot be built"):
        build_output_sink(IOFactoryOutputSpec())
    with pytest.raises(TypeError, match="does not create a replay OutputTarget"):
        build_output_target(LocalWindowOutputSpec())
    with pytest.raises(TypeError, match="does not create a replay OutputTarget"):
        build_output_target(IOFactoryOutputSpec())
    with pytest.raises(ValueError, match="requires explicit fps and output_layout"):
        build_output_target(
            Mp4OutputSpec(
                path=tmp_path / "application.mp4",
                fps=None,
                output_layout=None,
            )
        )


def test_sinks_do_not_retain_step_result_references(tmp_path: Path) -> None:
    mp4_sink = Mp4OutputSink(
        output_path=tmp_path / "out.mp4",
        fps=24,
        writer=lambda *args: tmp_path / "out.mp4",
        move_to_cpu=False,
    )
    mp4_sink.open(SessionInfo(output_layout="bvtchw", steady_output_frame_count=1))
    mp4_result = StepResult.from_video_chunk(
        step_index=0,
        video_chunk=torch.zeros((1, 1, 1, 3, 2, 2)),
        layout="bvtchw",
    )

    mp4_sink.write(mp4_result)

    null_sink = NullOutputSink(store_results=True)
    null_sink.open(SessionInfo())
    null_result = StepResult(
        step_index=1,
        output=object(),
        frame_count=2,
        metrics={"model_step_s": 0.1},
        metadata={"source": "fake"},
    )

    null_sink.write(null_result)

    assert not _object_graph_contains(mp4_sink, mp4_result)
    assert not _object_graph_contains(null_sink, null_result)
    assert null_sink.results == [
        {
            "step_index": 1,
            "frame_count": 2,
            "metrics": {"model_step_s": 0.1},
            "metadata": {"source": "fake"},
        }
    ]


def test_null_output_sink_records_steps_without_artifacts() -> None:
    sink = NullOutputSink(store_results=True)
    sink.open(SessionInfo(output_layout="fake-video", steady_output_frame_count=1))

    sink.write(StepResult(step_index=0, output="first", frame_count=1))
    sink.write(StepResult(step_index=1, output="second", frame_count=2))
    artifacts = tuple(sink.close())

    assert artifacts == ()
    assert tuple(sink.close()) == ()
    assert sink.output_count == 2
    assert sink.results == [
        {"step_index": 0, "frame_count": 1, "metrics": {}, "metadata": {}},
        {"step_index": 1, "frame_count": 2, "metrics": {}, "metadata": {}},
    ]


def test_benchmark_stats_output_sink_writes_runtime_metric_samples(
    tmp_path: Path,
) -> None:
    sink = BenchmarkStatsOutputSink(output_path=tmp_path / "stats" / "run.json")
    sink.open(
        SessionInfo(
            output_layout="bvtchw",
            steady_output_frame_count=3,
            metadata={"scenario_id": "fake"},
        )
    )
    sink.begin_generation(0)

    decision = sink.write(
        StepResult.from_video_chunk(
            step_index=2,
            video_chunk=torch.zeros((1, 1, 2, 3, 4, 5)),
            layout="bvtchw",
            output_window=TimeWindow(start_s=1.0, end_s=1.5),
            metadata={"provider": "fake"},
            metrics={
                "model_step_s": 0.25,
                "encode_ms": 12.0,
                "pixel_fps": 30.0,
            },
        )
    )
    artifacts = tuple(sink.close())
    second_close = tuple(sink.close())

    assert decision == OutputDecision()
    assert artifacts == second_close
    assert artifacts == (
        OutputArtifact(
            kind="application/json",
            uri=str(tmp_path / "stats" / "run.json"),
            metadata={
                "artifact_type": "benchmark_stats",
                "schema_version": 1,
                "step_count": 1,
                "sample_count": 3,
            },
        ),
    )

    payload = json.loads((tmp_path / "stats" / "run.json").read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "flashdreams.runtime.demo.benchmark_stats"
    assert payload["session"] == {
        "output_layout": "bvtchw",
        "steady_output_frame_count": 3,
        "metadata": {"scenario_id": "fake"},
    }
    assert payload["steps"] == [
        {
            "frame_count": 2,
            "layout": "bvtchw",
            "metadata": {"provider": "fake"},
            "metrics": {
                "encode_ms": 12.0,
                "model_step_s": 0.25,
                "pixel_fps": 30.0,
            },
            "output_window": [1.0, 1.5],
            "sample_count": 3,
            "step_index": 2,
        }
    ]
    assert payload["samples"] == [
        {
            "category": "timing",
            "metadata": {
                "frame_count": 2,
                "layout": "bvtchw",
                "output_window": {"end_s": 1.5, "start_s": 1.0},
                "result_metadata": {"provider": "fake"},
            },
            "name": "model_step_s",
            "step_index": 2,
            "unit": "s",
            "value": 0.25,
        },
        {
            "category": "timing",
            "metadata": {
                "frame_count": 2,
                "layout": "bvtchw",
                "output_window": {"end_s": 1.5, "start_s": 1.0},
                "result_metadata": {"provider": "fake"},
            },
            "name": "encode_s",
            "step_index": 2,
            "unit": "s",
            "value": 0.012,
        },
        {
            "category": "throughput",
            "metadata": {
                "frame_count": 2,
                "layout": "bvtchw",
                "output_window": {"end_s": 1.5, "start_s": 1.0},
                "result_metadata": {"provider": "fake"},
            },
            "name": "pixel_fps",
            "step_index": 2,
            "unit": "fps",
            "value": 30.0,
        },
    ]


def test_benchmark_output_sink_supports_stats_only(tmp_path: Path) -> None:
    sink = build_benchmark_output_sink(None, stats_path=tmp_path / "stats.json")
    sink.open(SessionInfo())

    sink.write(StepResult(step_index=0, frame_count=1, metrics={"model_step_s": 0.1}))
    artifacts = tuple(sink.close())

    assert isinstance(sink, BenchmarkStatsOutputSink)
    assert sink.produces_artifacts
    assert artifacts[0].uri == str(tmp_path / "stats.json")


def test_benchmark_output_sink_composes_with_null_output(tmp_path: Path) -> None:
    sink = build_benchmark_output_sink(
        NullOutputSpec(store_results=True),
        stats_path=tmp_path / "stats.json",
    )
    sink.open(SessionInfo())

    sink.write(StepResult(step_index=0, frame_count=1, metrics={"model_step_s": 0.1}))
    artifacts = tuple(sink.close())

    assert isinstance(sink, CompositeOutputSink)
    assert sink.produces_artifacts
    assert isinstance(sink.sinks[0], NullOutputSink)
    assert sink.sinks[0].results == [
        {
            "frame_count": 1,
            "metadata": {},
            "metrics": {"model_step_s": 0.1},
            "step_index": 0,
        }
    ]
    assert artifacts == (
        OutputArtifact(
            kind="application/json",
            uri=str(tmp_path / "stats.json"),
            metadata={
                "artifact_type": "benchmark_stats",
                "schema_version": 1,
                "step_count": 1,
                "sample_count": 1,
            },
        ),
    )


def test_benchmark_output_sink_composes_with_mp4_output(tmp_path: Path) -> None:
    writer_calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
        layout: str,
        install_hint: str,
    ) -> Path:
        del install_hint
        writer_calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
            }
        )
        return path

    sink = build_benchmark_output_sink(
        Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=12),
        stats_path=tmp_path / "stats.json",
        mp4_writer=fake_writer,
    )
    sink.open(SessionInfo(output_layout="bvtchw"))

    sink.write(
        StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((1, 1, 2, 3, 4, 5)),
            layout="bvtchw",
            metrics={"model_step_s": 0.1},
        )
    )
    artifacts = tuple(sink.close())

    assert isinstance(sink, CompositeOutputSink)
    assert isinstance(sink.sinks[0], Mp4OutputSink)
    assert artifacts == (
        OutputArtifact(
            kind="video/mp4",
            uri=str(tmp_path / "demo.mp4"),
            metadata={
                "fps": 12,
                "source_layout": "bvtchw",
                "shape": (1, 1, 2, 3, 4, 5),
                "stats_history": (
                    {
                        "frames": 2,
                        "model_step_s": 0.1,
                        "step_index": 0,
                    },
                ),
            },
        ),
        OutputArtifact(
            kind="application/json",
            uri=str(tmp_path / "stats.json"),
            metadata={
                "artifact_type": "benchmark_stats",
                "schema_version": 1,
                "step_count": 1,
                "sample_count": 1,
            },
        ),
    )
    assert writer_calls == [
        {
            "shape": (2, 4, 5, 3),
            "path": tmp_path / "demo.mp4",
            "fps": 12,
            "layout": "thwc",
        }
    ]


def test_composite_output_sink_closes_siblings_after_close_failure(
    tmp_path: Path,
) -> None:
    failing_sink = _RecordingOutputSink(close_error=RuntimeError("video encode failed"))
    stats_path = tmp_path / "stats.json"
    stats_sink = BenchmarkStatsOutputSink(output_path=stats_path)
    sink = CompositeOutputSink((failing_sink, stats_sink))
    sink.open(SessionInfo())

    sink.write(StepResult(step_index=0, frame_count=1, metrics={"model_step_s": 0.25}))
    with pytest.raises(CompositeOutputSinkError, match="close failed") as exc_info:
        sink.close()

    assert len(exc_info.value.errors) == 1
    assert failing_sink.close_count == 1
    assert stats_path.exists()
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["steps"][0]["step_index"] == 0
    assert tuple(sink.close()) == (
        OutputArtifact(
            kind="application/json",
            uri=str(stats_path),
            metadata={
                "artifact_type": "benchmark_stats",
                "schema_version": 1,
                "step_count": 1,
                "sample_count": 1,
            },
        ),
    )


def test_composite_output_sink_closes_opened_sinks_after_open_failure() -> None:
    first_sink = _RecordingOutputSink()
    failing_sink = _RecordingOutputSink(open_error=RuntimeError("open failed"))
    last_sink = _RecordingOutputSink()
    sink = CompositeOutputSink((first_sink, failing_sink, last_sink))

    with pytest.raises(CompositeOutputSinkError, match="open failed") as exc_info:
        sink.open(SessionInfo())

    assert len(exc_info.value.errors) == 1
    assert first_sink.open_count == 1
    assert first_sink.close_count == 1
    assert failing_sink.open_count == 1
    assert failing_sink.close_count == 0
    assert last_sink.open_count == 1
    assert last_sink.close_count == 1


class _RecordingOutputSink:
    produces_artifacts = False

    def __init__(
        self,
        *,
        open_error: Exception | None = None,
        close_error: Exception | None = None,
        artifacts: Sequence[OutputArtifact] = (),
    ) -> None:
        self.open_error = open_error
        self.close_error = close_error
        self.artifacts = tuple(artifacts)
        self.open_count = 0
        self.write_count = 0
        self.close_count = 0

    def open(self, session_info: SessionInfo) -> None:
        del session_info
        self.open_count += 1
        if self.open_error is not None:
            raise self.open_error

    def begin_generation(self, generation: int) -> None:
        del generation

    def write(self, result: StepResult) -> OutputDecision:
        del result
        self.write_count += 1
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error
        return self.artifacts


def _object_graph_contains(
    root: object,
    needle: object,
    *,
    seen: set[int] | None = None,
) -> bool:
    if root is needle:
        return True
    if seen is None:
        seen = set()
    root_id = id(root)
    if root_id in seen:
        return False
    seen.add(root_id)
    if root is None or isinstance(root, str | bytes | int | float | bool | Path):
        return False
    if isinstance(root, torch.Tensor):
        return False
    if callable(root):
        return False
    if isinstance(root, Mapping):
        return any(
            _object_graph_contains(key, needle, seen=seen)
            or _object_graph_contains(value, needle, seen=seen)
            for key, value in root.items()
        )
    if isinstance(root, list | tuple | set | frozenset):
        return any(_object_graph_contains(value, needle, seen=seen) for value in root)
    if is_dataclass(root):
        return any(
            _object_graph_contains(getattr(root, field.name), needle, seen=seen)
            for field in fields(root)
        )
    return False
