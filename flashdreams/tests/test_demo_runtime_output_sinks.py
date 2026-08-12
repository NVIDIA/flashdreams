# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.runtime import OutputArtifact, StepResult, TimeWindow
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSink,
    Mp4OutputSpec,
    NullOutputSink,
    NullOutputSpec,
    OutputDecision,
    SessionInfo,
    WebRTCOutputSpec,
    build_output_sink,
)

pytestmark = pytest.mark.ci_cpu


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
