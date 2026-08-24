# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the sink recording what a run measured.

What matters is that a benchmark can read what is written: the file says what
artifact it is, every step is accounted for, and a measurement means the same
thing whichever model reported it. What the reader expects of the file is
tested against ``tools.benchmarks.metrics`` itself.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_WIDTH = 128
"""Frame width the results describe."""

_HEIGHT = 64
"""Frame height they describe. Not square, so a transposed record shows up."""

_FPS = 16
"""Rate the frames are meant to play at."""

_FRAMES = 12
"""Frames a step generates."""


def _session_desc() -> SessionDesc:
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=_FPS,
        video_width=_WIDTH,
        video_height=_HEIGHT,
    )


def _result(
    step_index: int = 0,
    metrics: dict[str, float | int] | None = None,
    *,
    frames: int = _FRAMES,
) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.zeros((frames, 3, _HEIGHT, _WIDTH)),
        frame_count=frames,
        output_layout=VideoTensorLayout.tchw,
        metrics={} if metrics is None else metrics,
    )


def _write_run(path: Path, results: list[StepResult]) -> dict[str, Any]:
    """Run a whole session's worth of results through a sink and read it back."""
    sink = MetricsOutputSink(path)
    sink.open(_session_desc())
    for result in results:
        sink.write(result)
    sink.close()
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


## What a benchmark reads


def test_a_run_is_recorded_as_the_artifact_the_benchmark_looks_for(
    tmp_path: Path,
) -> None:
    payload = _write_run(tmp_path / "stats_run.json", [_result(0, {"total_ms": 25.0})])

    assert payload["artifact_type"] == "flashdreams.runtime.demo.benchmark_stats"
    assert payload["schema_version"] == 1


def test_every_step_is_recorded_with_the_video_it_generated(tmp_path: Path) -> None:
    """A report divides one by the other to say how fast a model generated video."""
    payload = _write_run(
        tmp_path / "stats_run.json",
        [
            _result(0, {"total_ms": 30.0}, frames=9),
            _result(1, {"total_ms": 20.0}, frames=12),
        ],
    )

    assert payload["steps"] == [
        {"step_index": 0, "frame_count": 9, "sample_count": 1},
        {"step_index": 1, "frame_count": 12, "sample_count": 1},
    ]


def test_what_was_being_generated_is_recorded_alongside_the_timings(
    tmp_path: Path,
) -> None:
    # A step takes as long as its frames are large, so a timing alone says
    # nothing.
    payload = _write_run(tmp_path / "stats_run.json", [_result()])

    assert payload["session"] == {
        "output_layout": "tchw",
        "frames_per_second_for_step": _FPS,
        "video_width": _WIDTH,
        "video_height": _HEIGHT,
    }


## What a measurement means


def test_a_measurement_in_milliseconds_is_recorded_in_seconds(tmp_path: Path) -> None:
    """Two runs measured in different units cannot be compared, so there is one."""
    payload = _write_run(
        tmp_path / "stats_run.json", [_result(0, {"total_ms": 1500.0})]
    )

    assert payload["samples"] == [
        {
            "name": "total_s",
            "value": 1.5,
            "unit": "s",
            "category": "timing",
            "step_index": 0,
            "metadata": {"frame_count": _FRAMES},
        }
    ]


@pytest.mark.parametrize(
    "name,unit,category",
    [
        ("generated_fps", "fps", "throughput"),
        # An unrecognized name is still recorded: it was measured for a reason.
        ("guidance_scale", "value", "runtime"),
    ],
)
def test_a_measurement_is_labelled_with_what_its_name_says_it_is(
    tmp_path: Path, name: str, unit: str, category: str
) -> None:
    """One suffix the reader knows and one it does not. The rest are a lookup
    table, covered by the reader's own tests."""
    payload = _write_run(tmp_path / "stats_run.json", [_result(0, {name: 4.0})])

    sample = payload["samples"][0]
    assert (sample["name"], sample["unit"], sample["category"]) == (
        name,
        unit,
        category,
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {"total_ms": math.nan},
        {"finished": True},
        {"   ": 1.0},
    ],
)
def test_a_measurement_a_report_could_not_average_is_left_out(
    tmp_path: Path, metrics: dict[str, float | int]
) -> None:
    """A pipeline reporting one of these is reporting something other than a
    measurement, and a report reading it would carry the nonsense forward."""
    payload = _write_run(tmp_path / "stats_run.json", [_result(0, metrics)])

    assert payload["samples"] == []
    assert payload["steps"] == [
        {"step_index": 0, "frame_count": _FRAMES, "sample_count": 0}
    ]


## Writing the file


def test_the_file_is_written_where_it_was_asked_for(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "today" / "stats_run.json"

    _write_run(path, [_result()])

    assert path.is_file()


def test_a_run_that_generated_nothing_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "stats_run.json"

    MetricsOutputSink(path).close()

    assert not path.exists()


def test_closing_twice_writes_once(tmp_path: Path) -> None:
    path = tmp_path / "stats_run.json"
    sink = MetricsOutputSink(path)
    sink.open(_session_desc())
    sink.write(_result(0, {"total_ms": 10.0}))

    sink.close()
    written = path.read_text(encoding="utf-8")
    sink.close()

    assert path.read_text(encoding="utf-8") == written


def test_nothing_can_be_recorded_before_a_session_is_open(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="open.. must run before write"):
        MetricsOutputSink(tmp_path / "stats_run.json").write(_result())
