# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink recording what each step measured, for a benchmark to read."""

import json
import math
from pathlib import Path
from typing import Any

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult

_ARTIFACT_TYPE = "flashdreams.runtime.demo.benchmark_stats"
"""What a stats file declares itself to be.

The same as the v1 sink writes, so one report can hold runs of both APIs.
"""

_SCHEMA_VERSION = 1
"""Version of the payload below, as the benchmark tool numbers this artifact."""


class MetricsOutputSink(OutputSink):
    """Write what a run measured, where a benchmark report reads it from.

    A clip says nothing about what it cost, so a run compared for speed writes
    this alongside its video. Each step's measurements are recorded against the
    frames it generated, so a report can say how fast video was generated
    rather than only how long a step took.

    The measurements are the model's own, normalized the way the v1 sink
    normalizes them: milliseconds become seconds, since a report cannot compare
    two units.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path: JSON file to write. Parent directories are created.
        """
        self._path = Path(path)
        self._session_desc: SessionDesc | None = None
        self._steps: list[dict[str, Any]] = []
        self._samples: list[dict[str, Any]] = []
        self._written = False

    def open(self, session_desc: SessionDesc) -> None:
        """Start recording a session's measurements.

        The description is recorded too, so a report knows what was being
        generated while it was timed.
        """
        self._session_desc = session_desc
        self._steps = []
        self._samples = []
        self._written = False

    def write(self, result: StepResult) -> None:
        """Record what ``result`` measured, and how much video it came with.

        Frames are counted rather than kept; writing them out belongs to
        whatever sink is writing the video.

        Raises:
            RuntimeError: Called before :meth:`open`.
        """
        if self._session_desc is None:
            raise RuntimeError("MetricsOutputSink.open() must run before write().")
        samples = _samples_from(result)
        self._samples.extend(samples)
        self._steps.append(
            {
                "step_index": result.step_index,
                "frame_count": result.frame_count,
                "sample_count": len(samples),
            }
        )

    def close(self) -> None:
        """Write the file.

        A sink that was never opened leaves no file behind, and one closed
        twice writes it once.
        """
        session_desc = self._session_desc
        self._session_desc = None
        if session_desc is None or self._written:
            return
        self._written = True
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "session": {
                "output_layout": session_desc.output_layout.value,
                "frames_per_second_for_step": session_desc.frames_per_second_for_step,
                "video_width": session_desc.video_width,
                "video_height": session_desc.video_height,
            },
            "steps": self._steps,
            "samples": self._samples,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _samples_from(result: StepResult) -> list[dict[str, Any]]:
    """Return one record per measurement a step reported.

    A measurement that is not a finite number is dropped, since a report can
    only average what it can compare.
    """
    samples: list[dict[str, Any]] = []
    for name, value in result.metrics.items():
        if not name.strip():
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(float(value)):
            continue
        sample_name, sample_value, unit, category = _normalized_sample(name, value)
        samples.append(
            {
                "name": sample_name,
                "value": sample_value,
                "unit": unit,
                "category": category,
                "step_index": result.step_index,
                "metadata": {"frame_count": result.frame_count},
            }
        )
    return samples


def _normalized_sample(
    name: str, value: float | int
) -> tuple[str, float | int, str, str]:
    """Return a measurement as a report reads it: name, value, unit, category.

    Milliseconds become seconds; everything else keeps its value and is
    labelled with what its name says it is. This is
    ``flashdreams.demo.outputs._normalize_metric_sample`` for v2 results, and
    the two agreeing is what lets one report hold runs of both.
    """
    if name.endswith("_ms"):
        return f"{name[:-3]}_s", float(value) / 1000.0, "s", "timing"
    if name.endswith("_s"):
        return name, value, "s", "timing"
    if name.endswith("_fps"):
        return name, value, "fps", "throughput"
    if name.endswith("_bytes"):
        return name, value, "bytes", "runtime"
    if name.endswith("_gib"):
        return name, value, "gib", "memory"
    if name.endswith("_count") or name in {"frames", "frame_count"}:
        return name, value, "count", "runtime"
    return name, value, "value", "runtime"
