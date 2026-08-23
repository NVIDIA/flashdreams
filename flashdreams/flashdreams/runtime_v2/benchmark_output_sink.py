# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink recording model-step measurements for a benchmark."""

import json
import math
from pathlib import Path
from typing import Any

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult

_ARTIFACT_TYPE = "flashdreams.runtime.demo.benchmark_stats"
"""Artifact type shared with the v1 benchmark sink."""

_SCHEMA_VERSION = 1
"""Benchmark artifact schema version."""


class BenchmarkOutputSink(OutputSink):
    """Write model metrics to a benchmark JSON file.

    Each result records its frame count and finite numeric metrics. Metric names
    ending in ``_ms`` are converted to seconds to match the v1 sink.
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
        """Start a new benchmark record."""
        self._session_desc = session_desc
        self._steps = []
        self._samples = []
        self._written = False

    def write(self, result: StepResult) -> None:
        """Record one model result's frame count and metrics.

        Raises:
            RuntimeError: Called before :meth:`open`.
        """
        if self._session_desc is None:
            raise RuntimeError("BenchmarkOutputSink.open() must run before write().")
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
        """Write the JSON file once."""
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
    """Return the finite numeric metrics from one result."""
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
    """Return the name, value, unit, and category used by benchmark reports."""
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
