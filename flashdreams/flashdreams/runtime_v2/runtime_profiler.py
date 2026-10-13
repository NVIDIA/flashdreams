# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side input-latency profiling for the V2 runtime."""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path
from typing import IO, Any

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_ARTIFACT_TYPE = "flashdreams.runtime_v2.input_latency_profile"
"""Artifact type for V2 host-side input-latency records."""

_SCHEMA_VERSION = 2
"""Input-latency artifact schema version."""

_SUMMARY_SAMPLE_LIMIT = 1_024
"""Maximum observations retained in each pending or session quantile reservoir."""


class _MetricAccumulator:
    """Keep exact counts and maxima with bounded quantile samples."""

    def __init__(self, random_source: random.Random) -> None:
        self._count = 0
        self._maximum: float | None = None
        self._samples: list[float] = []
        self._random = random_source

    def add(self, value: float) -> None:
        """Add one sample using deterministic reservoir sampling."""
        self._count += 1
        self._maximum = value if self._maximum is None else max(self._maximum, value)
        if len(self._samples) < _SUMMARY_SAMPLE_LIMIT:
            self._samples.append(value)
            return
        replacement = self._random.randrange(self._count)
        if replacement < _SUMMARY_SAMPLE_LIMIT:
            self._samples[replacement] = value

    def add_batch(
        self, *, count: int, maximum_s: float, samples_s: list[float]
    ) -> None:
        """Merge a batch's count, maximum, and uniform quantile reservoir."""
        if count == 1:
            self.add(samples_s[0])
            return
        total_count = self._count + count
        if total_count <= _SUMMARY_SAMPLE_LIMIT:
            self._samples.extend(samples_s)
        else:
            # Select quotas from the full populations, then from their reservoirs.
            old_count = sum(
                index < self._count
                for index in self._random.sample(
                    range(total_count), _SUMMARY_SAMPLE_LIMIT
                )
            )
            self._samples = self._random.sample(
                self._samples, old_count
            ) + self._random.sample(samples_s, _SUMMARY_SAMPLE_LIMIT - old_count)
        self._count = total_count
        self._maximum = (
            maximum_s if self._maximum is None else max(self._maximum, maximum_s)
        )

    def summary(self) -> dict[str, float | int | bool]:
        """Return bounded-memory summary statistics."""
        if self._count == 0:
            return {"count": 0}
        ordered = sorted(self._samples)
        maximum = self._maximum
        assert maximum is not None
        return {
            "count": self._count,
            "median_s": statistics.median(ordered),
            "p90_s": _percentile(ordered, 0.9),
            "max_s": maximum,
            "quantile_sample_count": len(ordered),
            "quantiles_approximate": self._count > len(ordered),
        }


class RuntimeProfiler:
    """Write input-to-IUILoop and input-to-window-write records as JSONL.

    One runtime UI thread owns an instance. Input timestamps and runtime
    observations share ``time.monotonic_ns`` through the source-provided session
    origin. Each claim is written immediately and identifies the next window
    write in its generation. Bounded timestamp samples provide session summaries.
    Reopening the profiler appends an independent session segment to the artifact.
    """

    def __init__(self, path: str | Path) -> None:
        """Configure a profile artifact that opens with the session."""
        self._path = Path(path).expanduser()
        self._output: IO[str] | None = None
        self._input_timestamp_origin_ns: int | None = None
        self._input_generation: int | None = None
        self._window_write_index = 0
        self._pending_input_count = 0
        self._earliest_pending_input_ns: int | None = None
        self._latest_pending_input_ns: int | None = None
        self._pending_input_times_ns: list[int] = []
        self._random = random.Random(0)
        self._samples = _metric_accumulators(self._random)
        self._opened_once = False

    @property
    def path(self) -> Path:
        """Return the profile output path."""
        return self._path

    def validate_output_path(self, output_path: str | Path) -> None:
        """Reject another artifact writer targeting the profile file.

        Args:
            output_path: Path as used by the other writer, with its own path
                expansion already applied.

        Raises:
            ValueError: Both writers target the same file.
        """
        profile_path = self._path.resolve()
        output_path = Path(output_path).resolve()
        if profile_path == output_path or (
            profile_path.exists()
            and output_path.exists()
            and profile_path.samefile(output_path)
        ):
            raise ValueError("Runtime profile must use a distinct output path.")

    def session_started(
        self,
        *,
        input_timestamp_origin_ns: int | None,
        session_desc: SessionDesc,
        client_window_type: str,
        time_ns: int | None = None,
    ) -> None:
        """Open the artifact and bind input timestamps to the runtime clock."""
        if self._output is not None:
            raise RuntimeError("RuntimeProfiler session already started.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._opened_once else "w"
        self._output = self._path.open(mode, encoding="utf-8")
        self._opened_once = True
        self._input_timestamp_origin_ns = input_timestamp_origin_ns
        self._input_generation = None
        self._window_write_index = 0
        self._clear_pending_inputs()
        self._samples = _metric_accumulators(self._random)
        self._write(
            "session_started",
            _now_ns(time_ns),
            input_timestamp_origin_ns=input_timestamp_origin_ns,
            measurement_endpoints={
                "input_to_ui_step_s": "ui_loop_begin_run",
                "input_to_window_write_s": "client_window_write_return",
            },
            runtime_settings={
                "client_window_type": client_window_type,
                "output_layout": session_desc.output_layout.value,
                "frames_per_second_for_ui": session_desc.frames_per_second_for_ui,
                "frames_per_second_for_step": session_desc.frames_per_second_for_step,
                "video_width": session_desc.video_width,
                "video_height": session_desc.video_height,
                "presentation_mode": session_desc.presentation_mode.value,
                "backpressure_mode": session_desc.backpressure_mode.value,
            },
        )

    def ui_step_started(
        self,
        events: UserInputEvents,
        *,
        generation: int,
        step: int,
        time_ns: int | None = None,
    ) -> None:
        """Record the UI step that first claims each input event."""
        started_at_ns = _now_ns(time_ns)
        self._ensure_started()
        if generation != self._input_generation:
            self._clear_pending_inputs()
            self._input_generation = generation

        origin_ns = self._input_timestamp_origin_ns
        if origin_ns is None:
            return
        for event in events.get_events():
            received_at_ns = origin_ns + int(event.get_timestamp()) * 1_000
            claim_duration_s = _elapsed_s(received_at_ns, started_at_ns)
            self._write(
                "input_to_ui_step",
                started_at_ns,
                generation=generation,
                window_write_index=self._window_write_index,
                step=step,
                input_type=event.get_type_name(),
                input_timestamp_us=int(event.get_timestamp()),
                duration_s=claim_duration_s,
            )
            self._samples["input_to_ui_step_s"].add(claim_duration_s)
            self._pending_input_count += 1
            self._earliest_pending_input_ns = (
                received_at_ns
                if self._earliest_pending_input_ns is None
                else min(self._earliest_pending_input_ns, received_at_ns)
            )
            self._latest_pending_input_ns = (
                received_at_ns
                if self._latest_pending_input_ns is None
                else max(self._latest_pending_input_ns, received_at_ns)
            )
            if len(self._pending_input_times_ns) < _SUMMARY_SAMPLE_LIMIT:
                self._pending_input_times_ns.append(received_at_ns)
            else:
                replacement = self._random.randrange(self._pending_input_count)
                if replacement < _SUMMARY_SAMPLE_LIMIT:
                    self._pending_input_times_ns[replacement] = received_at_ns

    def window_write_completed(
        self,
        *,
        generation: int,
        ui_step: int,
        time_ns: int | None = None,
    ) -> None:
        """Finish pending input correlations at the next window write return."""
        completed_at_ns = _now_ns(time_ns)
        self._ensure_started()
        if generation != self._input_generation:
            self._clear_pending_inputs()
        earliest_ns = self._earliest_pending_input_ns
        latest_ns = self._latest_pending_input_ns
        if earliest_ns is not None and latest_ns is not None:
            _elapsed_s(latest_ns, completed_at_ns)
            self._samples["input_to_window_write_s"].add_batch(
                count=self._pending_input_count,
                maximum_s=_elapsed_s(earliest_ns, completed_at_ns),
                samples_s=[
                    _elapsed_s(received_at_ns, completed_at_ns)
                    for received_at_ns in self._pending_input_times_ns
                ],
            )
        self._write(
            "window_write_completed",
            completed_at_ns,
            generation=generation,
            window_write_index=self._window_write_index,
            presented_ui_step=ui_step,
            input_count=self._pending_input_count,
        )
        self._clear_pending_inputs()
        self._window_write_index += 1

    def summary(self) -> dict[str, dict[str, float | int | bool]]:
        """Return summary statistics for both host-side latency metrics."""
        return {name: values.summary() for name, values in self._samples.items()}

    def close(self) -> None:
        """Write summaries and close the active session segment."""
        output = self._output
        if output is None:
            return
        failure: BaseException | None = None
        try:
            observed_at_ns = time.monotonic_ns()
            for metric, values in self._samples.items():
                self._write(
                    "profile_summary",
                    observed_at_ns,
                    metric=metric,
                    **values.summary(),
                )
        except BaseException as error:
            failure = error
        self._clear_pending_inputs()
        try:
            output.close()
        except BaseException as error:
            if failure is None:
                failure = error
        self._output = None
        self._input_timestamp_origin_ns = None
        self._input_generation = None
        if failure is not None:
            raise failure

    def _clear_pending_inputs(self) -> None:
        self._pending_input_count = 0
        self._earliest_pending_input_ns = None
        self._latest_pending_input_ns = None
        self._pending_input_times_ns.clear()

    def _write(self, phase: str, time_ns: int, **fields: Any) -> None:
        output = self._output
        if output is None:
            raise RuntimeError("RuntimeProfiler session has not started.")
        record = {
            **fields,
            "artifact_type": _ARTIFACT_TYPE,
            "schema_version": _SCHEMA_VERSION,
            "phase": phase,
            "time_ns": time_ns,
        }
        output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        output.write("\n")

    def _ensure_started(self) -> None:
        if self._output is None:
            raise RuntimeError("RuntimeProfiler session has not started.")


def _now_ns(value: int | None) -> int:
    return time.monotonic_ns() if value is None else value


def _elapsed_s(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise ValueError("A runtime profile observation moved backward in time.")
    return (end_ns - start_ns) / 1_000_000_000


def _metric_accumulators(random_source: random.Random) -> dict[str, _MetricAccumulator]:
    return {
        "input_to_ui_step_s": _MetricAccumulator(random_source),
        "input_to_window_write_s": _MetricAccumulator(random_source),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = percentile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["RuntimeProfiler"]
