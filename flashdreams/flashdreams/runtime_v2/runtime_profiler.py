# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correlated host-side latency profiling for the V2 runtime."""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from flashdreams.runtime_v2.user_input_events import UserInputEvents

_SCHEMA_VERSION = 1
"""Runtime profile JSONL schema version."""


@dataclass(frozen=True, slots=True)
class _ClaimedInput:
    """One input event claimed by a UI step."""

    received_at_ns: int
    timestamp_us: int
    event_type: str
    generation: int
    ui_step: int


@dataclass(frozen=True, slots=True)
class _PendingOutput:
    """One selected model frame waiting for a client-window write."""

    generation: int
    step: int
    frame: int
    selected_at_ns: int


class RuntimeProfiler:
    """Write correlated V2 runtime latency records as line-delimited JSON.

    Input events use their session-relative timestamp as the causal origin. The
    profiler follows each event from its first UI step to the next observable
    client-window write. Model and presentation stages carry independent
    ``(generation, step)`` identities. All durations use one host monotonic
    clock. A profiler instance belongs to one session and is thread-safe.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        """Open a new runtime profile.

        Args:
            path: JSONL output path. Parent directories are created.
            clock_ns: Monotonic clock used for every runtime observation.
        """
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._output: IO[str] = self._path.open("w", encoding="utf-8")
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._input_timestamp_origin_ns: int | None = None
        self._pending_inputs: list[_ClaimedInput] = []
        self._input_generation: int | None = None
        self._pending_output: _PendingOutput | None = None
        self._samples: dict[str, list[float]] = {
            "input_to_ui_step_s": [],
            "input_to_window_write_s": [],
            "model_step_s": [],
            "ui_step_s": [],
            "publish_wait_s": [],
            "frame_to_window_write_s": [],
        }
        self._closed = False
        with self._lock:
            self._write_locked("profile_started", self._clock_ns())

    @property
    def path(self) -> Path:
        """Return the profile output path."""
        return self._path

    def timestamp_ns(self) -> int:
        """Return the profiler's monotonic timestamp."""
        return self._clock_ns()

    def session_started(
        self,
        *,
        input_timestamp_origin_ns: int | None,
        time_ns: int | None = None,
    ) -> None:
        """Set the host-clock origin for session-relative input timestamps."""
        observed_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._input_timestamp_origin_ns = input_timestamp_origin_ns
            self._write_locked(
                "session_started",
                observed_at_ns,
                input_timestamp_origin_ns=input_timestamp_origin_ns,
            )

    def record(self, phase: str, *, time_ns: int | None = None, **fields: Any) -> None:
        """Write one timestamped runtime phase."""
        if not phase.strip():
            raise ValueError("Runtime profile phases must be non-empty.")
        observed_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._write_locked(phase, observed_at_ns, **fields)

    def model_step_started(
        self,
        *,
        generation: int,
        step: int,
        time_ns: int | None = None,
    ) -> None:
        """Record one model-step entry."""
        started_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._write_locked(
                "model_step_started",
                started_at_ns,
                generation=generation,
                step=step,
            )

    def model_step_completed(
        self,
        *,
        generation: int,
        step: int,
        duration_s: float,
        time_ns: int | None = None,
    ) -> None:
        """Record one completed model step."""
        duration_s = _duration(duration_s)
        completed_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._samples["model_step_s"].append(duration_s)
            self._write_locked(
                "model_step_completed",
                completed_at_ns,
                generation=generation,
                step=step,
                duration_s=duration_s,
            )

    def ui_step_started(
        self,
        events: UserInputEvents,
        *,
        generation: int,
        step: int,
        time_ns: int | None = None,
    ) -> None:
        """Record one UI-step entry and the input events it claims."""
        started_at_ns = self._clock_ns() if time_ns is None else time_ns
        event_list = events.get_events()
        with self._lock:
            self._ensure_open_locked()
            if generation != self._input_generation:
                self._pending_inputs.clear()
                self._input_generation = generation
            origin_ns = self._input_timestamp_origin_ns
            observations = (
                ()
                if origin_ns is None
                else tuple(
                    _ClaimedInput(
                        received_at_ns=origin_ns + int(event.get_timestamp()) * 1_000,
                        timestamp_us=int(event.get_timestamp()),
                        event_type=type(event).__name__,
                        generation=generation,
                        ui_step=step,
                    )
                    for event in event_list
                )
            )
            self._pending_inputs.extend(observations)
            self._write_locked(
                "ui_step_started",
                started_at_ns,
                generation=generation,
                step=step,
                input_count=len(event_list),
                timed_input_count=len(observations),
            )
            for observation in observations:
                duration_s = _elapsed_s(observation.received_at_ns, started_at_ns)
                self._samples["input_to_ui_step_s"].append(duration_s)
                self._write_locked(
                    "input_to_ui_step",
                    started_at_ns,
                    generation=generation,
                    step=step,
                    input_type=observation.event_type,
                    input_timestamp_us=observation.timestamp_us,
                    duration_s=duration_s,
                )

    def ui_step_completed(
        self,
        *,
        generation: int,
        step: int,
        duration_s: float,
        presented: bool,
        time_ns: int | None = None,
    ) -> None:
        """Record one completed UI step."""
        duration_s = _duration(duration_s)
        completed_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._samples["ui_step_s"].append(duration_s)
            self._write_locked(
                "ui_step_completed",
                completed_at_ns,
                generation=generation,
                step=step,
                duration_s=duration_s,
                presented=presented,
            )

    def chunk_published(
        self,
        *,
        generation: int,
        step: int,
        frame_count: int,
        wait_s: float,
        time_ns: int | None = None,
    ) -> None:
        """Record admission to the presentation queue."""
        wait_s = _duration(wait_s)
        published_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._samples["publish_wait_s"].append(wait_s)
            self._write_locked(
                "chunk_published",
                published_at_ns,
                generation=generation,
                step=step,
                frame_count=frame_count,
                wait_s=wait_s,
            )

    def chunk_dropped(
        self,
        *,
        generation: int,
        step: int,
        reason: str,
        time_ns: int | None = None,
    ) -> None:
        """Discard stage state for a dropped model chunk."""
        dropped_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            if self._pending_output is not None and (
                self._pending_output.generation,
                self._pending_output.step,
            ) == (generation, step):
                self._pending_output = None
            self._write_locked(
                "chunk_dropped",
                dropped_at_ns,
                generation=generation,
                step=step,
                reason=reason,
            )

    def frame_selected(
        self,
        *,
        generation: int,
        step: int,
        frame: int,
        frame_count: int,
        time_ns: int | None = None,
    ) -> None:
        """Record model-frame selection for the next client-window write."""
        selected_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._write_locked(
                "frame_selected",
                selected_at_ns,
                generation=generation,
                step=step,
                frame=frame,
                frame_count=frame_count,
            )
            self._pending_output = _PendingOutput(
                generation=generation,
                step=step,
                frame=frame,
                selected_at_ns=selected_at_ns,
            )

    def window_write_completed(
        self,
        *,
        endpoint: str | None,
        generation: int,
        ui_step: int,
        time_ns: int | None = None,
    ) -> None:
        """Record a client-window write and finish successful correlations."""
        completed_at_ns = self._clock_ns() if time_ns is None else time_ns
        with self._lock:
            self._ensure_open_locked()
            self._write_locked(
                "window_write_completed",
                completed_at_ns,
                endpoint=endpoint,
                generation=generation,
                ui_step=ui_step,
            )
            if endpoint is None:
                return
            pending_output = self._pending_output
            self._pending_output = None
            if pending_output is not None:
                frame_duration_s = _elapsed_s(
                    pending_output.selected_at_ns, completed_at_ns
                )
                self._samples["frame_to_window_write_s"].append(frame_duration_s)
                self._write_locked(
                    "frame_to_window_write",
                    completed_at_ns,
                    generation=pending_output.generation,
                    step=pending_output.step,
                    frame=pending_output.frame,
                    endpoint=endpoint,
                    duration_s=frame_duration_s,
                )
            pending_inputs = tuple(self._pending_inputs)
            self._pending_inputs.clear()
            for observation in pending_inputs:
                duration_s = _elapsed_s(observation.received_at_ns, completed_at_ns)
                self._samples["input_to_window_write_s"].append(duration_s)
                self._write_locked(
                    "input_to_window_write",
                    completed_at_ns,
                    generation=observation.generation,
                    claimed_ui_step=observation.ui_step,
                    presented_ui_step=ui_step,
                    endpoint=endpoint,
                    input_type=observation.event_type,
                    input_timestamp_us=observation.timestamp_us,
                    duration_s=duration_s,
                )

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return summary statistics for every recorded duration."""
        with self._lock:
            return {name: _summarize(values) for name, values in self._samples.items()}

    def close(self) -> None:
        """Write summary records and close the profile once."""
        with self._lock:
            if self._closed:
                return
            observed_at_ns = self._clock_ns()
            failure: BaseException | None = None
            try:
                for metric, values in self._samples.items():
                    self._write_locked(
                        "profile_summary",
                        observed_at_ns,
                        metric=metric,
                        **_summarize(values),
                    )
            except BaseException as error:
                failure = error
            self._pending_inputs.clear()
            self._pending_output = None
            self._closed = True
            try:
                self._output.close()
            except BaseException as error:
                if failure is None:
                    failure = error
            if failure is not None:
                raise failure

    def _write_locked(self, phase: str, time_ns: int, **fields: Any) -> None:
        record = {
            **fields,
            "schema_version": _SCHEMA_VERSION,
            "phase": phase,
            "time_ns": time_ns,
        }
        self._output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        self._output.write("\n")

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("RuntimeProfiler is closed.")


def _summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize one duration distribution."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_s": statistics.fmean(ordered),
        "median_s": statistics.median(ordered),
        "p90_s": _percentile(ordered, 0.9),
        "max_s": ordered[-1],
    }


def _elapsed_s(start_ns: int, end_ns: int) -> float:
    """Return a valid elapsed duration on one monotonic clock."""
    if end_ns < start_ns:
        raise ValueError("A runtime profile observation moved backward in time.")
    return (end_ns - start_ns) / 1_000_000_000


def _duration(value: float) -> float:
    """Return a finite nonnegative duration."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Runtime profile durations must be finite and nonnegative.")
    return value


def _percentile(ordered: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile from sorted samples."""
    if len(ordered) == 1:
        return ordered[0]
    index = percentile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["RuntimeProfiler"]
