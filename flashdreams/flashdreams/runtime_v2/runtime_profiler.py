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

"""Host-side input-latency profiling for the V2 runtime."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from flashdreams.runtime_v2.user_input_events import UserInputEvents

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ClaimedInput:
    received_at_ns: int
    claimed_at_ns: int
    claim_duration_s: float
    timestamp_us: int
    event_type: str
    generation: int
    ui_step: int


class RuntimeProfiler:
    """Write input-to-IUILoop and input-to-window-write records as JSONL.

    One runtime UI thread owns an instance. Input timestamps and runtime
    observations share ``time.monotonic_ns`` through the source-provided session
    origin. Each claimed input remains pending through the first subsequent
    client-window write.
    """

    def __init__(self, path: str | Path) -> None:
        """Configure a profile artifact that opens with the session."""
        self._path = Path(path).expanduser()
        self._output: IO[str] | None = None
        self._input_timestamp_origin_ns: int | None = None
        self._input_generation: int | None = None
        self._pending_inputs: list[_ClaimedInput] = []
        self._samples: dict[str, list[float]] = {
            "input_to_ui_step_s": [],
            "input_to_window_write_s": [],
        }
        self._closed = False

    @property
    def path(self) -> Path:
        """Return the profile output path."""
        return self._path

    def session_started(
        self,
        *,
        input_timestamp_origin_ns: int | None,
        time_ns: int | None = None,
    ) -> None:
        """Open the artifact and bind input timestamps to the runtime clock."""
        if self._output is not None:
            raise RuntimeError("RuntimeProfiler session already started.")
        self._ensure_open()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self._path.open("w", encoding="utf-8")
        self._input_timestamp_origin_ns = input_timestamp_origin_ns
        self._write(
            "session_started",
            _now_ns(time_ns),
            input_timestamp_origin_ns=input_timestamp_origin_ns,
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
            self._write_claims(self._pending_inputs)
            self._pending_inputs.clear()
            self._input_generation = generation

        origin_ns = self._input_timestamp_origin_ns
        if origin_ns is None:
            return
        for event in events.get_events():
            received_at_ns = origin_ns + int(event.get_timestamp()) * 1_000
            claim_duration_s = _elapsed_s(received_at_ns, started_at_ns)
            self._pending_inputs.append(
                _ClaimedInput(
                    received_at_ns=received_at_ns,
                    claimed_at_ns=started_at_ns,
                    claim_duration_s=claim_duration_s,
                    timestamp_us=int(event.get_timestamp()),
                    event_type=type(event).__name__,
                    generation=generation,
                    ui_step=step,
                )
            )
            self._samples["input_to_ui_step_s"].append(claim_duration_s)

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
        pending = tuple(
            observation
            for observation in self._pending_inputs
            if observation.generation == generation
        )
        self._pending_inputs.clear()
        self._write_claims(pending)
        for observation in pending:
            duration_s = _elapsed_s(observation.received_at_ns, completed_at_ns)
            self._samples["input_to_window_write_s"].append(duration_s)
            self._write(
                "input_to_window_write",
                completed_at_ns,
                generation=generation,
                claimed_ui_step=observation.ui_step,
                presented_ui_step=ui_step,
                input_type=observation.event_type,
                input_timestamp_us=observation.timestamp_us,
                duration_s=duration_s,
            )

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return summary statistics for both host-side latency metrics."""
        return {name: _summarize(values) for name, values in self._samples.items()}

    def close(self) -> None:
        """Write summaries and close the profile once."""
        if self._closed:
            return
        self._closed = True
        output = self._output
        if output is None:
            return
        failure: BaseException | None = None
        try:
            self._write_claims(self._pending_inputs)
            observed_at_ns = time.monotonic_ns()
            for metric, values in self._samples.items():
                self._write(
                    "profile_summary",
                    observed_at_ns,
                    metric=metric,
                    **_summarize(values),
                )
        except BaseException as error:
            failure = error
        self._pending_inputs.clear()
        try:
            output.close()
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def _write_claims(
        self,
        observations: list[_ClaimedInput] | tuple[_ClaimedInput, ...],
    ) -> None:
        for observation in observations:
            self._write(
                "input_to_ui_step",
                observation.claimed_at_ns,
                generation=observation.generation,
                step=observation.ui_step,
                input_type=observation.event_type,
                input_timestamp_us=observation.timestamp_us,
                duration_s=observation.claim_duration_s,
            )

    def _write(self, phase: str, time_ns: int, **fields: Any) -> None:
        output = self._output
        if output is None:
            raise RuntimeError("RuntimeProfiler session has not started.")
        record = {
            **fields,
            "schema_version": _SCHEMA_VERSION,
            "phase": phase,
            "time_ns": time_ns,
        }
        output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        output.write("\n")

    def _ensure_started(self) -> None:
        self._ensure_open()
        if self._output is None:
            raise RuntimeError("RuntimeProfiler session has not started.")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RuntimeProfiler is closed.")


def _now_ns(value: int | None) -> int:
    return time.monotonic_ns() if value is None else value


def _elapsed_s(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise ValueError("A runtime profile observation moved backward in time.")
    return (end_ns - start_ns) / 1_000_000_000


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median_s": statistics.median(ordered),
        "p90_s": _percentile(ordered, 0.9),
        "max_s": ordered[-1],
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
