# SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni.
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

"""Profiler implementations: NVTX ranges, CUDA-event stage timings, frame rates
and input latency."""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager

import torch

from flashdreams.api_v2.profiler import IProfiler
from flashdreams.infra.profiler import EventProfiler

_STAGE_PREFIX = "pipeline."
_STAGES = ("latency", "queue", "present")
"""Input latency and the two parts of it measured on their own."""


def _step_label(step: tuple[int, int]) -> str:
    generation, step_index = step
    if generation == 0:
        return f"[step {step_index}]"
    return f"[gen {generation}, step {step_index}]"


class NullProfiler(IProfiler):
    """Records nothing. The default, so an unprofiled run pays one no-op."""


class NVTXProfiler(IProfiler):
    """Marks ranges for Nsight Systems.

    Each batch of input also gets an ``input.wait`` range, from reaching the
    runtime to the first frame of the step that consumed it being shown. It is a
    start/end range, since waits overlap one another and the steps between them.

    ``torch.cuda.nvtx.range_push`` raises on CPU-only builds, so construct this
    only when CUDA is available.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Open input.wait ranges: by arrival until a step consumes them, then by step.
        self._arrived: dict[int, list[int]] = {}
        self._waiting: dict[tuple[int, int], list[int]] = {}

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def event(
        self, name: str, *, count: int = 1, step: tuple[int, int] | None = None
    ) -> None:
        del count
        torch.cuda.nvtx.mark(name if step is None else f"{name} {_step_label(step)}")

    def input_received(self, received_ns: int, count: int) -> None:
        if count <= 0:
            return
        range_id = torch.cuda.nvtx.range_start("input.wait")
        with self._lock:
            self._arrived.setdefault(received_ns, []).append(range_id)

    def _take_arrived(self, received_ns: Sequence[int]) -> list[int]:
        """Remove and return the waits of these arrivals. Hold the lock."""
        return [
            range_id
            for arrived_ns in set(received_ns)
            for range_id in self._arrived.pop(arrived_ns, [])
        ]

    def input_consumed(
        self, step: tuple[int, int], received_ns: Sequence[int], started_ns: int
    ) -> None:
        del started_ns
        with self._lock:
            if range_ids := self._take_arrived(received_ns):
                self._waiting.setdefault(step, []).extend(range_ids)

    def input_dropped(self, received_ns: Sequence[int]) -> None:
        with self._lock:
            ended = self._take_arrived(received_ns)
        for range_id in ended:
            torch.cuda.nvtx.range_end(range_id)

    def step_presented(self, step: tuple[int, int]) -> None:
        with self._lock:
            ended = self._waiting.pop(step, [])
            # An older step still waiting was dropped; its wait ends here.
            for older in [key for key in self._waiting if key < step]:
                ended += self._waiting.pop(older)
        for range_id in ended:
            torch.cuda.nvtx.range_end(range_id)

    def reset_counts(self) -> None:
        with self._lock:
            ended = [
                range_id
                for range_ids in (*self._arrived.values(), *self._waiting.values())
                for range_id in range_ids
            ]
            self._arrived.clear()
            self._waiting.clear()
        for range_id in ended:
            torch.cuda.nvtx.range_end(range_id)


class CudaEventProfiler(IProfiler):
    """Times ``pipeline.*`` ranges with CUDA events, as ``finalize`` reports them.

    Other ranges are marked but not timed: the timings are consecutive stages of
    one AR step, which nested or repeated names would not describe. Each
    :meth:`stage_scope` owner has its own timer, so a pipeline run between
    another's ``generate`` and ``finalize`` does not mix their stages. A timer is
    dropped once its owner is gone, since a run may skip its last ``finalize``.
    State is per-thread, so concurrent sessions do not interleave one another's
    stages.
    """

    def __init__(self) -> None:
        self._state = threading.local()

    def _timers(
        self,
    ) -> dict[int | None, tuple[weakref.ref | None, EventProfiler, set[str]]]:
        timers = getattr(self._state, "timers", None)
        if timers is None:
            timers = self._state.timers = {}
        return timers

    def _owner(self) -> tuple[object, int | None]:
        owner = getattr(self._state, "owner", None)
        return owner, None if owner is None else id(owner)

    @contextmanager
    def stage_scope(self, owner: object) -> Iterator[None]:
        previous = getattr(self._state, "owner", None)
        self._state.owner = owner
        try:
            yield
        finally:
            self._state.owner = previous

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        if not name.startswith(_STAGE_PREFIX):
            yield
            return
        stage = name.removeprefix(_STAGE_PREFIX)
        timers = self._timers()
        owner, key = self._owner()
        timer = timers.get(key)
        if (
            timer is None
            or stage in timer[2]
            or (timer[0] is not None and timer[0]() is not owner)
        ):
            # The first stage of a step, a step that raised before finalize
            # collected it, or an id reused after its owner was freed. Start a
            # fresh timer, and drop those whose owner is gone.
            for stale in [k for k, (ref, _, _) in timers.items() if ref and not ref()]:
                del timers[stale]
            ref = None if owner is None else weakref.ref(owner)
            timer = timers[key] = (ref, EventProfiler(), set())
        _, events, recorded = timer
        try:
            yield
        finally:
            events.record(stage)
            recorded.add(stage)

    def collect_stage_ms(self) -> dict[str, float]:
        timer = self._timers().pop(self._owner()[1], None)
        if timer is None:
            return {}
        return timer[1].sync_and_summarize()


class FrameRateProfiler(IProfiler):
    """Counts events and reports how many arrived per wall-clock second.

    The model loop reads the rates once per step, so ``<name>_fps`` covers one
    step, waits included, and ``<name>_avg_fps`` the session so far. This is
    not ``RecentFrameRateTracker``, which divides by time spent working and so
    leaves the waits out.

    The model thread and the UI thread both record, so the counts are guarded.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._since_read: dict[str, int] = {}
        self._since_start: dict[str, int] = {}
        self._started_at = time.monotonic()
        self._last_read_at = self._started_at

    def event(
        self, name: str, *, count: int = 1, step: tuple[int, int] | None = None
    ) -> None:
        del step
        if count <= 0:
            return
        with self._lock:
            self._since_read[name] = self._since_read.get(name, 0) + count
            self._since_start[name] = self._since_start.get(name, 0) + count

    def collect_fps(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            since_read_s = now - self._last_read_at
            since_start_s = now - self._started_at
            self._last_read_at = now
            rates: dict[str, float] = {}
            # Every name seen this session is reported, so a step in which
            # nothing was presented reads 0 rather than going missing.
            for name, total in self._since_start.items():
                if since_read_s > 0.0:
                    count = self._since_read.get(name, 0)
                    rates[f"{name}_fps"] = count / since_read_s
                if since_start_s > 0.0:
                    rates[f"{name}_avg_fps"] = total / since_start_s
            self._since_read.clear()
            return rates

    def reset_counts(self) -> None:
        with self._lock:
            self._since_read.clear()
            self._since_start.clear()
            self._started_at = time.monotonic()
            self._last_read_at = self._started_at


class InputLatencyProfiler(IProfiler):
    """Times each input from reaching the runtime to its first frame being shown.

    Every input a step consumed counts, and the reported value is their mean.
    ``input.queue_ms`` is the wait for the step to start, ``input.present_ms`` runs
    from the step returning to its frame being shown, and the rest is the step
    itself. Each key also has an ``_avg_ms`` form covering the session. Input read
    by a step that buffered its output counts once a later step shows it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Steps that carry input and have not shown a frame yet:
        # (generation, step) -> [(input arrival, read start, step end)], in ns.
        self._waiting: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        self._since_read: dict[str, list[float]] = {name: [] for name in _STAGES}
        self._totals: dict[str, float] = dict.fromkeys(_STAGES, 0.0)
        self._count = 0

    def input_consumed(
        self, step: tuple[int, int], received_ns: Sequence[int], started_ns: int
    ) -> None:
        if not received_ns:
            return
        ended_ns = time.monotonic_ns()
        with self._lock:
            self._waiting.setdefault(step, []).extend(
                (arrived_ns, started_ns, ended_ns) for arrived_ns in received_ns
            )

    def step_presented(self, step: tuple[int, int]) -> None:
        presented_ns = time.monotonic_ns()
        with self._lock:
            entries = self._waiting.pop(step, [])
            # An older step still waiting was dropped, so its frame never appears.
            for older in [key for key in self._waiting if key < step]:
                del self._waiting[older]
            for arrived_ns, started_ns, ended_ns in entries:
                stages_ms = {
                    "latency": (presented_ns - arrived_ns) / 1e6,
                    "queue": (started_ns - arrived_ns) / 1e6,
                    "present": (presented_ns - ended_ns) / 1e6,
                }
                for name, value in stages_ms.items():
                    self._since_read[name].append(value)
                    self._totals[name] += value
                self._count += 1

    def collect_input_latency_ms(self) -> dict[str, float]:
        with self._lock:
            result: dict[str, float] = {}
            # A record with no input shown since the last one leaves these keys out.
            for name, values in self._since_read.items():
                if values:
                    result[f"input.{name}_ms"] = sum(values) / len(values)
                    values.clear()
            if self._count:
                for name, total in self._totals.items():
                    result[f"input.{name}_avg_ms"] = total / self._count
            return result

    def reset_counts(self) -> None:
        with self._lock:
            self._waiting.clear()
            for values in self._since_read.values():
                values.clear()
            self._totals = dict.fromkeys(_STAGES, 0.0)
            self._count = 0


class CompositeProfiler(IProfiler):
    """Runs several backends over one set of call sites, outermost first."""

    def __init__(self, profilers: tuple[IProfiler, ...]) -> None:
        self._profilers = profilers

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        with ExitStack() as stack:
            for profiler in self._profilers:
                stack.enter_context(profiler.range(name))
            yield

    @contextmanager
    def stage_scope(self, owner: object) -> Iterator[None]:
        with ExitStack() as stack:
            for profiler in self._profilers:
                stack.enter_context(profiler.stage_scope(owner))
            yield

    def event(
        self, name: str, *, count: int = 1, step: tuple[int, int] | None = None
    ) -> None:
        for profiler in self._profilers:
            profiler.event(name, count=count, step=step)

    def input_received(self, received_ns: int, count: int) -> None:
        for profiler in self._profilers:
            profiler.input_received(received_ns, count)

    def input_consumed(
        self, step: tuple[int, int], received_ns: Sequence[int], started_ns: int
    ) -> None:
        for profiler in self._profilers:
            profiler.input_consumed(step, received_ns, started_ns)

    def input_dropped(self, received_ns: Sequence[int]) -> None:
        for profiler in self._profilers:
            profiler.input_dropped(received_ns)

    def step_presented(self, step: tuple[int, int]) -> None:
        for profiler in self._profilers:
            profiler.step_presented(step)

    def collect_input_latency_ms(self) -> dict[str, float]:
        latency_ms: dict[str, float] = {}
        for profiler in self._profilers:
            latency_ms.update(profiler.collect_input_latency_ms())
        return latency_ms

    def collect_stage_ms(self) -> dict[str, float]:
        # Drain every backend: one left uncollected would carry into the next step.
        stage_ms: dict[str, float] = {}
        for profiler in self._profilers:
            stage_ms.update(profiler.collect_stage_ms())
        return stage_ms

    def collect_fps(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for profiler in self._profilers:
            rates.update(profiler.collect_fps())
        return rates

    def reset_counts(self) -> None:
        for profiler in self._profilers:
            profiler.reset_counts()
