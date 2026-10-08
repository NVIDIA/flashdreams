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

"""Profiling interface, its implementations, and the CUDA-event timer."""

from __future__ import annotations

import os
import threading
import time
from abc import ABC
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar, Token
from typing import ContextManager

import torch

_STAGE_PREFIX = "pipeline."
_STAGES = ("latency", "queue", "present")
"""Input latency and the two parts of it measured on their own."""


def _step_label(step: tuple[int, int]) -> str:
    generation, step_index = step
    if generation == 0:
        return f"[step {step_index}]"
    return f"[gen {generation}, step {step_index}]"


class IProfiler(ABC):
    """Profiling backend for the runtime and the pipeline.

    Implementations decide what a range costs: a trace marker, a CUDA-event
    timing, or nothing. Call sites name a range and never import a backend.
    """

    def range(self, name: str) -> ContextManager[None]:
        """Return a context manager covering the block as ``name``."""
        del name
        return nullcontext()

    def stage_scope(self, owner: object) -> ContextManager[None]:
        """Return a context manager that files the stages timed inside under
        ``owner``, so a pipeline run inside another keeps its own timings."""
        del owner
        return nullcontext()

    def event(
        self, name: str, *, count: int = 1, step: tuple[int, int] | None = None
    ) -> None:
        """Mark that ``name`` happened, covering ``count`` items.

        A range has width; an event does not. ``count`` is how many items the
        event stands for, so one chunk of frames is a single call. ``step`` is the
        ``(generation, step_index)`` the event belongs to, which a trace shows
        beside the name so samples from one step can be found together.
        """
        del name, count, step

    def input_received(self, received_ns: int, count: int) -> None:
        """Note that ``count`` inputs reached the runtime at ``received_ns``.

        The time is ``time.monotonic_ns()`` and matches what the step that
        consumes them later reports to :meth:`input_consumed`.
        """
        del received_ns, count

    def collect_stage_ms(self) -> dict[str, float]:
        """Return and clear the pipeline stage timings of the step just run.

        Empty when the backend does not time stages.
        """
        return {}

    def collect_fps(self) -> dict[str, float]:
        """Return event rates since the previous call and since the session began.

        Empty when the backend does not count events.
        """
        return {}

    def input_consumed(
        self, step: tuple[int, int], received_ns: Sequence[int], started_ns: int
    ) -> None:
        """Note that the step keyed ``(generation, step_index)``, returning now,
        carries the output for inputs that reached the runtime at ``received_ns``
        and were read by a step started at ``started_ns``.

        A step that buffered its output read them earlier, so this is called once
        per step that read input since the last output. Times are
        ``time.monotonic_ns()``. The key is the one its result carries, which is
        how presentation identifies the step.
        """
        del step, received_ns, started_ns

    def input_dropped(self, received_ns: Sequence[int]) -> None:
        """Note that the inputs that reached the runtime at ``received_ns`` will
        never be shown, because a reset discarded the output they went into."""
        del received_ns

    def step_presented(self, step: tuple[int, int]) -> None:
        """Note that the first frame of the step keyed ``(generation, step_index)``
        was picked to show."""
        del step

    def collect_input_latency_ms(self) -> dict[str, float]:
        """Return input latency since the previous call and over the session.

        Empty when the backend does not measure it.
        """
        return {}

    def reset_counts(self) -> None:
        """Forget events so far. One profiler serves a run's whole session
        sequence, and a replacement session must not inherit the last one's."""


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
    another's ``generate`` and ``finalize`` does not mix their stages. State is
    per-thread, so concurrent sessions do not interleave one another's stages.
    """

    def __init__(self) -> None:
        self._state = threading.local()

    def _timers(self) -> dict[int | None, tuple[EventProfiler, set[str]]]:
        timers = getattr(self._state, "timers", None)
        if timers is None:
            timers = self._state.timers = {}
        return timers

    @contextmanager
    def stage_scope(self, owner: object) -> Iterator[None]:
        previous = getattr(self._state, "owner", None)
        self._state.owner = id(owner)
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
        owner = getattr(self._state, "owner", None)
        timer = timers.get(owner)
        if timer is None or stage in timer[1]:
            # Either the first stage of a step, or a step that raised before
            # finalize collected it. Start a fresh one rather than assert.
            timer = timers[owner] = (EventProfiler(), set())
        events, recorded = timer
        try:
            yield
        finally:
            events.record(stage)
            recorded.add(stage)

    def collect_stage_ms(self) -> dict[str, float]:
        timer = self._timers().pop(getattr(self._state, "owner", None), None)
        if timer is None:
            return {}
        return timer[0].sync_and_summarize()


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


_INFERENCE_PROFILER: ContextVar[IProfiler | None] = ContextVar(
    "flashdreams_inference_profiler", default=None
)
_DEFAULT_PROFILER: IProfiler | None = None
_DEFAULT_LOCK = threading.Lock()


def get_inference_profiler() -> IProfiler:
    """Return the profiler the inference API runs under.

    An application sets one for its sessions. Code that runs a pipeline without
    a session -- the batch runner, serving, benchmarks -- falls back to what the
    environment asks for, which is what it got before this interface existed.
    """
    profiler = _INFERENCE_PROFILER.get()
    if profiler is not None:
        return profiler
    global _DEFAULT_PROFILER
    if _DEFAULT_PROFILER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_PROFILER is None:
                _DEFAULT_PROFILER = create_profiler()
    return _DEFAULT_PROFILER


def _reset_default_profiler() -> None:
    """Drop the cached environment default. For tests that change the env."""
    global _DEFAULT_PROFILER
    _DEFAULT_PROFILER = None


@contextmanager
def set_flashdreams_inference_profiler(profiler: IProfiler) -> Iterator[IProfiler]:
    """Run the block with ``profiler`` as the inference API's profiler."""
    token = _INFERENCE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _INFERENCE_PROFILER.reset(token)


def bind_inference_profiler(profiler: IProfiler) -> Token[IProfiler | None]:
    """Bind ``profiler`` for the calling thread, which keeps its own context.

    A ``ContextVar`` set on one thread is invisible to another, so the model
    thread binds the profiler the session injected into its loop. Pass the token
    back to :func:`unbind_inference_profiler`: a loop driven on a caller's thread
    would otherwise leave the binding behind.
    """
    return _INFERENCE_PROFILER.set(profiler)


def unbind_inference_profiler(token: Token[IProfiler | None]) -> None:
    """Restore whatever the context held before the matching bind."""
    _INFERENCE_PROFILER.reset(token)


def create_profiler() -> IProfiler:
    """Build the profiler a session runs with, from the environment.

    ``FLASHDREAMS_NVTX`` adds Nsight Systems ranges,
    ``FLASHDREAMS_SYNC_AND_PROFILE`` adds the CUDA-event stage timings that
    ``finalize`` returns, ``FLASHDREAMS_FPS`` adds the frame rates the model loop
    reports, and ``FLASHDREAMS_INPUT_LATENCY`` adds input latency beside them.
    None set means nothing is recorded.
    """
    profilers: list[IProfiler] = []
    if os.environ.get("FLASHDREAMS_NVTX") == "1" and torch.cuda.is_available():
        profilers.append(NVTXProfiler())
    if os.environ.get("FLASHDREAMS_SYNC_AND_PROFILE") == "1":
        profilers.append(CudaEventProfiler())
    if os.environ.get("FLASHDREAMS_FPS") == "1":
        profilers.append(FrameRateProfiler())
    if os.environ.get("FLASHDREAMS_INPUT_LATENCY") == "1":
        profilers.append(InputLatencyProfiler())
    if not profilers:
        return NullProfiler()
    if len(profilers) == 1:
        return profilers[0]
    return CompositeProfiler(tuple(profilers))


class EventProfiler:
    """Times stages between a start event and one ``record(stage)`` per stage.

    Examples:

        profiler = EventProfiler()
        run_encoder()
        profiler.record("encode")
        run_diffusion()
        profiler.record("diffuse")
        run_decoder()
        profiler.record("decode")

        stages_ms = profiler.sync_and_summarize()
        # {"encode": 12.3, "diffuse": 102.4, "decode": 45.6}
    """

    def __init__(
        self,
        *,
        synchronize_distributed: bool = False,
        device: torch.device | str | int | None = None,
    ) -> None:
        """Start a CUDA-event profile.

        Distributed synchronization is opt-in. Implicit barriers in profiling
        code can interleave with model collectives and turn ordinary rank skew
        (notably first-use compilation) into a distributed hang.

        Args:
            synchronize_distributed: Synchronize distributed ranks before timing.
            device: CUDA device whose current stream is measured. ``None`` keeps
                the legacy behavior of using the caller's current CUDA device.
        """
        self._device = device
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if synchronize_distributed and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._start = torch.cuda.Event(enable_timing=True)
        self._ends: dict[str, torch.cuda.Event] = {}
        self._start.record(torch.cuda.current_stream(device))

    def record(self, stage: str) -> None:
        """Record an end-of-stage event under ``stage`` (must be unique)."""
        assert stage not in self._ends, f"stage {stage!r} already recorded"
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self._device))
        self._ends[stage] = event

    def elapsed_ms(self) -> dict[str, float]:
        """Return ``{stage: ms}`` in record order (no sync)."""
        prev = self._start
        out: dict[str, float] = {}
        for stage, end in self._ends.items():
            out[stage] = prev.elapsed_time(end)
            prev = end
        return out

    def sync_and_summarize(self) -> dict[str, float]:
        """``torch.cuda.synchronize()`` then return ``elapsed_ms``."""
        torch.cuda.synchronize(self._device)
        return self.elapsed_ms()

    @staticmethod
    def format_result_as_ms(
        stats_ms: dict[str, float],
        *,
        collect_totals: bool = False,
        collect_vram_info: bool = True,
    ) -> dict[str, float]:
        """Alias for :func:`format_result_as_ms`, kept for pipelines that call it."""
        return format_result_as_ms(
            stats_ms,
            collect_totals=collect_totals,
            collect_vram_info=collect_vram_info,
        )


def record_event(profiler: EventProfiler | None, stage: str) -> None:
    """Record ``stage`` on ``profiler`` when it is not ``None``; no-op otherwise."""
    if profiler is not None:
        profiler.record(stage)


def format_result_as_ms(
    stats_ms: dict[str, float],
    *,
    collect_totals: bool = False,
    collect_vram_info: bool = True,
) -> dict[str, float]:
    """Format stage durations as millisecond metrics.

    Args:
        stats_ms: Stage durations keyed by stage name.
        collect_totals: Whether to include totals with and without finalize.
        collect_vram_info: Whether to include CUDA memory usage when available.

    Returns:
        Stage durations keyed by stage name with an ``_ms`` suffix, optional
        totals, and CUDA memory usage in GiB.
    """
    result = {f"{stage}_ms": ms for stage, ms in stats_ms.items()}
    if collect_totals:
        total_ms = sum(stats_ms.values())
        result["total_ms"] = total_ms
        result["total_ms_wo_finalize"] = total_ms - stats_ms.get("finalize", 0.0)
    if collect_vram_info and torch.cuda.is_available():
        device = torch.cuda.current_device()
        gib = 1024**3
        result["mem_alloc_gib"] = torch.cuda.memory_allocated(device) / gib
        result["mem_reserved_gib"] = torch.cuda.memory_reserved(device) / gib
        result["mem_peak_gib"] = torch.cuda.max_memory_allocated(device) / gib
    return result
