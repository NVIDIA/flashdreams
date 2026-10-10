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

"""The profiling interface the runtime and the pipeline record through."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from contextlib import nullcontext
from typing import ContextManager


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
