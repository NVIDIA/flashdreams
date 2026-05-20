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

"""CUDA-event timer."""

from __future__ import annotations

import torch


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

    def __init__(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._start = torch.cuda.Event(enable_timing=True)
        self._ends: dict[str, torch.cuda.Event] = {}
        self._start.record()

    def record(self, stage: str) -> None:
        """Record an end-of-stage event under ``stage`` (must be unique)."""
        assert stage not in self._ends, f"stage {stage!r} already recorded"
        event = torch.cuda.Event(enable_timing=True)
        event.record()
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
        torch.cuda.synchronize()
        return self.elapsed_ms()


def record_event(profiler: EventProfiler | None, stage: str) -> None:
    """Record ``stage`` on ``profiler`` when it is not ``None``; no-op otherwise."""
    if profiler is not None:
        profiler.record(stage)
