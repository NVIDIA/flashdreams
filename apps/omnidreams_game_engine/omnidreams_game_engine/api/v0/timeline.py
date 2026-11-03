# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ticks, model frames and chunks of one epoch.

An epoch starts at its seed tick ``t0``. Model frame ``g`` belongs to engine
tick ``t0 + frame_ticks * g``; frame 0 is the seed. Chunk ``k`` holds the
frames the pipeline generates in autoregressive step ``k``: with OmniDreams
5 frames in chunk 0 (the seed included) and 8 in every later chunk. The plan
is taken from the pipeline (``get_num_output_frames``), not hard-coded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

_PROBE_CHUNKS = 16


@dataclass(frozen=True, slots=True)
class ChunkTimeline:
    """Frame and tick arithmetic shared by the host and the engine."""

    plan: tuple[int, ...] = (5, 8)
    """Frames per chunk; the last entry repeats for every later chunk."""
    sim_hz: int = 60
    model_fps: int = 30

    def __post_init__(self) -> None:
        if not self.plan or any(int(n) <= 0 for n in self.plan):
            raise ValueError("A chunk plan needs positive frame counts")
        if self.sim_hz <= 0 or self.model_fps <= 0:
            raise ValueError("Rates must be positive")
        if self.sim_hz % self.model_fps:
            raise ValueError(
                f"sim_hz {self.sim_hz} is not a multiple of model_fps {self.model_fps}"
            )
        object.__setattr__(self, "plan", tuple(int(n) for n in self.plan))

    @classmethod
    def from_frame_count(
        cls,
        frame_count: Callable[[int], int],
        *,
        sim_hz: int = 60,
        model_fps: int = 30,
    ) -> ChunkTimeline:
        """Build the plan from a pipeline's ``get_num_output_frames``.

        The counts of the first chunks are probed; they must settle on one
        repeating count, as every autoregressive pipeline in the repository does.
        """
        counts = [int(frame_count(k)) for k in range(_PROBE_CHUNKS)]
        last = counts[-1]
        settle = len(counts) - 1
        while settle > 0 and counts[settle - 1] == last:
            settle -= 1
        return cls(plan=tuple(counts[: settle + 1]), sim_hz=sim_hz, model_fps=model_fps)

    @property
    def frame_ticks(self) -> int:
        """Engine ticks per model frame."""
        return self.sim_hz // self.model_fps

    @property
    def plan_list(self) -> list[int]:
        """The plan as sent in ``config.chunk_plan``."""
        return list(self.plan)

    def frames_in_chunk(self, chunk: int) -> int:
        if chunk < 0:
            raise ValueError("Chunk indices start at 0")
        return self.plan[min(chunk, len(self.plan) - 1)]

    def first_frame(self, chunk: int) -> int:
        """Model frame index of the first frame of ``chunk``."""
        if chunk < 0:
            raise ValueError("Chunk indices start at 0")
        if chunk < len(self.plan):
            return sum(self.plan[:chunk])
        head = sum(self.plan)
        return head + (chunk - len(self.plan)) * self.plan[-1]

    def last_frame(self, chunk: int) -> int:
        return self.first_frame(chunk) + self.frames_in_chunk(chunk) - 1

    def chunk_frames(self, chunk: int) -> range:
        first = self.first_frame(chunk)
        return range(first, first + self.frames_in_chunk(chunk))

    def chunk_of(self, g: int) -> int:
        """Chunk that contains model frame ``g``."""
        if g < 0:
            raise ValueError("Frame indices start at 0")
        start = 0
        for chunk, count in enumerate(self.plan):
            if g < start + count:
                return chunk
            start += count
        return len(self.plan) + (g - start) // self.plan[-1]

    def tick_of(self, g: int, t0: int) -> int:
        """Engine tick of model frame ``g`` in an epoch seeded at ``t0``."""
        return t0 + self.frame_ticks * g

    def seal_tick(self, chunk: int, t0: int) -> int:
        """Tick whose state completes ``chunk``."""
        return self.tick_of(self.last_frame(chunk), t0)

    @property
    def frame_interval_us(self) -> int:
        """Model frame interval, rounded as the in-process engine rounds it."""
        return round(1_000_000 / self.model_fps)

    def timestamp_us(self, tick: int, *, t0: int, base_us: int) -> int:
        """Model timestamp of an engine tick: ``base_us`` at the seed tick.

        Frame ``g`` (tick ``t0 + frame_ticks * g``) is at
        ``base_us + g * frame_interval_us``, the convention of the in-process
        engine, so both produce the same timestamps; a tick between frames is
        interpolated.
        """
        return base_us + round((tick - t0) * self.frame_interval_us / self.frame_ticks)

    def grant_frames(self, through_chunk: int, *, after_g: int) -> Sequence[int]:
        """Model frames up to the end of ``through_chunk`` not yet granted."""
        return range(after_g + 1, self.last_frame(through_chunk) + 1)


__all__ = ["ChunkTimeline"]
