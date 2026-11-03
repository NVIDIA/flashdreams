# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-frame engine states collected into the chunks the model generates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from omnidreams_game_engine.api.v0.state import ActorState, FrameState
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline
from omnidreams_game_engine.types import DynamicActorTrajectory, TrajectoryChunk

TRACK_SEPARATOR = "~"
"""Joins an actor id and its track number when a gap splits its track."""


@dataclass(frozen=True, slots=True)
class ExternalChunk:
    """The engine states of one autoregressive step, in frame order."""

    epoch: int
    ar_index: int
    t0_tick: int
    """Seed tick of the epoch."""
    frames: tuple[FrameState, ...]

    @property
    def g_range(self) -> range:
        """Model frame indices of the chunk."""
        return range(self.frames[0].g, self.frames[-1].g + 1)

    @property
    def ticks(self) -> tuple[int, ...]:
        """Engine tick of every frame."""
        return tuple(frame.tick for frame in self.frames)

    def timestamps_us(
        self, timeline: ChunkTimeline, *, base_timestamp_us: int
    ) -> np.ndarray:
        """Return the model timestamp of every frame."""
        return np.asarray(
            [
                timeline.timestamp_us(
                    frame.tick, t0=self.t0_tick, base_us=base_timestamp_us
                )
                for frame in self.frames
            ],
            dtype=np.int64,
        )

    def to_trajectory(
        self, timeline: ChunkTimeline, *, base_timestamp_us: int
    ) -> TrajectoryChunk:
        """Build the trajectory the condition renderer draws."""
        timestamps = self.timestamps_us(timeline, base_timestamp_us=base_timestamp_us)
        poses = np.stack(
            [np.asarray(frame.rig_to_world, dtype=np.float32) for frame in self.frames]
        )
        return TrajectoryChunk(
            timestamps_us=timestamps,
            rig_poses_world=poses,
            vehicle_states=tuple(frame.vehicle for frame in self.frames),
            boundary_state_after_chunk=self.frames[-1].vehicle,
            applied_commands=tuple(frame.command for frame in self.frames),
            dynamic_actors=actor_tracks(self.frames, timestamps),
        )


def actor_tracks(
    frames: tuple[FrameState, ...], timestamps_us: np.ndarray
) -> tuple[DynamicActorTrajectory, ...]:
    """Return one trajectory per actor and contiguous run of frames.

    An actor missing from some frames of a chunk is split at the gap; the
    second and later runs get ``<id>~<n>``. Tracks are marked simulated so the
    renderer rebuilds their boxes every chunk.
    """
    runs: dict[str, list[list[tuple[int, ActorState]]]] = {}
    for index, frame in enumerate(frames):
        for actor in frame.actors:
            tracks = runs.setdefault(actor.entity_id, [])
            if tracks and tracks[-1][-1][0] == index - 1:
                tracks[-1].append((index, actor))
            else:
                tracks.append([(index, actor)])
    result: list[DynamicActorTrajectory] = []
    for entity_id in sorted(runs):
        for number, track in enumerate(runs[entity_id]):
            indices = [index for index, _ in track]
            actors = [actor for _, actor in track]
            first = actors[0]
            result.append(
                DynamicActorTrajectory(
                    entity_id=(
                        entity_id
                        if number == 0
                        else f"{entity_id}{TRACK_SEPARATOR}{number}"
                    ),
                    object_type=first.object_type,
                    timestamps_us=np.asarray(timestamps_us[indices], dtype=np.int64),
                    translations_world=np.stack(
                        [np.asarray(a.position_world, dtype=np.float32) for a in actors]
                    ),
                    orientations_xyzw=np.stack(
                        [
                            np.asarray(a.orientation_xyzw, dtype=np.float32)
                            for a in actors
                        ]
                    ),
                    dimensions_lwh=np.asarray(first.dimensions_lwh, dtype=np.float32),
                    is_simulated=True,
                )
            )
    return tuple(result)


@dataclass(slots=True)
class AssemblerStats:
    """Counts of the frames the assembler received."""

    added: int = 0
    duplicates: int = 0
    stale: int = 0
    """Frames of another epoch or of a chunk already taken."""
    chunks: int = 0


@dataclass(slots=True)
class ChunkAssembler:
    """Frames of the current epoch keyed by model frame index."""

    timeline: ChunkTimeline
    epoch: int | None = None
    t0_tick: int | None = None
    stats: AssemblerStats = field(default_factory=AssemblerStats)
    _frames: dict[int, FrameState] = field(default_factory=dict)
    _taken_through: int = -1
    """Last chunk handed out in this epoch."""

    def begin_epoch(self, epoch: int, timeline: ChunkTimeline | None = None) -> None:
        """Start a new epoch (a discontinuity): earlier frames are dropped."""
        self.epoch = epoch
        if timeline is not None:
            self.timeline = timeline
        self.t0_tick = None
        self._frames.clear()
        self._taken_through = -1

    def add(
        self, epoch: int, frame: FrameState
    ) -> Literal["added", "duplicate", "stale"]:
        """Store one frame; return ``added``, ``duplicate`` or ``stale``."""
        if epoch != self.epoch:
            self.stats.stale += 1
            return "stale"
        if frame.g < 0 or self.timeline.chunk_of(frame.g) <= self._taken_through:
            self.stats.stale += 1
            return "stale"
        if frame.g in self._frames:
            self.stats.duplicates += 1
            return "duplicate"
        self._frames[frame.g] = frame
        if frame.g == 0:
            self.t0_tick = frame.tick
        self.stats.added += 1
        return "added"

    def missing(self, ar_index: int) -> list[int]:
        """Return the frames of chunk ``ar_index`` not received yet."""
        return [
            g for g in self.timeline.chunk_frames(ar_index) if g not in self._frames
        ]

    def ready(self, ar_index: int) -> bool:
        """Return whether chunk ``ar_index`` is complete and not yet taken."""
        return (
            self.epoch is not None
            and self.t0_tick is not None
            and ar_index > self._taken_through
            and not self.missing(ar_index)
        )

    def take(self, ar_index: int) -> ExternalChunk | None:
        """Remove and return chunk ``ar_index`` when all its frames are here."""
        if not self.ready(ar_index):
            return None
        assert self.epoch is not None and self.t0_tick is not None
        frames = tuple(
            self._frames.pop(g) for g in self.timeline.chunk_frames(ar_index)
        )
        self._taken_through = ar_index
        for g in [g for g in self._frames if self.timeline.chunk_of(g) <= ar_index]:
            del self._frames[g]
        self.stats.chunks += 1
        return ExternalChunk(
            epoch=self.epoch,
            ar_index=ar_index,
            t0_tick=self.t0_tick,
            frames=frames,
        )


__all__ = [
    "TRACK_SEPARATOR",
    "AssemblerStats",
    "ChunkAssembler",
    "ExternalChunk",
    "actor_tracks",
]
