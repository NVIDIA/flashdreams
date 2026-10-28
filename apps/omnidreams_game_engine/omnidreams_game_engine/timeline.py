# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded gameplay history and transactional generation cursors."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, fields, replace
from typing import Any
from itertools import islice

import numpy as np

from omnidreams_game_engine.engine import SimulationStep, _sample_trajectory
from omnidreams_game_engine.types import DynamicActorTrajectory, PhysXChunkTimings


@dataclass(frozen=True, slots=True)
class WorldSample:
    """One authoritative physics tick and its application presentation state."""

    step: SimulationStep
    presentation: Any = None
    transition_timestamp_us: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationWindow:
    """Frozen recorded path selected for one model chunk."""

    start_tick: int
    end_tick: int
    samples: tuple[WorldSample, ...]
    indices: tuple[int, ...]

    def transition_timestamps_us(self) -> tuple[int | None, ...]:
        """Associate intervening input transitions with the next generated frame."""
        result = []
        start = 0
        for end in self.indices:
            result.append(
                next(
                    (
                        s.transition_timestamp_us
                        for s in reversed(self.samples[start : end + 1])
                        if s.transition_timestamp_us is not None
                    ),
                    None,
                )
            )
            start = end + 1
        return tuple(result)

    def recorded_step(self) -> SimulationStep:
        """Keep intervening collisions and actor tracks when sampling the path."""
        steps = tuple(sample.step for sample in self.samples)
        paths = tuple(step.trajectory for step in steps)
        actors: dict[str, list[DynamicActorTrajectory]] = {}
        for path in paths:
            for actor in path.dynamic_actors:
                actors.setdefault(actor.entity_id, []).append(actor)
        tracks = []
        for series in actors.values():
            timestamps = np.concatenate([actor.timestamps_us for actor in series])
            # Visual-only actors can repeat their complete authored track. Keep
            # the latest observation at each timestamp, in chronological order.
            _, reversed_indices = np.unique(timestamps[::-1], return_index=True)
            indices = len(timestamps) - 1 - reversed_indices
            tracks.append(
                replace(
                    series[-1],
                    timestamps_us=timestamps[indices],
                    translations_world=np.concatenate(
                        [a.translations_world for a in series]
                    )[indices],
                    orientations_xyzw=np.concatenate(
                        [a.orientations_xyzw for a in series]
                    )[indices],
                )
            )
        timings = [
            path.physx_timings for path in paths if path.physx_timings is not None
        ]
        totals = (
            None
            if not timings
            else PhysXChunkTimings(
                **{
                    field.name: (max if field.name.startswith("max_") else sum)(
                        getattr(timing, field.name) for timing in timings
                    )
                    for field in fields(PhysXChunkTimings)
                }
            )
        )
        actor_collision = next(
            (i for i, p in enumerate(paths) if p.actor_collision_detected), None
        )
        static_collision = next(
            (i for i, p in enumerate(paths) if p.static_collision_detected), None
        )
        trajectory = replace(
            paths[-1],
            timestamps_us=np.concatenate([p.timestamps_us for p in paths]),
            rig_poses_world=np.concatenate([p.rig_poses_world for p in paths]),
            vehicle_states=tuple(p.vehicle_states[0] for p in paths),
            applied_commands=tuple(p.applied_commands[0] for p in paths),
            physics_debug_frames=tuple(
                f for p in paths for f in p.physics_debug_frames
            ),
            dynamic_actors=tuple(tracks),
            actor_collision_detected=actor_collision is not None,
            actor_collision_frame_index=actor_collision,
            static_collision_detected=static_collision is not None,
            static_collision_frame_index=static_collision,
            physx_elapsed_s=sum(p.physx_elapsed_s or 0 for p in paths),
            physx_timings=totals,
        )
        return SimulationStep(
            _sample_trajectory(trajectory, self.indices),
            tuple(steps[i].game_frames[0] for i in self.indices),
            {
                key: sum(step.metrics.get(key, 0) for step in steps)
                for key in steps[0].metrics
            },
        )


class WorldTimeline:
    """Single-producer history with an independently committed render cursor.

    Overflow fails explicitly: silently dropping a path would make the next
    condition jump away from the world already represented by the model cache.
    """

    def __init__(self, *, ticks_per_frame: int = 4, capacity: int = 3600) -> None:
        if ticks_per_frame < 1 or capacity < ticks_per_frame:
            raise ValueError("Timeline needs positive sampling and history capacity")
        self.ticks_per_frame = ticks_per_frame
        self.capacity = capacity
        self._samples: deque[WorldSample] = deque()
        self._condition = threading.Condition()
        self.gameplay_tick = -1
        self.generation_tick = -1
        self._pending: GenerationWindow | None = None
        self._closed = False
        self.finished = False
        self._failure: BaseException | None = None
        self.latest: WorldSample | None = None

    def append(self, sample: WorldSample) -> None:
        """Publish one tick; never overwrite an unrendered path."""
        if len(sample.step.game_frames) != 1:
            raise ValueError("World history requires one physics tick per sample")
        with self._condition:
            if self._closed:
                return
            if len(self._samples) >= self.capacity:
                raise RuntimeError("Generation exceeded the gameplay history capacity")
            self._samples.append(sample)
            self.latest = sample
            self.gameplay_tick += 1
            self._condition.notify_all()

    def prepare(self, count: int, stop: threading.Event) -> GenerationWindow | None:
        """Sample the recorded path through the latest available gameplay tick.

        Wait for a nominal chunk when generation is faster than gameplay. When
        slower, represent all elapsed gameplay in the next chunk: a fixed speed
        limit would accumulate unbounded input-to-video delay. The seed keeps
        its original cadence and initial pose.
        """
        if count < 1:
            raise ValueError("Invalid generation sample count")
        with self._condition:
            if self._pending is not None:
                raise RuntimeError("Commit the pending generation window first")
            first = self.generation_tick < 0
            start = self.generation_tick + 1
            intervals = count - int(first)
            nominal_end = (
                0 if first else self.generation_tick
            ) + intervals * self.ticks_per_frame
            while self.gameplay_tick < nominal_end and not self.finished:
                if self._failure is not None:
                    raise RuntimeError("Gameplay simulation failed") from self._failure
                if self._closed or stop.is_set():
                    return None
                self._condition.wait(0.01)
            if self._failure is not None:
                raise RuntimeError("Gameplay simulation failed") from self._failure
            if self._closed or stop.is_set() or not self._samples:
                return None
            # Resample the entire intervening path, including collisions, rather
            # than jumping straight to the latest pose. No future state is used.
            # ponytail: slow generation increases distance between model frames;
            # preserving both nominal motion and low latency needs faster output.
            span = intervals * self.ticks_per_frame
            if not first:
                span = self.gameplay_tick - self.generation_tick
            ticks = (
                tuple(
                    (0 if first else self.generation_tick)
                    + round((i + 1) * span / intervals)
                    for i in range(intervals)
                )
                if intervals
                else ()
            )
            if first:
                ticks = (0, *ticks)
            if self.finished:
                ticks = tuple(min(tick, self.gameplay_tick) for tick in ticks)
            end = ticks[-1]
            samples = tuple(islice(self._samples, end - start + 1))
            self._pending = GenerationWindow(
                start, end, samples, tuple(t - start for t in ticks)
            )
            return self._pending

    def commit(self, window: GenerationWindow) -> None:
        """Advance only after a chunk has generated and finalized successfully."""
        with self._condition:
            if window is not self._pending:
                raise ValueError("Generation window is not the pending transaction")
            for _ in window.samples:
                self._samples.popleft()
            self.generation_tick = window.end_tick
            self._pending = None

    def finish(self) -> None:
        """Freeze gameplay; the final generation window repeats its last sample."""
        with self._condition:
            self.finished = True
            self._condition.notify_all()

    def close(self, failure: BaseException | None = None) -> None:
        """Wake waiters during shutdown or simulation failure."""
        with self._condition:
            self._closed = True
            if failure is not None:
                self._failure = failure
            self._condition.notify_all()
