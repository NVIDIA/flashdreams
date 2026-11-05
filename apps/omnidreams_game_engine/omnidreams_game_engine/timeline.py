# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded gameplay history and transactional generation cursors."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, fields, replace
from itertools import islice
from typing import Any

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
        # Split tracks wherever recorded presence ends, including reappearances.
        series: list[list[tuple[DynamicActorTrajectory, int]]] = []
        active: dict[str, list[tuple[DynamicActorTrajectory, int]]] = {}
        for path in paths:
            present = {}
            timestamp = int(path.timestamps_us[0])
            for actor in path.dynamic_actors:
                if not actor.timestamps_us[0] <= timestamp <= actor.timestamps_us[-1]:
                    continue
                segment = active.get(actor.entity_id)
                if segment is None:
                    segment = []
                    series.append(segment)
                segment.append((actor, timestamp))
                present[actor.entity_id] = segment
            active = present
        tracks = []
        seen = set()
        for series_index, segment in enumerate(series):
            timestamps = np.concatenate([actor.timestamps_us for actor, _ in segment])
            _, reversed_indices = np.unique(timestamps[::-1], return_index=True)
            indices = len(timestamps) - 1 - reversed_indices
            actor = segment[-1][0]
            entity_id = actor.entity_id
            if entity_id in seen:
                entity_id = f"{entity_id}/presence-{series_index}"
            seen.add(actor.entity_id)
            track = replace(
                actor,
                entity_id=entity_id,
                timestamps_us=timestamps[indices],
                translations_world=np.concatenate(
                    [a.translations_world for a, _ in segment]
                )[indices],
                orientations_xyzw=np.concatenate(
                    [a.orientations_xyzw for a, _ in segment]
                )[indices],
            )
            tracks.append(_clip_actor_track(track, segment[0][1], segment[-1][1]))
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


def _clip_actor_track(
    actor: DynamicActorTrajectory, start: int, end: int
) -> DynamicActorTrajectory:
    timestamps = actor.timestamps_us
    if start == timestamps[0] and end == timestamps[-1]:
        return actor
    clipped = np.unique(
        np.concatenate(
            ([start, end], timestamps[(timestamps >= start) & (timestamps <= end)])
        )
    )
    upper = (
        np.searchsorted(timestamps, clipped, side="right").clip(1, len(timestamps) - 1)
        if len(timestamps) > 1
        else np.zeros(len(clipped), dtype=int)
    )
    lower = np.maximum(upper - 1, 0)
    span = timestamps[upper] - timestamps[lower]
    fraction = (clipped - timestamps[lower]) / np.maximum(span, 1)
    positions = (
        actor.translations_world[lower] * (1 - fraction[:, None])
        + actor.translations_world[upper] * fraction[:, None]
    )
    q0 = actor.orientations_xyzw[lower]
    q1 = actor.orientations_xyzw[upper].copy()
    dot = np.sum(q0 * q1, axis=1)
    q1[dot < 0] *= -1
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    near = sine < 1e-6
    divisor = np.where(near, 1, sine)
    a = np.where(near, 1 - fraction, np.sin((1 - fraction) * angle) / divisor)
    b = np.where(near, fraction, np.sin(fraction * angle) / divisor)
    orientations = a[:, None] * q0 + b[:, None] * q1
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    return replace(
        actor,
        timestamps_us=clipped,
        translations_world=positions.astype(np.float32),
        orientations_xyzw=orientations.astype(np.float32),
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
