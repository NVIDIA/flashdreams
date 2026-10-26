# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-thread sequencing for simulation, rules, and conditioning."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from omnidreams_game_engine.contracts import (
    ConditionRenderer,
    GameRules,
    SimulationWorld,
)
from omnidreams_game_engine.types import ConditionBatch, DriverCommand, TrajectoryChunk


@dataclass(frozen=True, slots=True)
class EngineStep:
    """All synchronized non-model results for one autoregressive step."""

    trajectory: TrajectoryChunk
    game_frames: tuple[object, ...]
    condition: ConditionBatch
    metrics: Mapping[str, float] = field(default_factory=dict)
    """Wall and model-thread CPU timings for the engine's major stages."""

    simulation_frame_indices: tuple[int, ...] = ()
    """Physics samples selected for conditioning and frame-aligned overlays."""


class GameEngine:
    """Advance one session's mutable game state on the model thread."""

    def __init__(
        self,
        *,
        simulation: SimulationWorld,
        rules: GameRules,
        condition_renderer: ConditionRenderer,
        frame_interval_s: float,
        physics_frame_interval_s: float | None = None,
        preserve_initial_frame: bool = False,
    ) -> None:
        if not math.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be positive")
        self.simulation = simulation
        self.rules = rules
        self.condition_renderer = condition_renderer
        self.frame_interval_s = float(frame_interval_s)
        self.physics_frame_interval_s = (
            frame_interval_s
            if physics_frame_interval_s is None
            else physics_frame_interval_s
        )
        if (
            not math.isfinite(self.physics_frame_interval_s)
            or not 0 < self.physics_frame_interval_s <= frame_interval_s
        ):
            raise ValueError(
                "Physics interval must be positive and no longer than a frame"
            )
        self._preserve_initial_frame = preserve_initial_frame
        self._remaining_ticks = 0.0
        self._closed = False

    @property
    def is_running(self) -> bool:
        """Return whether rules accept another generated block."""
        return not self._closed and self.rules.is_running

    @property
    def current_game_frame(self) -> object:
        """Return game state coherent with the current vehicle boundary."""
        return self.rules.snapshot(self.simulation.current_state)

    def submit_text(self, value: str) -> object:
        """Forward application text at the current simulation boundary."""
        return self.rules.submit_text(value, self.simulation.current_state)

    def step(
        self,
        commands: tuple[DriverCommand, ...],
        *,
        simulation_duration_s: float | None = None,
    ) -> EngineStep:
        """Advance exactly one model block."""
        if self._closed:
            raise RuntimeError("GameEngine is closed")
        if not commands:
            raise ValueError("A game-engine step requires at least one command")

        count = len(commands)
        duration = count * self.frame_interval_s
        if simulation_duration_s is not None:
            if (
                not math.isfinite(simulation_duration_s)
                or simulation_duration_s < duration
            ):
                raise ValueError(
                    "Simulation duration must cover at least the nominal chunk"
                )
            duration = simulation_duration_s
        # The seed image depicts the untouched spawn. Its first sample must not
        # advance physics, even when subsequent frames contain several ticks.
        initial = int(self._preserve_initial_frame)
        if initial and count == 1 and duration > self.frame_interval_s:
            raise ValueError("A seed-only chunk cannot advance simulation time")
        ticks = (
            duration - initial * self.frame_interval_s
        ) / self.physics_frame_interval_s
        ticks += self._remaining_ticks
        whole_ticks = math.floor(ticks + 1e-9)
        self._remaining_ticks = ticks - whole_ticks
        tick_count = whole_ticks + initial
        indices = (
            ((np.arange(1, count - initial + 1) * whole_ticks) // (count - initial))
            + initial
            - 1
            if count > initial
            else np.empty(0, dtype=np.int64)
        )
        if initial:
            indices = np.concatenate(([0], indices))
        repeats = np.diff(np.concatenate(([-1], indices)))
        physics_commands = tuple(
            command
            for command, repeat in zip(commands, repeats, strict=True)
            for _ in range(int(repeat))
        )

        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        simulation_wall_started = time.perf_counter()
        simulation_cpu_started = time.thread_time()
        trajectory = self.simulation.pose_chunk(
            commands=physics_commands,
            chunk_size=tick_count,
            frame_interval_s=self.physics_frame_interval_s,
            extrapolation_offset_s=0.0,
        )
        simulation_wall_ms = (time.perf_counter() - simulation_wall_started) * 1000.0
        simulation_cpu_ms = (time.thread_time() - simulation_cpu_started) * 1000.0

        rules_wall_started = time.perf_counter()
        rules_cpu_started = time.thread_time()
        update = self.rules.advance_frames(trajectory, self.physics_frame_interval_s)
        if len(update.frames) != tick_count:
            raise ValueError("Game frames must align with simulated commands")
        trajectory = replace(
            trajectory,
            dynamic_actors=(*trajectory.dynamic_actors, *update.dynamic_actors),
        )
        selected = tuple(int(index) for index in indices)
        trajectory = _sample_trajectory(trajectory, selected)
        self._preserve_initial_frame = False
        rules_wall_ms = (time.perf_counter() - rules_wall_started) * 1000.0
        rules_cpu_ms = (time.thread_time() - rules_cpu_started) * 1000.0

        conditioning_wall_started = time.perf_counter()
        conditioning_cpu_started = time.thread_time()
        condition = self.condition_renderer.render(trajectory)
        conditioning_wall_ms = (
            time.perf_counter() - conditioning_wall_started
        ) * 1000.0
        conditioning_cpu_ms = (time.thread_time() - conditioning_cpu_started) * 1000.0
        if int(condition.hdmap_bvtchw.shape[2]) != len(commands):
            raise ValueError("Condition frames must align with simulated commands")
        return EngineStep(
            trajectory=trajectory,
            game_frames=tuple(update.frames[index] for index in selected),
            condition=condition,
            simulation_frame_indices=selected,
            metrics={
                "physics_steps": float(tick_count - initial),
                "simulation_duration_ms": tick_count
                * self.physics_frame_interval_s
                * 1000,
                "simulation_wall_ms": simulation_wall_ms,
                "simulation_cpu_ms": simulation_cpu_ms,
                "rules_wall_ms": rules_wall_ms,
                "rules_cpu_ms": rules_cpu_ms,
                "conditioning_wall_ms": conditioning_wall_ms,
                "conditioning_cpu_ms": conditioning_cpu_ms,
                "engine_step_wall_ms": (time.perf_counter() - step_wall_started)
                * 1000.0,
                "engine_step_cpu_ms": (time.thread_time() - step_cpu_started) * 1000.0,
            },
        )

    def close(self) -> None:
        """Release session-local physics and renderer resources."""
        if self._closed:
            return
        self._closed = True
        try:
            self.simulation.close()
        finally:
            self.condition_renderer.close()


def _sample_trajectory(
    trajectory: TrajectoryChunk, indices: tuple[int, ...]
) -> TrajectoryChunk:
    def collision_index(index: int | None) -> int | None:
        return None if index is None else int(np.searchsorted(indices, index))

    return replace(
        trajectory,
        timestamps_us=trajectory.timestamps_us[list(indices)],
        rig_poses_world=trajectory.rig_poses_world[list(indices)],
        vehicle_states=tuple(trajectory.vehicle_states[i] for i in indices),
        applied_commands=tuple(trajectory.applied_commands[i] for i in indices),
        physics_debug_frames=tuple(trajectory.physics_debug_frames[i] for i in indices)
        if trajectory.physics_debug_frames
        else (),
        actor_collision_frame_index=collision_index(
            trajectory.actor_collision_frame_index
        ),
        static_collision_frame_index=collision_index(
            trajectory.static_collision_frame_index
        ),
    )
