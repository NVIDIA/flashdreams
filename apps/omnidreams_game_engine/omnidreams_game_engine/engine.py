# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-thread sequencing for simulation, rules, and conditioning."""

from __future__ import annotations

import time

import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from omnidreams_game_engine.contracts import (
    ConditionRenderer,
    GameRules,
    SimulationWorld,
)
from omnidreams_game_engine.types import ConditionBatch, DriverCommand, TrajectoryChunk


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """Recorded physics and rules, independent of model rendering."""

    trajectory: TrajectoryChunk
    game_frames: tuple[object, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineStep:
    """All synchronized non-model results for one autoregressive step."""

    trajectory: TrajectoryChunk
    game_frames: tuple[object, ...]
    condition: ConditionBatch
    metrics: Mapping[str, float] = field(default_factory=dict)
    """Wall and model-thread CPU timings for the engine's major stages."""


class GameEngine:
    """Advance one session's mutable game state on the model thread."""

    def __init__(
        self,
        *,
        simulation: SimulationWorld,
        rules: GameRules,
        condition_renderer: ConditionRenderer | None,
        frame_interval_s: float,
    ) -> None:
        if frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be positive")
        self.simulation = simulation
        self.rules = rules
        self.condition_renderer = condition_renderer
        self.frame_interval_s = float(frame_interval_s)
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

    def step(self, commands: tuple[DriverCommand, ...]) -> EngineStep:
        """Advance and render one model block on the owning loop."""
        return self.render(self.advance(commands))

    def advance(self, commands: tuple[DriverCommand, ...]) -> SimulationStep:
        """Advance fixed physics ticks and rules without touching the renderer.

        One thread must own this method and all other mutable simulation access.
        The returned observations may be handed to a different render thread.
        """
        if self._closed:
            raise RuntimeError("GameEngine is closed")
        if not commands:
            raise ValueError("A simulation step requires at least one command")
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()
        trajectory = self.simulation.pose_chunk(
            commands=commands,
            chunk_size=len(commands),
            frame_interval_s=self.frame_interval_s,
            extrapolation_offset_s=0.0,
        )
        simulation_wall_ms = (time.perf_counter() - wall_started) * 1000
        simulation_cpu_ms = (time.thread_time() - cpu_started) * 1000
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()
        update = self.rules.advance_frames(trajectory, self.frame_interval_s)
        if len(update.frames) != len(commands):
            raise ValueError("Game frames must align with simulated commands")
        return SimulationStep(
            trajectory=replace(
                trajectory,
                dynamic_actors=(*trajectory.dynamic_actors, *update.dynamic_actors),
            ),
            game_frames=update.frames,
            metrics={
                "physics_steps": float(len(commands)),
                "simulation_duration_ms": len(commands) * self.frame_interval_s * 1000,
                "simulation_wall_ms": simulation_wall_ms,
                "simulation_cpu_ms": simulation_cpu_ms,
                "rules_wall_ms": (time.perf_counter() - wall_started) * 1000,
                "rules_cpu_ms": (time.thread_time() - cpu_started) * 1000,
            },
        )

    def render(self, recorded: SimulationStep) -> EngineStep:
        """Render a frozen world observation without advancing gameplay."""
        if self._closed or self.condition_renderer is None:
            raise RuntimeError("GameEngine has no active condition renderer")
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()
        condition = self.condition_renderer.render(recorded.trajectory)
        wall_ms = (time.perf_counter() - wall_started) * 1000
        cpu_ms = (time.thread_time() - cpu_started) * 1000
        if int(condition.hdmap_bvtchw.shape[2]) != len(recorded.game_frames):
            raise ValueError("Condition frames must align with simulated commands")
        metrics = dict(recorded.metrics)
        metrics.update(
            conditioning_wall_ms=wall_ms,
            conditioning_cpu_ms=cpu_ms,
            engine_step_wall_ms=wall_ms
            + sum(metrics.get(k, 0) for k in ("simulation_wall_ms", "rules_wall_ms")),
            engine_step_cpu_ms=cpu_ms
            + sum(metrics.get(k, 0) for k in ("simulation_cpu_ms", "rules_cpu_ms")),
        )
        return EngineStep(recorded.trajectory, recorded.game_frames, condition, metrics)

    def close(self) -> None:
        """Release session-local physics and renderer resources."""
        if self._closed:
            return
        self._closed = True
        try:
            self.simulation.close()
        finally:
            if self.condition_renderer is not None:
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
