# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-thread sequencing for simulation, rules, and conditioning."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

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


class GameEngine:
    """Advance one session's mutable game state on the model thread."""

    def __init__(
        self,
        *,
        simulation: SimulationWorld,
        rules: GameRules,
        condition_renderer: ConditionRenderer,
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
        """Advance exactly one model block."""
        if self._closed:
            raise RuntimeError("GameEngine is closed")
        if not commands:
            raise ValueError("A game-engine step requires at least one command")

        step_wall_started = time.perf_counter()
        step_cpu_started = time.thread_time()
        simulation_wall_started = time.perf_counter()
        simulation_cpu_started = time.thread_time()
        trajectory = self.simulation.pose_chunk(
            commands=commands,
            chunk_size=len(commands),
            frame_interval_s=self.frame_interval_s,
            extrapolation_offset_s=0.0,
        )
        simulation_wall_ms = (time.perf_counter() - simulation_wall_started) * 1000.0
        simulation_cpu_ms = (time.thread_time() - simulation_cpu_started) * 1000.0

        rules_wall_started = time.perf_counter()
        rules_cpu_started = time.thread_time()
        update = self.rules.advance_frames(trajectory, self.frame_interval_s)
        if len(update.frames) != len(commands):
            raise ValueError("Game frames must align with simulated commands")
        trajectory = replace(
            trajectory,
            dynamic_actors=(*trajectory.dynamic_actors, *update.dynamic_actors),
        )
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
            game_frames=update.frames,
            condition=condition,
            metrics={
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
