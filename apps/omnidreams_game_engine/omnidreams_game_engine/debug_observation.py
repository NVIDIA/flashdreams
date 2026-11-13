# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation observations shared by native and recorded-world rendering."""

import time
from dataclasses import asdict, is_dataclass
from typing import Any

from omnidreams_game_engine.debug_input import HeadlessInput
from omnidreams_game_engine.engine import EngineStep
from omnidreams_game_engine.gameplay import GameplayTiming
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper
from omnidreams_game_engine.timeline import WorldSample


def _game_record(frame: object) -> object:
    return (
        asdict(frame) if is_dataclass(frame) and not isinstance(frame, type) else frame
    )


def simulation_records(
    step: EngineStep,
    *,
    commands,
    transition_timestamps_us,
    frame_start: int,
    rollout_epoch: int,
    model_step: int,
    autoregressive_index: int,
    elapsed_us: int | None,
    reused_video: bool = False,
    ground: GroundSnapper | None = None,
) -> tuple[dict[str, Any], ...]:
    """Record trajectory and optional native physics without inventing missing data."""
    trajectory = step.trajectory
    count = len(trajectory.timestamps_us)
    if len(commands) != count or len(transition_timestamps_us) != count:
        raise ValueError(
            "Observation commands and timestamps must match the trajectory"
        )
    if (
        trajectory.physics_debug_frames
        and len(trajectory.physics_debug_frames) != count
    ):
        raise ValueError("Physics observations must cover every trajectory frame")
    records = []
    for index in range(count):
        physics = (
            trajectory.physics_debug_frames[index]
            if trajectory.physics_debug_frames
            else None
        )
        physical = None if physics is None else physics.vehicle_state
        ground_height = (
            None
            if ground is None or physical is None
            else ground.ground_height_at(physical.x_m, physical.y_m, physical.z_m)
        )
        records.append(
            {
                "frame": frame_start + index,
                "reused_video": reused_video,
                "produced_ns": time.monotonic_ns(),
                "elapsed_us": elapsed_us,
                "transition_timestamp_us": transition_timestamps_us[index],
                "physics_started_ns": None
                if physics is None
                else physics.physics_started_ns,
                "physics_completed_ns": None
                if physics is None
                else physics.physics_completed_ns,
                "rollout_epoch": rollout_epoch,
                "simulation_timestamp_us": int(trajectory.timestamps_us[index]),
                "command": asdict(commands[index]),
                "vehicle": asdict(trajectory.vehicle_states[index]),
                "physics_vehicle": None if physical is None else asdict(physical),
                "rig_pose_world": trajectory.rig_poses_world[index].copy(),
                "ground_height_m": ground_height,
                "ground_error_m": None
                if ground_height is None or physical is None
                else physical.z_m - ground_height,
                "actor_collision": None if physics is None else physics.actor_collision,
                "static_barrier_collision": None
                if physics is None
                else physics.static_barrier_collision,
                "physics_colliders": None if physics is None else asdict(physics),
                "game": _game_record(step.game_frames[index]),
                "model_step": model_step,
                "autoregressive_index": autoregressive_index,
                "chunk_frame": index,
            }
        )
    return tuple(records)


class GameplayCapture:
    """Connect physics-process observations to the headless clock and capture writer."""

    def __init__(self, control: HeadlessInput, epoch: int):
        self.control = control
        self.epoch = epoch

    def start(self) -> float | None:
        self.control.start()
        return None if self.control.ready_ns is None else self.control.ready_ns / 1e9

    def consumed(self, events, *, step: int, consumed_ns: int) -> None:
        self.control.consumed(
            events, step=step, consumed_ns=consumed_ns, consumer="gameplay"
        )

    def sample(self, sample: WorldSample, timing: GameplayTiming) -> None:
        capture = self.control.capture
        if capture is None:
            return
        step = sample.step
        capture.gameplay(
            {
                **asdict(timing),
                "rollout_epoch": self.epoch,
                "simulation_timestamp_us": int(step.trajectory.timestamps_us[0]),
                "transition_timestamp_us": sample.transition_timestamp_us,
                "command": asdict(step.trajectory.applied_commands[0]),
                "vehicle": asdict(step.trajectory.vehicle_states[0]),
                "game": _game_record(step.game_frames[0]),
            }
        )
