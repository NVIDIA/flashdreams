# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections.abc import Callable, Iterator

from interactive_drive.backends.base import RenderBackend
from interactive_drive.config import ChunkConfig, VehicleConfig
from interactive_drive.input.keyboard import command_from_snapshot
from interactive_drive.simulation.ego_vehicle_kinematics import (
    build_ground_snapper,
    integrate_vehicle,
    sample_chunk_trajectory,
    state_from_initial_pose,
)
from interactive_drive.types import (
    DriverCommand,
    FrameChunk,
    SceneBundle,
)

__all__ = [
    "command_from_snapshot",
    "integrate_vehicle",
    "iterate_frame_chunks",
    "sample_chunk_trajectory",
    "state_from_initial_pose",
]


def iterate_frame_chunks(
    scene: SceneBundle,
    backend: RenderBackend,
    chunk_config: ChunkConfig,
    vehicle_config: VehicleConfig,
    command_source: Callable[[], DriverCommand],
) -> Iterator[FrameChunk]:
    """Yield rendered chunks from `backend` driven by `command_source`.

    Caller is responsible for calling `backend.warmup(scene)` once before iterating,
    and for stopping iteration (break, islice, etc.). Simulation advances in fixed
    timesteps (`chunk_config.frame_interval_us`) and is fully deterministic given a
    deterministic `command_source`.
    """
    state = state_from_initial_pose(
        initial_rig_to_world=scene.initial_rig_to_world,
        initial_yaw_rad=scene.initial_yaw_rad,
        initial_speed_mps=scene.initial_speed_mps,
    )
    ground_snapper = build_ground_snapper(scene)
    next_timestamp_us = scene.initial_timestamp_us
    is_first_chunk = True
    while True:
        chunk_size = backend.initial_chunk_frames if is_first_chunk else backend.chunk_frames
        trajectory = sample_chunk_trajectory(
            start_state=state,
            start_timestamp_us=next_timestamp_us,
            command=command_source(),
            chunk_size=chunk_size,
            chunk_config=chunk_config,
            vehicle_config=vehicle_config,
            ground_snapper=ground_snapper,
        )
        yield (
            backend.render_first_chunk(trajectory)
            if is_first_chunk
            else backend.render_next_chunk(trajectory)
        )
        state = trajectory.boundary_state_after_chunk
        next_timestamp_us = int(trajectory.timestamps_us[-1] + chunk_config.frame_interval_us)
        is_first_chunk = False
