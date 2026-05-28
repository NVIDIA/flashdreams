# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test-only fake backends and minimal-scene factories.

Lives in the installed package (rather than under ``tests/``) so root
pytest's ``--import-mode=importlib`` can resolve the import
unambiguously -- a bare ``from pipeline_fakes import ...`` from inside
a test module would only work under the legacy ``prepend`` import mode
that puts each test directory on ``sys.path``. The leading underscore
signals this module isn't part of the public ``interactive_drive``
runtime API.
"""

from pathlib import Path

import numpy as np
from omnidreams.interactive_drive.types import (
    CameraCalibration,
    FrameChunk,
    PresentedFrame,
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)


def minimal_scene() -> SceneBundle:
    camera = CameraCalibration(
        clipgt_name="cam",
        logical_name="cam",
        width=4,
        height=4,
        cx=2.0,
        cy=2.0,
        polynomial=np.zeros(4, dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.zeros(3, dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    return SceneBundle(
        scene_path=Path("/dev/null"),
        scene_id="test-scene",
        metadata={},
        selected_camera=camera,
        initial_rig_to_world=np.eye(4, dtype=np.float32),
        initial_timestamp_us=0,
        initial_yaw_rad=0.0,
        initial_speed_mps=0.0,
        initial_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        prompt="",
        line_layers=(),
        triangle_layers=(),
    )


def make_trajectory(chunk_size: int) -> TrajectoryChunk:
    return TrajectoryChunk(
        timestamps_us=np.arange(chunk_size, dtype=np.int64),
        rig_poses_world=np.repeat(
            np.eye(4, dtype=np.float32)[None], chunk_size, axis=0
        ),
        boundary_state_after_chunk=VehicleState(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
            speed_mps=0.0,
            steer_rad=0.0,
        ),
    )


class FakeVideoModelBackend:
    """Deterministic backend stub used by pipeline and loop tests."""

    def __init__(self, frames_per_render: int, rgb_value: int = 0) -> None:
        self._frames_per_render = frames_per_render
        self._rgb_value = rgb_value
        self.warmup_calls = 0
        self.reset_calls = 0

    def warmup(self, scene: SceneBundle) -> None:
        del scene
        self.warmup_calls += 1

    def reset(self) -> None:
        self.reset_calls += 1

    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        frames = tuple(
            PresentedFrame(
                timestamp_us=int(trajectory.timestamps_us[idx]),
                rgb_host_uint8=np.full((4, 4, 3), self._rgb_value, dtype=np.uint8),
                depth_host_f32=None,
            )
            for idx in range(self._frames_per_render)
        )
        return FrameChunk(
            frames=frames,
            boundary_state_after_chunk=trajectory.boundary_state_after_chunk,
            source_name="fake",
        )
