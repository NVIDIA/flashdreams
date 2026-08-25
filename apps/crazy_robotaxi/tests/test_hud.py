# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for frame-aligned V2 HUD channels."""

import numpy as np
import pytest
import torch
from crazy_robotaxi.hud import render_hud
from crazy_robotaxi.rules import TaxiGameSnapshot
from omnidreams_game_engine.types import CameraCalibration

pytestmark = pytest.mark.ci_cpu


def test_hud_returns_one_rgba_layer_per_generated_frame() -> None:
    calibration = CameraCalibration(
        clipgt_name="front",
        logical_name="front",
        width=160,
        height=96,
        cx=80.0,
        cy=48.0,
        polynomial=np.asarray([0.0, 100.0, 0.0, 0.0], dtype=np.float32),
        is_backward_polynomial=False,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    snapshot = TaxiGameSnapshot(
        phase="seeking_pickup",
        target_xyz_m=(25.0, 0.0, 0.0),
        distance_m=25.0,
        relative_bearing_rad=0.0,
        target_radius_m=5.0,
        remaining_time_s=None,
        score=0,
        global_remaining_time_s=60.0,
    )

    result = render_hud(
        (snapshot, snapshot),
        rig_poses_world=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        calibration=calibration,
        bev_tchw=None,
        width=160,
        height=96,
        device="cpu",
        dtype=torch.float32,
    )

    assert result.shape == (2, 4, 96, 160)
    assert torch.all((0.0 <= result[:, 3]) & (result[:, 3] <= 1.0))
    assert torch.any(result[:, 3] > 0.0)
