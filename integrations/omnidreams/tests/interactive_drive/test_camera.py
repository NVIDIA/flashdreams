# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import numpy as np
from omnidreams.interactive_drive.camera import FThetaCameraModel
from omnidreams.interactive_drive.types import CameraCalibration


def test_backward_ftheta_angle_to_radius_extrapolates_past_lut_domain() -> None:
    calibration = CameraCalibration(
        clipgt_name="camera:test",
        logical_name="camera_test",
        width=100,
        height=80,
        cx=50.0,
        cy=40.0,
        polynomial=np.array([0.0, 0.01], dtype=np.float32),
        is_backward_polynomial=True,
        linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=np.eye(4, dtype=np.float32),
    )
    model = FThetaCameraModel(calibration)

    sample_angles = np.array(
        [model.max_angle_rad - 0.01, model.max_angle_rad + 0.10], dtype=np.float32
    )
    radii = model.angle_to_radius(sample_angles)

    expected_extrapolated = model.max_radius_px + 0.10 * model.tail_slope_px_per_rad
    assert radii[0] < model.max_radius_px
    assert np.isclose(radii[1], expected_extrapolated, atol=1e-4)
    assert radii[1] > model.max_radius_px
