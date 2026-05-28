# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from omnidreams.interactive_drive.math3d import invert_transform, transform_points
from omnidreams.interactive_drive.types import CameraCalibration


@dataclass
class FThetaCameraModel:
    calibration: CameraCalibration
    output_width: int | None = None
    output_height: int | None = None
    radius_lut: npt.NDArray[np.float32] = field(init=False)
    theta_lut: npt.NDArray[np.float32] = field(init=False)
    max_angle_rad: float = field(init=False)
    max_radius_px: float = field(init=False)
    tail_slope_px_per_rad: float = field(init=False)
    linear_matrix: npt.NDArray[np.float32] = field(init=False)
    uv_scale: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(self) -> None:
        max_radius = float(
            np.hypot(
                max(self.calibration.cx, self.calibration.width - self.calibration.cx),
                max(self.calibration.cy, self.calibration.height - self.calibration.cy),
            )
        )
        self.radius_lut = np.linspace(0.0, max_radius * 1.10, 4096, dtype=np.float32)
        self.theta_lut = self._poly_eval(self.radius_lut)
        self.theta_lut = np.maximum.accumulate(self.theta_lut).astype(np.float32)
        self.max_angle_rad = float(self.theta_lut[-1])
        self.max_radius_px = float(self.radius_lut[-1])
        theta_step = float(self.theta_lut[-1] - self.theta_lut[-2])
        radius_step = float(self.radius_lut[-1] - self.radius_lut[-2])
        self.tail_slope_px_per_rad = radius_step / max(theta_step, 1e-6)

        c, d, e = self.calibration.linear_cde.tolist()
        self.linear_matrix = np.array([[c, d], [e, 1.0]], dtype=np.float32)
        target_width = float(self.output_width or self.calibration.width)
        target_height = float(self.output_height or self.calibration.height)
        self.uv_scale = np.array(
            [
                target_width / float(self.calibration.width),
                target_height / float(self.calibration.height),
            ],
            dtype=np.float32,
        )

    def _poly_eval(
        self, value: npt.NDArray[np.float32] | float
    ) -> npt.NDArray[np.float32]:
        result = np.zeros_like(np.asarray(value, dtype=np.float32), dtype=np.float32)
        for power, coefficient in enumerate(self.calibration.polynomial):
            result = result + np.float32(coefficient) * np.power(
                value, power, dtype=np.float32
            )
        return result.astype(np.float32)

    def angle_to_radius(
        self, angle_rad: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if not self.calibration.is_backward_polynomial:
            return self._poly_eval(angle_rad)

        angles = np.asarray(angle_rad, dtype=np.float32)
        flat_angles = angles.reshape(-1)
        clipped = np.clip(flat_angles, self.theta_lut[0], self.theta_lut[-1])
        radii = np.interp(clipped, self.theta_lut, self.radius_lut).astype(np.float32)

        high_mask = flat_angles > self.max_angle_rad
        if np.any(high_mask):
            radii[high_mask] = (
                self.max_radius_px
                + (flat_angles[high_mask] - self.max_angle_rad)
                * self.tail_slope_px_per_rad
            )

        return radii.reshape(angles.shape).astype(np.float32)

    def project_camera_rdf(
        self, points_camera_rdf: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        xy_norm = np.linalg.norm(points_camera_rdf[:, :2], axis=1).astype(np.float32)
        ray_norm = np.linalg.norm(points_camera_rdf, axis=1).astype(np.float32)
        cos_alpha = np.divide(
            points_camera_rdf[:, 2],
            np.maximum(ray_norm, 1e-6),
            out=np.zeros_like(ray_norm),
            where=ray_norm > 1e-6,
        ).astype(np.float32)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha).astype(np.float32)
        radius = self.angle_to_radius(alpha)

        scale = np.divide(
            radius,
            np.maximum(xy_norm, 1e-6),
            out=np.zeros_like(radius),
            where=xy_norm > 1e-6,
        )
        pixels_rel = points_camera_rdf[:, :2] * scale[:, None]
        pixels_rel[xy_norm <= 1e-6] = 0.0
        uv = (pixels_rel @ self.linear_matrix.T) + np.array(
            [self.calibration.cx, self.calibration.cy], dtype=np.float32
        )
        uv = uv * self.uv_scale
        depth = points_camera_rdf[:, 2].astype(np.float32)
        return uv.astype(np.float32), depth

    def project_world(
        self,
        points_world_xyz: npt.NDArray[np.float32],
        rig_to_world: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        sensor_to_world = rig_to_world @ self.calibration.sensor_to_rig_flu
        world_to_sensor = invert_transform(sensor_to_world)
        points_sensor_flu = transform_points(world_to_sensor, points_world_xyz)
        points_camera_rdf = np.stack(
            [
                -points_sensor_flu[:, 1],
                -points_sensor_flu[:, 2],
                points_sensor_flu[:, 0],
            ],
            axis=1,
        ).astype(np.float32)
        uv, depth = self.project_camera_rdf(points_camera_rdf)
        valid = depth > 0.0
        return uv, depth, valid
