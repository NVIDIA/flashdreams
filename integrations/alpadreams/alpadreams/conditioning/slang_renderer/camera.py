# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""F-theta camera model adapter for the SlangPy rasterizer.

Adapts a flashdreams :class:`FThetaCamera` (which already carries forward and
backward polynomials in :class:`numpy.polynomial.Polynomial` form) into the
LUT + linear-matrix representation that the Slang shader consumes.

The adapter is conceptually equivalent to ``roaddreams.camera.FThetaCameraModel``
but builds its LUTs from the flashdreams polynomial directly so we don't have
to re-fit an inverse on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera


@dataclass
class FThetaCameraModel:
    """Per-camera projection state needed by the Slang rasterizer.

    Constructed from a flashdreams :class:`FThetaCamera`. ``output_width`` /
    ``output_height`` are the framebuffer dimensions (post-resize); the
    polynomial coefficients are interpreted in the camera's native pixel
    coordinates and an explicit ``uv_scale`` accounts for any downsample.
    """

    source: FThetaCamera
    output_width: int
    output_height: int
    cx: float = field(init=False)
    cy: float = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    linear_cde: npt.NDArray[np.float32] = field(init=False)
    fw_poly_coef: npt.NDArray[np.float32] = field(init=False)
    bw_poly_coef: npt.NDArray[np.float32] = field(init=False)
    radius_lut: npt.NDArray[np.float32] = field(init=False)
    theta_lut: npt.NDArray[np.float32] = field(init=False)
    max_angle_rad: float = field(init=False)
    max_radius_px: float = field(init=False)
    angle_to_radius_tail_slope: float = field(init=False)
    radius_to_angle_tail_slope: float = field(init=False)
    linear_matrix: npt.NDArray[np.float32] = field(init=False)
    uv_scale: npt.NDArray[np.float32] = field(init=False)

    def __post_init__(self) -> None:
        center = np.asarray(self.source.center, dtype=np.float32)
        self.cx = float(center[0])
        self.cy = float(center[1])
        self.width = int(self.source.width)
        self.height = int(self.source.height)
        self.linear_cde = np.asarray(self.source.linear_cde, dtype=np.float32)

        self.fw_poly_coef = np.asarray(self.source._fw_poly.coef, dtype=np.float32)
        self.bw_poly_coef = np.asarray(self.source._bw_poly.coef, dtype=np.float32)
        is_backward = self.source.reference_poly == "bw"

        max_radius = float(
            np.hypot(
                max(self.cx, self.width - self.cx),
                max(self.cy, self.height - self.cy),
            )
        )
        sample_count = 4096
        if is_backward:
            # ``pixeldistance-to-angle``: sample radii uniformly and evaluate
            # the polynomial directly to get angles.
            self.radius_lut = np.linspace(
                0.0, max_radius * 1.10, sample_count, dtype=np.float32
            )
            self.theta_lut = _eval_poly(self.bw_poly_coef, self.radius_lut)
        else:
            # ``angle-to-pixeldistance``: sample angles uniformly and evaluate
            # the *forward* polynomial to get pixel distances. We must NOT
            # re-evaluate the inverted bw_poly here because for synthetic /
            # low-degree forward polynomials the inverse fit can be unstable
            # (~1e18 outputs) and break ``angle_to_radius`` downstream.
            # See https://gitlab-master.nvidia.com/sil/omni-dreams/-/commit/92091220
            # for the original report on clipgt-065dcac9-...
            max_angle_guess = 1.6  # ~92deg half-FOV is plenty for any FTheta cam
            angles = np.linspace(0.0, max_angle_guess, sample_count, dtype=np.float32)
            radii = _eval_poly(self.fw_poly_coef, angles)
            # Monotonise before clipping so the inverse interpolation
            # (radius -> angle) stays well defined.
            radii_monotonic = np.maximum.accumulate(radii)
            within_range = radii_monotonic <= max_radius * 1.10
            if within_range.all():
                cutoff = sample_count
            else:
                cutoff = int(np.argmin(within_range))
                cutoff = max(cutoff, 2)
            self.theta_lut = angles[:cutoff].astype(np.float32)
            self.radius_lut = radii_monotonic[:cutoff].astype(np.float32)

        self.theta_lut = np.maximum.accumulate(self.theta_lut).astype(np.float32)
        self.max_angle_rad = float(self.theta_lut[-1])
        self.max_radius_px = float(self.radius_lut[-1])
        theta_step = float(self.theta_lut[-1] - self.theta_lut[-2])
        radius_step = float(self.radius_lut[-1] - self.radius_lut[-2])
        self.angle_to_radius_tail_slope = radius_step / max(theta_step, 1e-6)
        self.radius_to_angle_tail_slope = theta_step / max(radius_step, 1e-6)

        c, d, e = self.linear_cde.tolist()
        self.linear_matrix = np.array([[c, d], [e, 1.0]], dtype=np.float32)
        target_width = float(self.output_width)
        target_height = float(self.output_height)
        self.uv_scale = np.array(
            [target_width / float(self.width), target_height / float(self.height)],
            dtype=np.float32,
        )

    def angle_to_radius(self, angle_rad: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert ray angle (rad) to pixel radius via the camera polynomial."""
        angles = np.asarray(angle_rad, dtype=np.float32)
        flat_angles = angles.reshape(-1)
        clipped = np.clip(flat_angles, self.theta_lut[0], self.theta_lut[-1])
        radii = np.interp(clipped, self.theta_lut, self.radius_lut).astype(np.float32)
        high_mask = flat_angles > self.max_angle_rad
        if np.any(high_mask):
            radii[high_mask] = (
                self.max_radius_px
                + (flat_angles[high_mask] - self.max_angle_rad) * self.angle_to_radius_tail_slope
            )
        return radii.reshape(angles.shape).astype(np.float32)

    def build_angle_to_radius_lut(self, sample_count: int = 4096) -> tuple[npt.NDArray[np.float32], int, float, float, float]:
        """Build the angle→radius LUT in the framebuffer's pixel coordinates."""
        max_angle = float(self.max_angle_rad)
        angles = np.linspace(0.0, max_angle, sample_count, dtype=np.float32)
        radii_native = self.angle_to_radius(angles).astype(np.float32)
        scale = 0.5 * (float(self.uv_scale[0]) + float(self.uv_scale[1]))
        radii = (radii_native * scale).astype(np.float32)
        max_radius = float(radii[-1])
        if sample_count >= 2:
            tail_slope = float(radii[-1] - radii[-2]) / max(float(max_angle / (sample_count - 1)), 1e-6)
        else:
            tail_slope = float(self.angle_to_radius_tail_slope) * scale
        return radii, sample_count, max_angle, max_radius, tail_slope

    def build_radius_to_angle_lut(self) -> tuple[npt.NDArray[np.float32], int, float, float]:
        """Build the radius→angle LUT in the framebuffer's pixel coordinates."""
        scale = 0.5 * (float(self.uv_scale[0]) + float(self.uv_scale[1]))
        radii_scaled = (np.asarray(self.radius_lut, dtype=np.float32) * scale).astype(np.float32)
        angles = np.asarray(self.theta_lut, dtype=np.float32)
        max_radius = float(radii_scaled[-1])
        if len(radii_scaled) >= 2:
            tail_slope = float(angles[-1] - angles[-2]) / max(float(radii_scaled[-1] - radii_scaled[-2]), 1e-6)
        else:
            tail_slope = float(self.radius_to_angle_tail_slope) / max(scale, 1e-6)
        return angles, int(len(angles)), max_radius, tail_slope

    def scaled_linear_rows(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return ``linear_row0/1`` and their inverse, in framebuffer pixels."""
        scale_x, scale_y = float(self.uv_scale[0]), float(self.uv_scale[1])
        row0 = np.array(
            [self.linear_matrix[0, 0] * scale_x, self.linear_matrix[0, 1] * scale_x],
            dtype=np.float32,
        )
        row1 = np.array(
            [self.linear_matrix[1, 0] * scale_y, self.linear_matrix[1, 1] * scale_y],
            dtype=np.float32,
        )
        linear = np.stack([row0, row1], axis=0).astype(np.float32)
        inverse = np.linalg.inv(linear).astype(np.float32)
        return row0, row1, inverse[0], inverse[1]

    def principal_px_scaled(self) -> npt.NDArray[np.float32]:
        return np.array(
            [self.cx * float(self.uv_scale[0]), self.cy * float(self.uv_scale[1])],
            dtype=np.float32,
        )


def _eval_poly(coeffs: npt.NDArray[np.float32], xs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Horner evaluation of a polynomial with float32 coefficients."""
    result = np.zeros_like(xs, dtype=np.float32)
    for coeff in reversed(coeffs.tolist()):
        result = result * xs + np.float32(coeff)
    return result.astype(np.float32)
