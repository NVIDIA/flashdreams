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

"""Canonical front-camera calibration for compiled semantic maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from omnidreams_game_engine.math3d import (
    euler_xyz_degrees_to_matrix,
    transform_from_rt,
)

if TYPE_CHECKING:
    from omnidreams_game_engine.types import CameraCalibration

DEFAULT_FRONT_CAMERA_CLIPGT_NAME = "camera:front:wide:120fov"
"""ClipGT sensor name embedded in compiled semantic-map archives."""

DEFAULT_FRONT_CAMERA_LOGICAL_NAME = "camera_front_wide_120fov"
"""Filesystem-safe name for the canonical front camera."""

DEFAULT_FIRST_FRAME_RESOLUTION_WH = (1280, 704)
"""Pixel resolution used by generated and authored first frames."""

_NATIVE_RESOLUTION_WH = (3848, 2168)
_PRINCIPAL_POINT_XY = (1921.318705874846, 1076.978854184438)
_POLYNOMIAL = (
    0.0,
    0.0005385247479413695,
    -1.598462177407655e-09,
    6.250864794463573e-12,
    -2.194585699335322e-15,
    4.525222700710391e-19,
)
_POLYNOMIAL_TEXT = (
    "0 0.0005385247479413695 -1.598462177407655e-09 "
    "6.250864794463573e-12 -2.194585699335322e-15 "
    "4.525222700710391e-19"
)
_NOMINAL_RPY_DEG = (
    0.292217969894409,
    0.464194804430008,
    -0.191304489970207,
)
_CORRECTION_RPY_DEG = (
    -0.1592078059911728,
    0.11539523303508759,
    0.5026581287384033,
)
_NOMINAL_TRANSLATION_M = (
    1.69035196304321,
    0.00553808081895113,
    1.45306670665741,
)
_CORRECTION_TRANSLATION_M = (
    -0.057110343128442764,
    -0.0032010308932513,
    0.008508340455591679,
)


def default_front_camera_calibration() -> CameraCalibration:
    """Build the canonical compiled-map front-camera calibration."""
    from omnidreams_game_engine.types import CameraCalibration

    nominal_rotation = euler_xyz_degrees_to_matrix(_NOMINAL_RPY_DEG)
    correction_rotation = euler_xyz_degrees_to_matrix(_CORRECTION_RPY_DEG)
    rotation = (nominal_rotation @ correction_rotation).astype(np.float32)
    translation = np.asarray(_NOMINAL_TRANSLATION_M, dtype=np.float32) + np.asarray(
        _CORRECTION_TRANSLATION_M, dtype=np.float32
    )
    return CameraCalibration(
        clipgt_name=DEFAULT_FRONT_CAMERA_CLIPGT_NAME,
        logical_name=DEFAULT_FRONT_CAMERA_LOGICAL_NAME,
        width=_NATIVE_RESOLUTION_WH[0],
        height=_NATIVE_RESOLUTION_WH[1],
        cx=_PRINCIPAL_POINT_XY[0],
        cy=_PRINCIPAL_POINT_XY[1],
        polynomial=np.asarray(_POLYNOMIAL, dtype=np.float32),
        is_backward_polynomial=True,
        linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sensor_to_rig_flu=transform_from_rt(rotation, translation.tolist()),
    )


def default_front_camera_rig() -> dict[str, object]:
    """Build the ClipGT rig record for the canonical front camera."""
    return {
        "rig": {
            "properties": {},
            "vehicle": {},
            "vehicleio": {},
            "sensors": [
                {
                    "name": DEFAULT_FRONT_CAMERA_CLIPGT_NAME,
                    "protocol": "camera.virtual",
                    "parameter": ("video=synthetic/camera_front_wide_120fov.mp4"),
                    "nominalSensor2Rig_FLU": {
                        "roll-pitch-yaw": list(_NOMINAL_RPY_DEG),
                        "t": list(_NOMINAL_TRANSLATION_M),
                    },
                    "correction_sensor_R_FLU": {
                        "roll-pitch-yaw": list(_CORRECTION_RPY_DEG),
                    },
                    "correction_rig_T": list(_CORRECTION_TRANSLATION_M),
                    "properties": {
                        "width": str(_NATIVE_RESOLUTION_WH[0]),
                        "height": str(_NATIVE_RESOLUTION_WH[1]),
                        "cx": str(_PRINCIPAL_POINT_XY[0]),
                        "cy": str(_PRINCIPAL_POINT_XY[1]),
                        "Model": "ftheta",
                        "polynomial-type": "pixeldistance-to-angle",
                        "polynomial": _POLYNOMIAL_TEXT,
                        "linear-c": "1.000000",
                        "linear-d": "0.000000",
                        "linear-e": "0.000000",
                    },
                }
            ],
        }
    }
