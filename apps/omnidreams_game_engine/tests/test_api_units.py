# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Unreal-to-FLU conversions.

The expected values follow Unreal's ``FRotationMatrix`` convention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnidreams_game_engine.api.v0 import units

pytestmark = pytest.mark.ci_cpu


def test_position_cm_to_flu_metres() -> None:
    np.testing.assert_allclose(
        units.ue_pos_to_flu([12345.6, -789.1, 23.4]),
        [123.456, 7.891, 0.23399999999999999],
        rtol=0,
        atol=1e-12,
    )
    batch = units.ue_pos_to_flu([[100.0, 200.0, 300.0], [0.0, -50.0, 0.0]])
    np.testing.assert_allclose(batch, [[1.0, -2.0, 3.0], [0.0, 0.5, 0.0]])


def test_rotator_matrix_matches_unreal() -> None:
    np.testing.assert_allclose(
        units.ue_rotator_matrix(3.5, -47.25, 1.75),
        [
            [0.6775346453111358, 0.7352455243895376, -0.018995348595570036],
            [-0.7329528499322274, 0.677115125963464, 0.06553797347879951],
            [0.06104853953485687, -0.03048155272678986, 0.9976692592056873],
        ],
        rtol=0,
        atol=1e-15,
    )


def test_rotation_to_flu_matches_unreal() -> None:
    np.testing.assert_allclose(
        units.ue_rot_to_flu([3.5, -47.25, 1.75]),
        [
            [0.6775346453111358, -0.7352455243895376, -0.018995348595570036],
            [0.7329528499322274, 0.677115125963464, -0.06553797347879951],
            [0.06104853953485687, 0.03048155272678986, 0.9976692592056873],
        ],
        rtol=0,
        atol=1e-15,
    )
    # Unreal yaw +90 turns right; in FLU the body then faces -y.
    np.testing.assert_allclose(
        units.ue_rot_to_flu([0.0, 90.0, 0.0]),
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-15,
    )


def test_transform_matches_unreal() -> None:
    np.testing.assert_allclose(
        units.ue_transform_to_flu([100.0, 200.0, 50.0], [-2.0, 135.0, 0.5]),
        [
            [-0.7066760308408344, 0.706864506156952, -0.030847323555414693, 1.0],
            [-0.7066760308408345, -0.7072952072970806, -0.018506138701084023, -2.0],
            [-0.03489949670250097, 0.008721219528731424, 0.9993527732787075, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        rtol=0,
        atol=1e-15,
    )
    assert units.ue_yaw_to_flu_rad(90.0) == pytest.approx(-math.pi / 2)


@pytest.mark.parametrize(
    "rotator",
    [
        (0.0, 0.0, 0.0),
        (3.5, -47.25, 1.75),
        (0.0, 180.0, 0.0),
        (180.0, 0.0, 0.0),
        (80.0, 10.0, -170.0),
    ],
)
def test_quaternion_reproduces_the_rotation(
    rotator: tuple[float, float, float],
) -> None:
    rotation = units.ue_rot_to_flu(rotator)
    x, y, z, w = units.rotation_to_quat_xyzw(rotation)
    assert w >= 0.0
    rebuilt = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(rebuilt, rotation, atol=1e-12)
