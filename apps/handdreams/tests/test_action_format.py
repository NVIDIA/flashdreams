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

"""Application wrist-rotation contracts on CPU."""

import numpy as np
import pytest
from handdreams import rotations

pytestmark = pytest.mark.ci_cpu


def test_rotation_contract_is_column_concatenated_and_right_handed() -> None:
    rotation = rotations.quat_xyzw_to_matrix([0, 0, 1, 1])
    np.testing.assert_allclose(rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(
        rotations.matrix_to_rot6d(rotation), [0, 1, 0, -1, 0, 0], atol=1e-6
    )
    np.testing.assert_allclose(rotations.validate_rotation(rotation, "wrist"), rotation)
    with pytest.raises(ValueError, match="right-handed"):
        rotations.validate_rotation(np.diag([1, 1, -1]), "wrist")
    with pytest.raises(ValueError, match="zero norm"):
        rotations.quat_xyzw_to_matrix([0, 0, 0, 0])
