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

"""Unit tests for hierarchical context parallel group creation."""

import pytest
from omnidreams.transformer.impl.context_parallel import (
    create_hierarchical_cp_groups,
)

pytestmark = pytest.mark.ci_cpu


def _get_expected_groups(world_size: int, V: int, T: int) -> dict:
    """Return expected group configurations for various world_size, V, T combinations."""
    if world_size == 1:
        return {
            "HW_groups": [(0,)],
            "T_groups": [(0,)],
            "THW_groups": [(0,)],
            "V_groups": [(0,)],
            "VHW_groups": [(0,)],
        }
    elif (
        (world_size == 2 and V == 1 and T == 1)
        or (world_size == 2 and V == 1 and T == 3)
        or (world_size == 2 and V == 3 and T == 1)
        or (world_size == 2 and V == 5 and T == 5)
    ):
        return {
            "HW_groups": [(0, 1)],
            "T_groups": [(0,), (1,)],
            "THW_groups": [(0, 1)],
            "V_groups": [(0,), (1,)],
            "VHW_groups": [(0, 1)],
        }
    elif (world_size == 2 and V == 1 and T == 2) or (
        world_size == 2 and V == 1 and T == 4
    ):
        return {
            "HW_groups": [(0,), (1,)],
            "T_groups": [(0, 1)],
            "THW_groups": [(0, 1)],
            "V_groups": [(0,), (1,)],
            "VHW_groups": [(0,), (1,)],
        }
    elif world_size == 4 and V == 1 and T == 2:
        return {
            "HW_groups": [(0, 1), (2, 3)],
            "T_groups": [(0, 2), (1, 3)],
            "THW_groups": [(0, 1, 2, 3)],
            "V_groups": [(0,), (1,), (2,), (3,)],
            "VHW_groups": [(0, 1), (2, 3)],
        }
    elif world_size == 2 and V == 2:
        return {
            "HW_groups": [(0,), (1,)],
            "T_groups": [(0,), (1,)],
            "THW_groups": [(0,), (1,)],
            "V_groups": [(0, 1)],
            "VHW_groups": [(0, 1)],
        }
    elif world_size == 8 and V == 2 and T == 2:
        return {
            "HW_groups": [(0, 1), (2, 3), (4, 5), (6, 7)],
            "T_groups": [(0, 2), (1, 3), (4, 6), (5, 7)],
            "THW_groups": [(0, 1, 2, 3), (4, 5, 6, 7)],
            "V_groups": [(0, 4), (1, 5), (2, 6), (3, 7)],
            "VHW_groups": [(0, 1, 4, 5), (2, 3, 6, 7)],
        }
    elif world_size == 4 and V == 2 and T == 2:
        return {
            "HW_groups": [(0,), (1,), (2,), (3,)],
            "T_groups": [(0, 1), (2, 3)],
            "THW_groups": [(0, 1), (2, 3)],
            "V_groups": [(0, 2), (1, 3)],
            "VHW_groups": [(0, 2), (1, 3)],
        }
    elif world_size == 8 and V == 1 and T == 4:
        return {
            "HW_groups": [(0, 1), (2, 3), (4, 5), (6, 7)],
            "T_groups": [(0, 2, 4, 6), (1, 3, 5, 7)],
            "THW_groups": [(0, 1, 2, 3, 4, 5, 6, 7)],
            "V_groups": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
            "VHW_groups": [(0, 1), (2, 3), (4, 5), (6, 7)],
        }
    else:
        raise ValueError(f"Unsupported world_size: {world_size}, V: {V}, T: {T}")


def _verify_hierarchical_cp_groups(world_size: int, V: int, T: int) -> None:
    """Verify that hierarchical CP groups are created correctly for given parameters."""
    expected = _get_expected_groups(world_size, V, T)

    results = {
        rank: create_hierarchical_cp_groups(world_size=world_size, rank=rank, V=V, T=T)
        for rank in range(world_size)
    }

    HW_groups = []
    T_groups = []
    THW_groups = []
    V_groups = []
    VHW_groups = []

    for rank, result in results.items():
        assert rank in result.HW_ranks, "Rank should be in its own HW group"
        assert rank in result.T_ranks, "Rank should be in its own T group"
        assert rank in result.THW_ranks, "Rank should be in its own THW group"
        assert rank in result.V_ranks, "Rank should be in its own V group"
        assert rank in result.VHW_ranks, "Rank should be in its own VHW group"
        HW_groups.append(result.HW_ranks)
        T_groups.append(result.T_ranks)
        THW_groups.append(result.THW_ranks)
        V_groups.append(result.V_ranks)
        VHW_groups.append(result.VHW_ranks)

    assert set(HW_groups) == set(expected["HW_groups"]), (
        f"HW_groups mismatch: {HW_groups} != {expected['HW_groups']}"
    )
    assert set(T_groups) == set(expected["T_groups"]), (
        f"T_groups mismatch: {T_groups} != {expected['T_groups']}"
    )
    assert set(THW_groups) == set(expected["THW_groups"]), (
        f"THW_groups mismatch: {THW_groups} != {expected['THW_groups']}"
    )
    assert set(V_groups) == set(expected["V_groups"]), (
        f"V_groups mismatch: {V_groups} != {expected['V_groups']}"
    )
    assert set(VHW_groups) == set(expected["VHW_groups"]), (
        f"VHW_groups mismatch: {VHW_groups} != {expected['VHW_groups']}"
    )


@pytest.mark.parametrize(
    "world_size,V,T",
    [
        # Single GPU - no CP needed
        (1, 1, 1),
        (1, 1, 3),
        (1, 4, 3),
        # Cannot split V or T, so split HW
        (2, 1, 1),
        (2, 1, 3),
        (2, 3, 1),
        (2, 5, 5),
        # Cannot split V but can split T
        (2, 1, 2),
        (2, 1, 4),
        # Cannot split V but can split T, also split HW
        (4, 1, 2),
        # Can split V
        (2, 2, 1),
        (2, 2, 3),
        # Can split V and T, also split HW
        (8, 2, 2),
        # Can split V and T
        (4, 2, 2),
        # Cannot split V but can split T, also split HW
        (8, 1, 4),
    ],
)
def test_hierarchical_cp_groups(world_size: int, V: int, T: int) -> None:
    """Test hierarchical context parallel group creation for various configurations."""
    _verify_hierarchical_cp_groups(world_size, V, T)
