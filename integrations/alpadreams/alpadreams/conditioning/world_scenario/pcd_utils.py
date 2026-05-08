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

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
File modified from https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
"""

import numpy as np


def interpolate_polyline_to_points(
    polyline: np.ndarray, segment_interval: float = 0.025
) -> np.ndarray:
    """
    polyline:
        numpy.ndarray, shape (N, 3) or list of points

    Returns:
        points: numpy array, shape (interpolate_num*N, 3)
    """

    def interpolate_points(
        previous_vertex: np.ndarray, vertex: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            previous_vertex: (x, y, z)
            vertex: (x, y, z)

        Returns:
            points: numpy array, shape (interpolate_num, 3)
        """
        interpolate_num = int(
            np.linalg.norm(np.array(vertex) - np.array(previous_vertex))
            / segment_interval
        )
        interpolate_num = max(interpolate_num, 2)

        # interpolate between previous_vertex and vertex
        x = np.linspace(previous_vertex[0], vertex[0], num=interpolate_num)
        y = np.linspace(previous_vertex[1], vertex[1], num=interpolate_num)
        z = np.linspace(previous_vertex[2], vertex[2], num=interpolate_num)

        # remove the last point, we will include it in the next interpolation
        return np.stack([x, y, z], axis=1)[:-1]

    points = []
    previous_vertex = polyline[0]
    for vertex in polyline[1:]:
        points.extend(interpolate_points(previous_vertex, vertex))
        previous_vertex = vertex

    # add the last point
    points.append(polyline[-1])

    return np.array(points)
