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

"""Lane line utilities."""

from omnidreams.conditioning.world_scenario.data_types import (
    LaneLineColor,
    LaneLineStyle,
    LaneLineType,
)


def build_lane_line_type(
    color: LaneLineColor | None = None,
    style: LaneLineStyle | None = None,
) -> LaneLineType:
    """Build a LaneLineType from color and style.

    Args:
        color: Lane line color, or ``None`` to default to UNKNOWN.
        style: Lane line style, or ``None`` to default to UNKNOWN.

    Returns:
        A LaneLineType; missing color or style defaults to UNKNOWN.
    """
    if color and style:
        return LaneLineType(color=color, style=style)

    if not color:
        color = LaneLineColor.UNKNOWN
    if not style:
        style = LaneLineStyle.UNKNOWN

    return LaneLineType(color=color, style=style)
