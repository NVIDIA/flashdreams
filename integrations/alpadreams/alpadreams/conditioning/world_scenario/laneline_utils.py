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

from alpadreams.conditioning.world_scenario.data_types import (
    LaneLineColor,
    LaneLineStyle,
    LaneLineType,
)


def build_lane_line_type(
    color: LaneLineColor | None = None,
    style: LaneLineStyle | None = None,
    lane_type_hint: str | None = None,
) -> LaneLineType:
    """
    Build a LaneLineType instance.

    Args:
        color: Optional color enum value
        style: Optional style enum value
        lane_type_hint: Optional hint string from data (e.g., "YELLOW SOLID_SINGLE") - only used if color/style not provided

    Returns:
        LaneLineType instance
    """
    # If we have both color and style, just use them directly
    if color and style:
        return LaneLineType(color=color, style=style)

    # If we have a lane_type_hint but missing color/style, try to parse it
    if lane_type_hint and (not color or not style):
        # Normalize the hint (replace underscores with spaces)
        normalized_hint = lane_type_hint.replace("_", " ").upper()
        parts = normalized_hint.split(" ", 1)
        if len(parts) == 2:
            color_str, style_str = parts
            try:
                parsed_color = LaneLineColor[color_str] if not color else color
                parsed_style = LaneLineStyle[style_str] if not style else style
                return LaneLineType(color=parsed_color, style=parsed_style)
            except (KeyError, ValueError):
                pass

    # Fallback to UNKNOWN if missing
    if not color:
        color = LaneLineColor.UNKNOWN
    if not style:
        style = LaneLineStyle.UNKNOWN

    return LaneLineType(color=color, style=style)
