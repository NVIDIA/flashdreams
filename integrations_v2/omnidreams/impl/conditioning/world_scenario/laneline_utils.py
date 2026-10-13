# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lane line utilities."""

from omnidreams.impl.conditioning.world_scenario.data_types import (
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
