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

"""PIL-rendered floating turn arrows for Crazy Robotaxi presenters."""

from __future__ import annotations

from omnidreams.interactive_drive.crazy_robotaxi.navigation import TurnManeuver
from PIL import ImageDraw

_SIGN_GREEN = (118, 185, 0)


def draw_floating_turn_sign(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    maneuver: TurnManeuver,
    *,
    size_px: int = 60,
    alpha: int | None = None,
) -> None:
    """Draw one constant-size outlined turn arrow.

    Args:
        draw: PIL drawing context for the destination image.
        center: Sign center in destination pixels.
        maneuver: Direction represented by the arrow glyph.
        size_px: Diameter of the circular sign backing.
        alpha: Optional alpha applied to colors on an RGBA destination.
    """
    cx, cy = center
    radius = max(12, int(size_px) // 2)
    green = _with_alpha(_SIGN_GREEN, alpha)
    black = _with_alpha((0, 0, 0), alpha)

    if maneuver == "straight":
        _draw_straight_arrow(draw, center, radius, green, black)
    elif maneuver == "left":
        _draw_corner_arrow(draw, center, radius, green, black, direction=-1)
    elif maneuver == "right":
        _draw_corner_arrow(draw, center, radius, green, black, direction=1)
    else:
        _draw_u_turn_arrow(draw, center, radius, green, black)


def _draw_straight_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    green: tuple[int, ...],
    black: tuple[int, ...],
) -> None:
    cx, cy = center
    shaft_bottom = cy + int(radius * 0.55)
    shaft_top = cy - int(radius * 0.08)
    line_width = max(5, radius // 4)
    draw.line((cx, shaft_bottom, cx, shaft_top), fill=black, width=line_width + 5)
    draw.line((cx, shaft_bottom, cx, shaft_top), fill=green, width=line_width)
    tip = (cx, cy - int(radius * 0.68))
    base_y = cy - int(radius * 0.02)
    half_width = int(radius * 0.42)
    draw.polygon(
        [tip, (cx - half_width, base_y), (cx + half_width, base_y)],
        fill=green,
        outline=black,
        width=max(2, radius // 12),
    )


def _draw_corner_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    green: tuple[int, ...],
    black: tuple[int, ...],
    *,
    direction: int,
) -> None:
    cx, cy = center
    corner_y = cy - int(radius * 0.16)
    end_x = cx + direction * int(radius * 0.38)
    path = [(cx, cy + int(radius * 0.56)), (cx, corner_y), (end_x, corner_y)]
    line_width = max(5, radius // 4)
    draw.line(path, fill=black, width=line_width + 5, joint="curve")
    draw.line(path, fill=green, width=line_width, joint="curve")
    tip_x = cx + direction * int(radius * 0.68)
    base_x = cx + direction * int(radius * 0.1)
    half_height = int(radius * 0.4)
    draw.polygon(
        [
            (tip_x, corner_y),
            (base_x, corner_y - half_height),
            (base_x, corner_y + half_height),
        ],
        fill=green,
        outline=black,
        width=max(2, radius // 12),
    )


def _draw_u_turn_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    green: tuple[int, ...],
    black: tuple[int, ...],
) -> None:
    cx, cy = center
    path = [
        (cx + int(radius * 0.3), cy + int(radius * 0.52)),
        (cx + int(radius * 0.3), cy - int(radius * 0.08)),
        (cx + int(radius * 0.22), cy - int(radius * 0.32)),
        (cx, cy - int(radius * 0.47)),
        (cx - int(radius * 0.22), cy - int(radius * 0.32)),
        (cx - int(radius * 0.3), cy - int(radius * 0.08)),
        (cx - int(radius * 0.3), cy + int(radius * 0.22)),
    ]
    line_width = max(5, radius // 4)
    draw.line(path, fill=black, width=line_width + 5, joint="curve")
    draw.line(path, fill=green, width=line_width, joint="curve")
    tip_y = cy + int(radius * 0.65)
    base_y = cy + int(radius * 0.1)
    half_width = int(radius * 0.4)
    head_x = cx - int(radius * 0.3)
    draw.polygon(
        [
            (head_x, tip_y),
            (head_x - half_width, base_y),
            (head_x + half_width, base_y),
        ],
        fill=green,
        outline=black,
        width=max(2, radius // 12),
    )


def _with_alpha(color: tuple[int, int, int], alpha: int | None) -> tuple[int, ...]:
    return color if alpha is None else (*color, alpha)
