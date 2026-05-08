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

"""HD-map v3 color tables and lane-line style descriptors.

Lifted from ``roaddreams.colors`` so flashdreams' Slang rasterizer paints HD
map elements with the same palette ludus-renderer used.
"""

from __future__ import annotations

LANE_LINE_STYLE_CONFIG: dict[str, dict[str, object]] = {
    "WHITE SOLID_SINGLE": {"color": (1.0, 1.0, 1.0, 1.0), "pattern": "solid", "width_scale": 1.0},
    "WHITE LONG_DASHED_SINGLE": {
        "color": (1.0, 1.0, 1.0, 1.0),
        "pattern": "long_dashed",
        "width_scale": 1.0,
    },
    "WHITE SHORT_DASHED_SINGLE": {
        "color": (1.0, 1.0, 1.0, 1.0),
        "pattern": "short_dashed",
        "width_scale": 1.0,
    },
    "WHITE DOT_DASHED_SINGLE": {
        "color": (1.0, 1.0, 1.0, 1.0),
        "pattern": "dot_dashed",
        "width_scale": 1.0,
    },
    "WHITE SOLID_GROUP": {
        "color": (1.0, 1.0, 1.0, 1.0),
        "pattern": "dual",
        "dual_pattern": ("solid", "solid"),
        "width_scale": 1.0,
    },
    "YELLOW SOLID_SINGLE": {
        "color": (1.0, 1.0, 0.0, 1.0),
        "pattern": "solid",
        "width_scale": 1.0,
    },
    "YELLOW LONG_DASHED_SINGLE": {
        "color": (1.0, 1.0, 0.0, 1.0),
        "pattern": "long_dashed",
        "width_scale": 1.0,
    },
    "YELLOW DASHED_SOLID": {
        "color": (1.0, 1.0, 0.0, 1.0),
        "pattern": "dual",
        "dual_pattern": ("solid", "long_dashed"),
        "width_scale": 1.0,
    },
    "YELLOW SOLID_DASHED": {
        "color": (1.0, 1.0, 0.0, 1.0),
        "pattern": "dual",
        "dual_pattern": ("long_dashed", "solid"),
        "width_scale": 1.0,
    },
    "YELLOW DOT_SOLID_SINGLE": {
        "color": (1.0, 1.0, 0.0, 1.0),
        "pattern": "dotted_1_9",
        "width_scale": 1.0,
    },
    "YELLOW SOLID_GROUP": {
        "color": (1.0, 1.0, 0.0, 1.0),
        "pattern": "dual",
        "dual_pattern": ("solid", "solid"),
        "width_scale": 1.0,
    },
    "OTHER": {"color": (181.0 / 255.0, 164.0 / 255.0, 71.0 / 255.0, 1.0), "pattern": "solid", "width_scale": 1.0},
}

HDMAP_V3_COLORS: dict[str, tuple[float, float, float, float]] = {
    "lanelines": (98.0 / 255.0, 183.0 / 255.0, 249.0 / 255.0, 1.0),
    "road_boundaries": (253.0 / 255.0, 1.0 / 255.0, 232.0 / 255.0, 1.0),
    "wait_lines": (108.0 / 255.0, 179.0 / 255.0, 59.0 / 255.0, 1.0),
    "crosswalks": (139.0 / 255.0, 93.0 / 255.0, 1.0, 1.0),
    "road_markings": (20.0 / 255.0, 254.0 / 255.0, 185.0 / 255.0, 1.0),
    "poles": (183.0 / 255.0, 69.0 / 255.0, 177.0 / 255.0, 1.0),
    "traffic_signs": (8.0 / 255.0, 2.0 / 255.0, 1.0, 1.0),
    "traffic_lights": (100.0 / 255.0, 100.0 / 255.0, 100.0 / 255.0, 1.0),
    "intersection_areas": (87.0 / 255.0, 110.0 / 255.0, 1.0, 0.95),
    "road_islands": (1.0, 155.0 / 255.0, 37.0 / 255.0, 0.95),
}
