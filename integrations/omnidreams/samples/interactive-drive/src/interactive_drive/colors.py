# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
    "OTHER": {
        "color": (181.0 / 255.0, 164.0 / 255.0, 71.0 / 255.0, 1.0),
        "pattern": "solid",
        "width_scale": 1.0,
    },
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

BBOX_V3_COLORS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "Car": (
        (0.0 / 255.0, 46.0 / 255.0, 136.0 / 255.0),
        (126.0 / 255.0, 206.0 / 255.0, 255.0 / 255.0),
    ),
    "Truck": (
        (204.0 / 255.0, 55.0 / 255.0, 0.0 / 255.0),
        (255.0 / 255.0, 192.0 / 255.0, 64.0 / 255.0),
    ),
    "Pedestrian": (
        (148.0 / 255.0, 0.0 / 255.0, 62.0 / 255.0),
        (255.0 / 255.0, 124.0 / 255.0, 171.0 / 255.0),
    ),
    "Cyclist": (
        (0.0 / 255.0, 80.0 / 255.0, 66.0 / 255.0),
        (102.0 / 255.0, 208.0 / 255.0, 198.0 / 255.0),
    ),
    "Others": (
        (53.0 / 255.0, 26.0 / 255.0, 20.0 / 255.0),
        (166.0 / 255.0, 136.0 / 255.0, 125.0 / 255.0),
    ),
}
