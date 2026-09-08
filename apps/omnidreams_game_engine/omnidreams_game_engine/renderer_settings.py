# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""File-backed renderer and BEV settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.yaml_config import (
    StrictConfigError,
    load_yaml_mapping,
    require_bool,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_version,
)

_RASTER_FLOAT_FIELDS = {
    "near_plane_m",
    "far_plane_m",
    "fog_start_m",
    "fog_end_m",
    "fog_power",
    "triangle_raytrace_distance_m",
    "lane_segment_interval_m",
    "polyline_segment_interval_m",
    "line_width_px",
    "pole_width_px",
    "dual_line_offset_m",
}
_RASTER_INT_FIELDS = {"width", "height", "triangle_raytrace_edge_samples"}
_BEV_FLOAT_FIELDS = {"height_m", "fov_deg", "tilt_deg"}
_BEV_INT_FIELDS = {"width", "height"}


@dataclass(frozen=True)
class RendererSettings:
    """Portable visual settings loaded from a renderer YAML file."""

    raster: RasterConfig
    """Primary-camera rasterization settings."""

    bev: BevConfig
    """Top-down HUD rasterization settings."""


def load_renderer_settings(path: Path) -> RendererSettings:
    """Load a complete renderer YAML document.

    Args:
        path: Renderer YAML path.

    Returns:
        Validated visual settings.
    """
    doc = load_yaml_mapping(path)
    require_exact_keys(doc, {"schema_version", "raster", "bev"}, "renderer")
    require_version(doc, "renderer")
    raw_raster = require_mapping(doc["raster"], "renderer.raster")
    require_exact_keys(
        raw_raster, _RASTER_FLOAT_FIELDS | _RASTER_INT_FIELDS, "renderer.raster"
    )
    raster_values = {
        name: require_float(raw_raster[name], f"renderer.raster.{name}", minimum=0.0)
        for name in _RASTER_FLOAT_FIELDS
    }
    raster_values.update(
        {
            name: require_int(raw_raster[name], f"renderer.raster.{name}")
            for name in _RASTER_INT_FIELDS
        }
    )
    if raster_values["near_plane_m"] >= raster_values["far_plane_m"]:
        raise StrictConfigError(
            "renderer.raster.near_plane_m must be less than far_plane_m"
        )
    if raster_values["fog_start_m"] >= raster_values["fog_end_m"]:
        raise StrictConfigError(
            "renderer.raster.fog_start_m must be less than fog_end_m"
        )

    raw_bev = require_mapping(doc["bev"], "renderer.bev")
    require_exact_keys(
        raw_bev, {"enabled"} | _BEV_FLOAT_FIELDS | _BEV_INT_FIELDS, "renderer.bev"
    )
    bev_values = {
        name: require_float(raw_bev[name], f"renderer.bev.{name}", minimum=0.0)
        for name in _BEV_FLOAT_FIELDS
    }
    bev_values.update(
        {
            name: require_int(raw_bev[name], f"renderer.bev.{name}")
            for name in _BEV_INT_FIELDS
        }
    )
    bev_values["enabled"] = require_bool(raw_bev["enabled"], "renderer.bev.enabled")
    if not 0.0 < bev_values["fov_deg"] < 180.0:
        raise StrictConfigError("renderer.bev.fov_deg must be between 0 and 180")
    return RendererSettings(
        raster=RasterConfig(**cast(Any, raster_values)),
        bev=BevConfig(**cast(Any, bev_values)),
    )
