# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from omnidreams.interactive_drive._sample_assets import SAMPLE_SCENE
from omnidreams.interactive_drive.colors import BBOX_V3_COLORS
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.scene_loader import (
    _build_lane_centerlines,
    _discover_prompts,
    load_scene_bundle,
)


def _point(x_m: float, y_m: float, z_m: float = 0.0) -> dict[str, float]:
    return {"x": x_m, "y": y_m, "z": z_m}


def test_lane_centerlines_use_car_lane_rail_midpoints() -> None:
    rows = [
        {
            "lane": {
                "left_rail": [_point(0.0, 2.0), _point(10.0, 2.0)],
                "right_rail": [_point(10.0, -2.0), _point(0.0, -2.0)],
                "vehicle_types": ["CAR"],
            }
        },
        {
            "lane": {
                "left_rail": [_point(0.0, 12.0), _point(10.0, 12.0)],
                "right_rail": [_point(0.0, 8.0), _point(10.0, 8.0)],
                "vehicle_types": ["BICYCLE"],
            }
        },
    ]

    centerlines = _build_lane_centerlines(rows)

    assert len(centerlines) == 1
    np.testing.assert_allclose(
        centerlines[0],
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_usdz_prompt_discovery_accepts_legacy_numeric_suffix() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("prompt1.txt", "legacy one")
        zf.writestr("prompt_2.txt", "canonical two")
        zf.writestr("promptnight.txt", "ignored")

    archive.seek(0)
    with zipfile.ZipFile(archive, "r") as zf:
        prompts = _discover_prompts(zf)

    assert prompts["default"] == "legacy one"
    assert prompts["1"] == "legacy one"
    assert prompts["2"] == "canonical two"
    assert "night" not in prompts


# Opportunistic: exercises the real USDZ loader, so this test is silently
# skipped on machines where ``prepare.py`` hasn't fetched the production asset.
@pytest.mark.skipif(
    not SAMPLE_SCENE.exists(),
    reason="sample scene is not available on this workstation",
)
def test_load_scene_bundle_from_real_usdz() -> None:
    bundle = load_scene_bundle(
        scene_path=SAMPLE_SCENE,
        camera_name="camera_front_wide_120fov",
        variant="1",
        prompt_override=None,
        raster=RasterConfig(width=640, height=352),
    )

    assert bundle.scene_id.startswith("clipgt-")
    assert bundle.selected_camera.logical_name == "camera_front_wide_120fov"
    assert bundle.initial_rgb.shape == (352, 640, 3)
    assert bundle.initial_timestamp_us > 0
    assert bundle.reference_route_world.ndim == 2
    assert bundle.reference_route_world.shape[1] == 3
    assert len(bundle.reference_route_world) >= 2
    assert len(bundle.navigation_routes_world) > 100
    navigation_points = np.concatenate(bundle.navigation_routes_world, axis=0)
    assert np.ptp(navigation_points[:, 0]) > 200.0
    assert np.ptp(navigation_points[:, 1]) > 200.0
    assert len(bundle.line_layers) > 0
    assert any(layer.color_rgba == (1.0, 1.0, 0.0, 1.0) for layer in bundle.line_layers)
    assert any(
        layer.layer_name == "traffic_signs" and len(layer.triangles_world) > 0
        for layer in bundle.triangle_layers
    )
    assert any(
        layer.layer_name == "crosswalks" and len(layer.polygons_world) > 0
        for layer in bundle.polygon_layers
    )
    assert len(bundle.vehicle_bbox_tracks) > 0
    sample_track = bundle.vehicle_bbox_tracks[0]
    assert sample_track.object_type in BBOX_V3_COLORS
    assert (
        sample_track.interpolate_at_timestamp(bundle.initial_timestamp_us) is not None
    )
