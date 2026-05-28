# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from pathlib import Path

from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.scene_fixture import build_synthetic_scene_usdz
from omnidreams.interactive_drive.scene_loader import load_scene_bundle


def test_build_synthetic_scene_usdz_round_trip() -> None:
    fixture_path = build_synthetic_scene_usdz(
        Path(__file__).resolve().parent / "output" / "synthetic_scene_fixture.usdz"
    )
    bundle = load_scene_bundle(
        scene_path=fixture_path,
        camera_name="camera_front_wide_120fov",
        variant="1",
        prompt_override=None,
        raster=RasterConfig(width=640, height=352),
    )

    assert bundle.scene_id == "synthetic-test-scene"
    assert bundle.selected_camera.logical_name == "camera_front_wide_120fov"
    assert bundle.initial_rgb.shape == (352, 640, 3)
    assert bundle.prompt == "Synthetic prompt variant 1."
    assert bundle.initial_timestamp_us > 0
    assert 9.0 <= bundle.initial_speed_mps <= 11.0

    line_names = {layer.layer_name for layer in bundle.line_layers}
    assert any(name.startswith("lanelines_") for name in line_names)
    assert "road_boundaries" in line_names
    assert "wait_lines" in line_names
    assert "poles" in line_names
    assert "traffic_lights" in line_names

    triangle_names = {layer.layer_name for layer in bundle.triangle_layers}
    assert triangle_names == {"traffic_signs"}
    assert bundle.triangle_layers[0].triangles_world.shape[0] > 0

    polygon_names = {layer.layer_name for layer in bundle.polygon_layers}
    assert polygon_names == {
        "crosswalks",
        "road_markings",
        "intersection_areas",
        "road_islands",
    }
    assert all(len(layer.polygons_world) > 0 for layer in bundle.polygon_layers)

    # One track per BBOX_V3_COLORS category (Car, Truck, Pedestrian, Cyclist,
    # Others). Each track has two samples covering the full trajectory span so
    # ``interpolate_at_timestamp`` resolves at every render frame.
    track_types = {track.object_type for track in bundle.vehicle_bbox_tracks}
    assert track_types == {"Car", "Truck", "Pedestrian", "Cyclist", "Others"}
    for track in bundle.vehicle_bbox_tracks:
        assert len(track.timestamps_us) == 2
        assert track.interpolate_at_timestamp(bundle.initial_timestamp_us) is not None
