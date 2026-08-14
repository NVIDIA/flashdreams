# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np
import pytest
from crazy_robotaxi.scene import (
    _build_fallback_perimeter,
    _build_lane_centerlines,
    _build_lane_network_perimeter,
    _build_navigation_lanes,
    load_scene_data,
)
from omnidreams_game_engine._sample_assets import SAMPLE_SCENE
from omnidreams_game_engine.colors import BBOX_V3_COLORS
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.scene_loader import (
    _discover_prompts,
    load_scene_bundle,
)
from shapely.geometry import Point, Polygon


def _point(x_m: float, y_m: float, z_m: float = 0.0) -> dict[str, float]:
    return {"x": x_m, "y": y_m, "z": z_m}


def _lane_row_from_rails(
    left_rail: tuple[dict[str, float], ...],
    right_rail: tuple[dict[str, float], ...],
    *,
    vehicle_types: tuple[str, ...] = ("CAR",),
) -> dict[str, Any]:
    return {
        "lane": {
            "left_rail": list(left_rail),
            "right_rail": list(right_rail),
            "vehicle_types": list(vehicle_types),
        }
    }


def _lane_row(
    *,
    start_x_m: float,
    end_x_m: float,
    center_y_m: float,
    map_end: str,
    use_types: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "lane": {
            "left_rail": [
                _point(start_x_m, center_y_m + 2.0),
                _point(end_x_m, center_y_m + 2.0),
            ],
            "right_rail": [
                _point(start_x_m, center_y_m - 2.0),
                _point(end_x_m, center_y_m - 2.0),
            ],
            "vehicle_types": ["CAR"],
            "map_end": map_end,
            "use_types": list(use_types),
        }
    }


def _boundary_row(*points: dict[str, float]) -> dict[str, Any]:
    return {"road_boundary": {"location": list(points)}}


def _assert_segments_form_closed_rings(segments: np.ndarray) -> None:
    discontinuities = np.flatnonzero(
        np.linalg.norm(segments[:-1, 1] - segments[1:, 0], axis=1) > 1.0e-4
    )
    ring_starts = (0, *(int(index) + 1 for index in discontinuities))
    ring_stops = (*(int(index) + 1 for index in discontinuities), len(segments))
    for start, stop in zip(ring_starts, ring_stops, strict=True):
        ring = segments[start:stop]
        np.testing.assert_allclose(ring[:, 1], np.roll(ring[:, 0], -1, axis=0))


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


def test_lane_network_perimeter_is_closed_beyond_lane_rails() -> None:
    lanes = [_lane_row(start_x_m=0.0, end_x_m=30.0, center_y_m=0.0, map_end="NONE")]

    perimeter = _build_lane_network_perimeter(
        lanes,
        np.asarray([5.0, 0.0], dtype=np.float32),
    )

    assert len(perimeter) >= 4
    _assert_segments_form_closed_rings(perimeter)
    assert float(perimeter[:, :, 0].min()) < 0.0
    assert float(perimeter[:, :, 0].max()) > 30.0
    assert float(perimeter[:, :, 1].min()) < -2.0
    assert float(perimeter[:, :, 1].max()) > 2.0


def test_lane_network_perimeter_wraps_connected_branches_without_internal_caps() -> (
    None
):
    lanes = [
        _lane_row(start_x_m=0.0, end_x_m=30.0, center_y_m=0.0, map_end="NONE"),
        _lane_row_from_rails(
            (_point(13.0, 0.0), _point(13.0, 20.0)),
            (_point(17.0, 0.0), _point(17.0, 20.0)),
        ),
    ]

    perimeter = _build_lane_network_perimeter(
        lanes,
        np.asarray([5.0, 0.0], dtype=np.float32),
    )
    ring = Polygon(perimeter[:, 0, :2])

    assert ring.is_valid
    assert ring.covers(Point(15.0, 10.0))


def test_lane_network_perimeter_encloses_inner_block_edge() -> None:
    lanes = [
        _lane_row(start_x_m=0.0, end_x_m=30.0, center_y_m=0.0, map_end="NONE"),
        _lane_row(start_x_m=0.0, end_x_m=30.0, center_y_m=30.0, map_end="NONE"),
        _lane_row_from_rails(
            (_point(-2.0, 0.0), _point(-2.0, 30.0)),
            (_point(2.0, 0.0), _point(2.0, 30.0)),
        ),
        _lane_row_from_rails(
            (_point(28.0, 0.0), _point(28.0, 30.0)),
            (_point(32.0, 0.0), _point(32.0, 30.0)),
        ),
    ]

    perimeter = _build_lane_network_perimeter(
        lanes,
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    inner_segments = perimeter[
        np.all(
            (perimeter[:, :, :2] >= 4.0) & (perimeter[:, :, :2] <= 26.0), axis=(1, 2)
        )
    ]

    _assert_segments_form_closed_rings(perimeter)
    assert len(inner_segments) >= 4
    _assert_segments_form_closed_rings(inner_segments)


def test_lane_network_perimeter_excludes_disconnected_parking_area() -> None:
    lanes = [
        _lane_row(start_x_m=0.0, end_x_m=30.0, center_y_m=0.0, map_end="NONE"),
        _lane_row(
            start_x_m=100.0,
            end_x_m=120.0,
            center_y_m=100.0,
            map_end="NONE",
            use_types=("SERVICE_ROAD",),
        ),
    ]

    perimeter = _build_lane_network_perimeter(
        lanes,
        np.asarray([5.0, 0.0], dtype=np.float32),
    )

    assert float(perimeter[:, :, 0].max()) < 100.0
    assert float(perimeter[:, :, 1].max()) < 100.0


def test_fallback_perimeter_is_closed_outside_navigation_extent() -> None:
    rows = [_lane_row(start_x_m=0.0, end_x_m=10.0, center_y_m=5.0, map_end="NONE")]

    boundary_rows = [_boundary_row(_point(-50.0, 0.0), _point(-40.0, 0.0))]

    perimeter = _build_fallback_perimeter(  # type: ignore[arg-type]
        rows, boundary_rows
    )

    assert perimeter.shape == (4, 2, 3)
    np.testing.assert_allclose(perimeter[:, 1], np.roll(perimeter[:, 0], -1, axis=0))
    assert float(perimeter[:, :, 0].min()) < -50.0
    assert float(perimeter[:, :, 0].max()) > 10.0
    assert float(perimeter[:, :, 1].min()) < 3.0
    assert float(perimeter[:, :, 1].max()) > 7.0


def test_navigation_lanes_keep_only_road_edges_as_stopping_surfaces() -> None:
    rows = [
        {
            "lane": {
                "left_rail": [_point(0.0, 2.0), _point(10.0, 2.0)],
                "right_rail": [_point(0.0, -2.0), _point(10.0, -2.0)],
                "left_edge_styles": ["LONG_DASHED_SINGLE", "LONG_DASHED_SINGLE"],
                "right_edge_styles": ["TALL_CURB", "TALL_CURB"],
                "left_edge_colors": ["WHITE", "WHITE"],
                "right_edge_colors": ["UNKNOWN", "UNKNOWN"],
                "vehicle_types": ["CAR"],
            }
        },
        {
            "lane": {
                "left_rail": [_point(0.0, 6.0), _point(10.0, 6.0)],
                "right_rail": [_point(0.0, 2.0), _point(10.0, 2.0)],
                "left_edge_styles": ["LONG_DASHED_SINGLE", "LONG_DASHED_SINGLE"],
                "right_edge_styles": ["TALL_CURB", "VIRTUAL"],
                "left_edge_colors": ["WHITE", "WHITE"],
                "right_edge_colors": ["UNKNOWN", "UNKNOWN"],
                "vehicle_types": ["CAR"],
            }
        },
    ]

    lanes = _build_navigation_lanes(rows)

    assert len(lanes) == 2
    assert lanes[0].allows_taxi_stops
    assert lanes[0].road_edge_world is not None
    np.testing.assert_allclose(
        lanes[0].road_edge_world,
        np.array([[0.0, -2.0, 0.0], [10.0, -2.0, 0.0]], dtype=np.float32),
    )
    assert not lanes[1].allows_taxi_stops
    assert lanes[1].road_edge_world is None


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
    scene_data = load_scene_data(bundle)
    assert scene_data.reference_route_world.ndim == 2
    assert scene_data.reference_route_world.shape[1] == 3
    assert len(scene_data.reference_route_world) >= 2
    assert len(scene_data.navigation_routes_world) > 100
    assert len(scene_data.navigation_lanes) > 100
    assert len(scene_data.perimeter_segments_world) > 100
    _assert_segments_form_closed_rings(scene_data.perimeter_segments_world)
    navigation_points = np.concatenate(scene_data.navigation_routes_world, axis=0)
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
