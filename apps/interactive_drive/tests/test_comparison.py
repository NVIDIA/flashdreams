# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check visible trajectory clipping, steering conversion and evidence limits."""

import math

import numpy as np
import pytest
from interactive_drive.camera import FThetaCameraModel
from interactive_drive.comparison import ComparisonRenderer
from interactive_drive.types import CameraCalibration
from PIL import Image, ImageColor, ImageDraw

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("lens", ["pinhole", "brown", "fisheye"])
def test_ribbon_clips_through_near_plane_to_image_bottom_at_vehicle_width(lens):
    mount = np.array(
        [[0, 0, 1, 1.9], [-1, 0, 0, 0], [0, -1, 0, 0.7], [0, 0, 0, 1]], dtype=float
    )
    camera = {
        "sensor_to_ego_matrix": mount,
        "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
        "width": 1920,
        "height": 1080,
        "D": [0] * 8,
    }
    if lens == "brown":
        camera["D"] = [
            0.809254015,
            -0.0114217,
            -2.8852e-05,
            -7.845e-06,
            -0.009098022,
            1.209575384,
            0.203171112,
            -0.034187955,
        ]
    if lens == "fisheye":
        flu = np.eye(4, dtype=np.float32)
        flu[:3, 3] = [1.9, 0, 0.7]
        camera = FThetaCameraModel(
            CameraCalibration(
                "front",
                "front",
                1920,
                1080,
                960,
                540,
                np.array([0, 0.001, 0, 0], dtype=np.float32),
                True,
                np.array([1, 0, 0], dtype=np.float32),
                flu,
            )
        )
    frames = []
    for distance in np.sort(np.r_[np.arange(0, 21, 0.5), 2.48]):
        pose = np.eye(4)
        pose[0, 3] = distance
        frames.append({"pose": pose})
    source = {
        "frames": frames,
        "times": np.arange(len(frames)) * 100_000,
        "floor_offset": 0.8,
        "ribbon_half_width": 1,
        "front_camera": camera,
    }
    renderer = ComparisonRenderer.__new__(ComparisonRenderer)
    widths = []
    for half_width in (0.5, 1.0):
        source["ribbon_half_width"] = half_width
        polygons = renderer.front_ribbon(source, 0)
        assert polygons
        points = np.concatenate(polygons)
        assert (
            np.isfinite(points).all()
            and points.min() >= -1e-7
            and points.max() <= 1 + 1e-7
        )
        assert points[:, 1].min() >= 0.5, (
            "Ground points must not fold above the horizon"
        )
        bottom = points[np.isclose(points[:, 1], 1)]
        assert len(bottom) >= 2, (
            "A ribbon crossing the near plane must reach the bottom"
        )
        widths.append(np.ptp(bottom[:, 0]))
    assert 1.5 < widths[1] / widths[0] < 2.5
    for frame in frames:
        frame["pose"][0, 3] = 0
    assert renderer.front_ribbon(source, 0) == []


def test_wheel_uses_ratio_and_turns_counterclockwise_for_left_steering():
    renderer = ComparisonRenderer.__new__(ComparisonRenderer)
    renderer.steering_ratio, renderer.speed_scale = 15, 60
    accent = "#39c7aa"
    for sign, spoke_x, empty_x in ((1, 249, 207), (-1, 207, 249)):
        image = Image.new("RGB", (292, 208))
        renderer.instruments(
            ImageDraw.Draw(image),
            0,
            0,
            {
                "speed": 0,
                "steer": sign * math.radians(6),
                "throttle": 1e-10,
                "brake": 0,
            },
            accent,
        )
        assert image.getpixel((spoke_x, 78)) == ImageColor.getrgb(accent)
        assert image.getpixel((empty_x, 78)) != ImageColor.getrgb(accent)
        assert image.getpixel((120, 164)) != ImageColor.getrgb(accent), (
            "A zero pedal must have an empty bar"
        )


def test_ribbon_has_side_outlines_without_internal_frame_crossbars(tmp_path):
    path = tmp_path / "front.jpg"
    Image.new("RGB", (200, 200), (30, 30, 30)).save(path)
    renderer = ComparisonRenderer.__new__(ComparisonRenderer)
    image = Image.new("RGB", (202, 202))
    ribbon = [
        np.array([[0.25, a], [0.75, a], [0.75, b], [0.25, b]])
        for a, b in ((0.25, 0.5), (0.5, 0.8))
    ]
    renderer.camera(image, (0, 0, 202, 202), path, "FRONT", "#39c7aa", ribbon=ribbon)
    border = image.getpixel((51, 110))[1] - 30
    interior = image.getpixel((101, 110))[1] - 30
    seam = image.getpixel((101, 101))[1] - 30
    assert border > interior * 1.5
    assert abs(seam - interior) <= 2


def test_evaluation_does_not_pass_missing_checks_or_call_distance_a_route_check(
    tmp_path,
):
    (tmp_path / "map.json").write_text('{"lines": []}')
    (tmp_path / "scene.json").write_text('{"palette": {}}')
    (tmp_path / "tracks.json").write_text("[]")
    frames = []
    for i in range(4):
        pose = np.eye(4)
        pose[:2, 3] = [i, 0]
        frames.append(
            {
                "timestamp_us": i * 100_000,
                "pose": pose,
                "speed": 10,
                "steer": 0,
                "throttle": 0,
                "brake": 0,
                "acceleration": 4,
                "jerk": 0,
                "travel": i,
                "lateral_acceleration": 0,
            }
        )
    source = {"frames": frames, "times": np.arange(4) * 100_000}
    renderer = ComparisonRenderer(source, source, tmp_path, {})
    groups = dict(renderer.evaluation(2))
    assert all(
        status is None
        for group in ("SAFETY", "COMPLIANCE")
        for _, status in groups[group]
    )
    assert groups["NAVIGATION"] == [("GT Path Offset 0.00 m", "value")]
    assert dict(groups["COMFORT"])["Harsh Acceleration"] == "fail"
    assert dict(groups["COMFORT"])["Phantom Brake"] is None
    with pytest.raises(ValueError, match="Steering ratio"):
        ComparisonRenderer(source, source, tmp_path, {"steering_ratio": float("nan")})
    with pytest.raises(ValueError, match="Comfort display"):
        ComparisonRenderer(source, source, tmp_path, {"comfort_limits": {"jerk": -1}})
    for frame in frames:
        frame["speed"] = 0
    frames[2]["speed"] = 0.15
    moving = ComparisonRenderer(source, source, tmp_path, {})
    assert moving.start == 0 and moving.left["time_origin_us"] == 200_000
    assert moving.left["frames"][0]["source_timestamp_us"] == 200_000
    assert moving.left["frames"][0]["timestamp_us"] == 0
    assert frames[2]["timestamp_us"] == 200_000
    assert isinstance(moving.start, int) and isinstance(moving.end, int)
    frames[2]["speed"] = 0
    stopped = ComparisonRenderer(source, source, tmp_path, {})
    assert stopped.start == 0, "An entirely stationary clip must still render"


def test_motion_time_zero_preserves_source_and_recorded_actor_clocks():
    renderer = ComparisonRenderer.__new__(ComparisonRenderer)
    renderer.lines, renderer.ego_sprite = [], None
    renderer.tracks = [
        {
            "timestamps_us": [1_000_000],
            "translations": [[10, 2, 0]],
            "quaternions_xyzw": [[0, 0, 0, 1]],
            "scale": [4.5, 1.8, 1.6],
        }
    ]
    frame = {
        "pose": np.eye(4),
        "timestamp_us": 0,
        "source_timestamp_us": 1_000_000,
        "actors": [],
    }
    source = {
        "times": np.array([0]),
        "frames": [frame],
        "floor_offset": 0.8,
        "ribbon_half_width": 1,
    }
    predicted = np.array(renderer.map_view(source, 0, (632, 350), "#39c7aa"))
    assert (
        np.count_nonzero(
            predicted[..., 2].astype(int) - predicted[..., 0].astype(int) > 40
        )
        > 50
    )
    frame["actors"] = [
        {
            "translation_world": [10, 2, 0],
            "orientation_xyzw": [0, 0, 0, 1],
            "dimensions_lwh": [4.5, 1.8, 1.6],
        }
    ]
    recorded = np.array(renderer.map_view(source, 0, (632, 350), "#39c7aa"))
    np.testing.assert_array_equal(predicted, recorded)
