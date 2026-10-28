# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check one-sided road semantics using independent debug geometry."""

import numpy as np
import pytest
import torch
from ludus_renderer import (
    CAMERA_TYPE_BEV,
    CAMERA_TYPE_REGULAR,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_POLE,
    PRIM_CROSSWALK,
    TimestampedPolygonPool,
    FThetaCamera,
    LudusCudaTimestampedContext,
    TimestampedPolylinePool,
    TimestampedScene,
)
from ludus_renderer._ops.primitives import PRIM_DOT_WHITE
from omnidreams_game_engine.conditioning import _with_road_surface_sides

pytestmark = pytest.mark.ci_gpu


def _line(points):
    return TimestampedPolylinePool(
        timestamps_us=torch.tensor([0], dtype=torch.int64, device="cuda"),
        timestamped_varrays_prefix_sum=torch.tensor(
            [1], dtype=torch.int32, device="cuda"
        ),
        varrays_prefix_sum=torch.tensor(
            [len(points)], dtype=torch.int32, device="cuda"
        ),
        vertices=torch.tensor(points, dtype=torch.float32, device="cuda"),
        prim_type_id=PRIM_LANE_LINE_WHITE_SOLID,
    )


def _camera_pose(position):
    position = np.asarray(position, dtype=np.float32)
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float32)
    # Ludus accepts forward/left/up camera poses and converts to optical axes.
    pose[:3, :3] = np.column_stack((forward, -right, -down))
    pose[:3, 3] = position
    return torch.from_numpy(np.linalg.inv(pose)).to("cuda")


@pytest.fixture
def renderer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for semantic rendering")
    context = LudusCudaTimestampedContext(device="cuda")
    context.set_depth_scaling(False)
    context.upload_cameras(
        [
            FThetaCamera(
                principal_point=torch.tensor(
                    [64, 64], dtype=torch.float32, device="cuda"
                ),
                image_size=torch.tensor([128, 128], dtype=torch.float32, device="cuda"),
                fw_poly=torch.tensor(
                    [0, 80, 0, 0, 0, 0], dtype=torch.float32, device="cuda"
                ),
                max_ray_angle=1.2,
                depth_max=100,
            )
        ]
    )
    yield context
    context.clear_scenes()


def _render(context, scene, camera, camera_type=CAMERA_TYPE_REGULAR):
    context.clear_scenes()
    scene_id = context.upload_scene(scene)
    return context.render_uniform(
        scene_id=scene_id,
        camera_id=0,
        timestamps_us=[0],
        camera_type_id=camera_type,
        camera_poses=_camera_pose(camera)[None],
        resolution=(128, 128),
    )[..., :3].cpu()


@pytest.mark.parametrize("grade", [(0, 0), (0.3, -0.2)])
@pytest.mark.parametrize("reverse_winding", [False, True])
def test_ground_hides_roads_only_from_below(renderer, grade, reverse_winding):
    # A separate synthetic debug surface, independent of all playable maps.
    vertices = np.array(
        [[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], np.float32
    )
    vertices[:, 2] = vertices[:, :2] @ np.asarray(grade)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    if reverse_winding:
        faces = faces[:, ::-1].copy()
    points = [[x, 0, x * grade[0] + 0.03] for x in (-5, 0, 5)]
    plain = TimestampedScene([_line(points)], [])
    sided = _with_road_surface_sides(plain, vertices, faces)
    # On the ramp, this lower camera has positive world Z but is still below
    # the local road plane. A global camera-height test would get it wrong.
    below = (10, 0, 1) if grade[0] else (0, 0, -5)
    for camera, hidden in [((0, 0, 5), False), (below, True)]:
        baseline = _render(renderer, plain, camera)
        actual = _render(renderer, sided, camera)
        assert torch.count_nonzero(baseline) > 30
        if hidden:
            assert torch.count_nonzero(actual) == 0
        else:
            assert torch.equal(actual, baseline)
    baseline = _render(renderer, plain, below, CAMERA_TYPE_BEV)
    actual = _render(renderer, sided, below, CAMERA_TYPE_BEV)
    assert torch.equal(actual, baseline)


def test_road_culling_preserves_non_surface_geometry(renderer):
    vertices = np.array(
        [[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], np.float32
    )
    overhead = _line([[-3, 0, 0.03], [3, 0, 0.03]])
    foreground = _line([[-1, 0.5, -3], [1, 0.5, -3]])
    foreground.prim_type_id = PRIM_POLE
    scene = _with_road_surface_sides(
        TimestampedScene([overhead, foreground], []),
        vertices,
        np.array([[0, 1, 2], [0, 2, 3]], np.int32),
    )
    expected = _render(renderer, TimestampedScene([foreground], []), (0, 0, -5))
    actual = _render(renderer, scene, (0, 0, -5))
    assert torch.count_nonzero(expected) > 30
    assert torch.equal(actual, expected)


def test_dots_and_crosswalk_polygons_are_one_sided(renderer):
    vertices = np.array(
        [[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    dots = _line([[-2, 0, 0.03], [2, 0, 0.03]])
    dots.prim_type_id = PRIM_DOT_WHITE
    paint = TimestampedPolygonPool(
        timestamps_us=dots.timestamps_us,
        timestamped_varrays_prefix_sum=dots.timestamped_varrays_prefix_sum,
        varrays_prefix_sum=torch.tensor([3], dtype=torch.int32, device="cuda"),
        triangle_prefix_sum=torch.tensor([1], dtype=torch.int32, device="cuda"),
        vertices=torch.tensor(
            [[-1, -1, 0.03], [1, -1, 0.03], [0, 1, 0.03]], device="cuda"
        ),
        triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda"),
        prim_type_id=PRIM_CROSSWALK,
    )
    plain = TimestampedScene([dots], [paint])
    sided = _with_road_surface_sides(plain, vertices, faces)
    for position, below in [((0, 0, 5), False), ((0, 0, -5), True)]:
        baseline = _render(renderer, plain, position)
        actual = _render(renderer, sided, position)
        assert torch.count_nonzero(baseline) > 30
        if below:
            assert torch.count_nonzero(actual) == 0
        else:
            assert torch.equal(actual, baseline)


def test_polyline_can_be_above_one_deck_and_below_another(renderer):
    road = _line([[-4, 0, 0], [-2, 0, 0], [2, 0, 2], [4, 0, 2]])
    camera = (0, -8, 1)
    baseline = _render(renderer, TimestampedScene([road], []), camera)
    road.surface_planes = torch.tensor(
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, -2], [0, 0, 1, -2]],
        dtype=torch.float32,
        device="cuda",
    )
    actual = _render(renderer, TimestampedScene([road], []), camera)
    upper_end = _render(
        renderer, TimestampedScene([_line([[3, 0, 2], [4, 0, 2]])], []), camera
    )
    assert 30 < torch.count_nonzero(actual) < torch.count_nonzero(baseline)
    assert torch.count_nonzero(upper_end) > 30
    assert torch.count_nonzero(actual[upper_end > 0]) == 0
