# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytic P/T cases for spinning-LiDAR projection (IDs follow the specification)."""

import math
import warnings

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    _lidar_projection as projection,
)

pytestmark = pytest.mark.ci_cpu

RINGS = np.array([0.25, 0.0, -0.25])
COLUMNS = 8
STEP = 2 * math.pi / COLUMNS


def toy(offsets=(0, 0, 0), columns=COLUMNS, **kwargs):
    return projection.SpinningLidarModel(
        RINGS, np.array(offsets, float), columns, **kwargs
    )


def synthetic_128(columns=1800, **kwargs):
    return projection.SpinningLidarModel(
        np.linspace(0.25, -0.45, 128), np.zeros(128), columns, **kwargs
    )


def pose(yaw=0.0, translation=(0, 0, 0)):
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    out[:3, 3] = translation
    return out


def direction(elevation, azimuth, length=1.0):
    return length * np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )


def centre_azimuth(column, columns=COLUMNS):
    return math.pi - (column + 0.5) * 2 * math.pi / columns


def T(array, dtype=torch.float64):
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def test_P01_construction_and_metadata():
    elevations, offsets = RINGS.copy(), np.zeros(3)
    model = projection.SpinningLidarModel(elevations, offsets, 20, 10, "toy")
    assert (model.n_rows, model.shape, model.sensor_name) == (3, (3, 20), "toy")
    np.testing.assert_allclose(model.row_elevations_deg, np.degrees(RINGS))
    np.testing.assert_allclose(
        model.column_azimuths_rad, math.pi - np.arange(20) * 2 * math.pi / 20
    )
    assert model.sweep_duration_us == 100_000
    assert np.array_equal(elevations, RINGS) and np.array_equal(offsets, np.zeros(3))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(row_elevations_rad=np.empty(0), row_azimuth_offsets_rad=np.empty(0)),
        dict(row_elevations_rad=RINGS, row_azimuth_offsets_rad=np.zeros(2)),
        dict(
            row_elevations_rad=np.array([np.nan, 0]),
            row_azimuth_offsets_rad=np.zeros(2),
        ),
        dict(
            row_elevations_rad=RINGS, row_azimuth_offsets_rad=np.array([0, np.inf, 0])
        ),
        dict(
            row_elevations_rad=RINGS, row_azimuth_offsets_rad=np.zeros(3), n_columns=1
        ),
        dict(
            row_elevations_rad=RINGS, row_azimuth_offsets_rad=np.zeros(3), n_columns=2.5
        ),
        dict(
            row_elevations_rad=RINGS,
            row_azimuth_offsets_rad=np.zeros(3),
            spinning_frequency_hz=0,
        ),
        dict(
            row_elevations_rad=RINGS,
            row_azimuth_offsets_rad=np.zeros(3),
            spinning_frequency_hz=float("nan"),
        ),
        dict(
            row_elevations_rad=RINGS,
            row_azimuth_offsets_rad=np.zeros(3),
            sorted_ring_lookup="yes",
        ),
        dict(
            row_elevations_rad=RINGS[::-1],
            row_azimuth_offsets_rad=np.zeros(3),
            sorted_ring_lookup=True,
        ),
        dict(
            row_elevations_rad=np.array([0.1, 0.1]),
            row_azimuth_offsets_rad=np.zeros(2),
            sorted_ring_lookup=True,
        ),
    ],
)
def test_P02_invalid_models_rejected(kwargs):
    with pytest.raises(ValueError):
        projection.SpinningLidarModel(**kwargs)


def test_P03_P04_centre_rays_and_offsets_do_not_move_rays():
    model, shifted = toy(), toy((STEP, 0, -STEP))
    rays = model.ray_directions()
    assert rays.shape == (3, COLUMNS, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-12)
    for r, elevation in enumerate(RINGS):
        for c in range(COLUMNS):
            np.testing.assert_allclose(
                rays[r, c], direction(elevation, centre_azimuth(c)), atol=1e-12
            )
    np.testing.assert_array_equal(rays, shifted.ray_directions())
    assert (
        rays[1, COLUMNS // 2 - 1, 0] > 0.9 and rays[1, COLUMNS // 2, 0] > 0.9
    )  # forward
    assert rays[1, 0, 0] < 0  # first column is at the rear
    # clockwise: successive columns decrease azimuth
    az = np.arctan2(rays[1, :, 1], rays[1, :, 0])
    assert np.all(np.diff(np.unwrap(az)) < 0)


def test_P03_inverse_binning_recovers_every_cell():
    model = synthetic_128(columns=64)
    rays = torch.as_tensor(model.ray_directions().reshape(-1, 3) * 7)
    ranges, _ = projection.points_to_range_map(rays, model)
    assert bool((ranges == 7).all())


@pytest.mark.parametrize("sorted_lookup", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_P05_nearest_rows(sorted_lookup, dtype):
    model = toy(sorted_ring_lookup=sorted_lookup)
    q = T([0.25, 0, -0.25, 0.3, -0.5, 0.125, -0.125, 0.12, -0.13, 0.2], dtype)
    expected = [0, 1, 2, 0, 2, 0, 1, 1, 2, 0]  # midpoints tie to the lower row
    assert model.nearest_rows(q).tolist() == expected
    assert model.nearest_rows(q).dtype == torch.int64
    assert model.nearest_rows(q[:0]).shape == (0,)


def test_P05_sorted_path_matches_baseline_on_large_and_adjacent_queries():
    rings = np.linspace(0.25, -0.45, 128)
    base, fast = (
        projection.SpinningLidarModel(rings, np.zeros(128), 10, sorted_ring_lookup=s)
        for s in (False, True)
    )
    gen = torch.Generator().manual_seed(17)
    q = (torch.rand(300_000, generator=gen, dtype=torch.float64) - 0.8) * 1.2
    mids = torch.as_tensor((rings[:-1] + rings[1:]) / 2)
    q = torch.cat(
        (q, mids, torch.nextafter(mids, mids + 1), torch.nextafter(mids, mids - 1))
    )
    assert torch.equal(base.nearest_rows(q, chunk_size=1000), fast.nearest_rows(q))
    q32 = q.float()
    assert torch.equal(base.nearest_rows(q32), fast.nearest_rows(q32))


def test_P06_sensor_angles_axes_and_validity():
    model = toy()
    pts = T(
        [
            [10, 0, 0],
            [0, 10, 0],
            [-10, 0, 0],
            [0, -10, 0],
            [10, 0, 10],
            [10, 0, -10],
            [0, 0, 0],
        ]
    )
    elevation, azimuth, valid = model.sensor_angles(pts)
    np.testing.assert_allclose(
        azimuth[:4].numpy(), [0, math.pi / 2, math.pi, -math.pi / 2], atol=1e-12
    )
    assert valid.tolist() == [True, True, True, True, False, False, False]
    assert elevation[6] == 0 and azimuth[6] == 0  # finite placeholders
    edge = lambda e: T([[math.cos(e), 0, math.sin(e)]])  # noqa: E731
    assert bool(model.sensor_angles(edge(0.25 + 0.9e-6))[2])
    assert not bool(model.sensor_angles(edge(0.25 + 2e-6))[2])


def test_P07_firing_time_and_offsets():
    model = toy()
    leading = [math.pi - c * STEP for c in range(COLUMNS)]
    pts = T([direction(0.0, a, 5) for a in leading])
    time, valid = model.relative_frame_times(pts)
    np.testing.assert_allclose(
        time.numpy(), np.arange(COLUMNS) / (COLUMNS - 1), atol=1e-12
    )
    assert valid.all() and time.min() == 0 and time.max() == 1
    shifted = toy((STEP, 0, -STEP))
    up = T([direction(0.25, a, 5) for a in leading])
    down = T([direction(-0.25, a, 5) for a in leading])
    mid = T([direction(0.0, a, 5) for a in leading])
    # positive offset: the point at leading edge + offset fires at that column's time
    t_up = shifted.relative_frame_times(
        T([direction(0.25, a + STEP, 5) for a in leading])
    )[0]
    np.testing.assert_allclose(
        t_up.numpy(), np.arange(COLUMNS) / (COLUMNS - 1), atol=1e-9
    )
    t_down = shifted.relative_frame_times(
        T([direction(-0.25, a - STEP, 5) for a in leading])
    )[0]
    np.testing.assert_allclose(
        t_down.numpy(), np.arange(COLUMNS) / (COLUMNS - 1), atol=1e-9
    )
    np.testing.assert_allclose(
        shifted.relative_frame_times(mid)[0].numpy(), time.numpy(), atol=1e-12
    )
    assert up.shape == down.shape


def test_P08_rear_seam_signed_zero():
    model = toy()
    pts = torch.tensor([[-10.0, 0.0, 0.0], [-10.0, -0.0, 0.0]], dtype=torch.float64)
    ranges, _ = projection.points_to_range_map(pts, model)
    assert ranges[1, 0] == 10 and ranges[1, COLUMNS - 1] == 10
    assert int((ranges > 0).sum()) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_P09_full_range_map_and_defaults(dtype):
    model = synthetic_128()
    rays = T(model.ray_directions().reshape(-1, 3) * 12, dtype)
    ranges, intensity = projection.points_to_range_map(rays, model)
    assert ranges.shape == intensity.shape == (128, 1800)
    assert ranges.dtype == intensity.dtype == torch.float32
    torch.testing.assert_close(
        ranges, torch.full_like(ranges, 12), atol=1e-4, rtol=1e-5
    )
    assert not intensity.any()


def test_P10_P11_P15_occlusion_and_intensity():
    model = toy()
    ray = direction(0.0, centre_azimuth(2))
    depths = [20, 5, 12, 99.999, 100, 100.001, 2]
    pts = T([ray * d for d in depths])
    intensity = T([1, 2, 3, 4, 5, 6, 7])
    ranges, values = projection.points_to_range_map(pts, model, intensity=intensity)
    assert ranges[1, 2] == pytest.approx(2, abs=1e-6) and values[1, 2] == 7
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    r2, v2 = projection.points_to_range_map(pts[perm], model, intensity=intensity[perm])
    assert torch.equal(r2, ranges) and torch.equal(v2, values)
    # ties: lowest point index wins, never a farther point's intensity
    tie = T([ray * 5, ray * 5, ray * 9])
    r, v = projection.points_to_range_map(tie, model, intensity=T([10, 20, 30]))
    assert r[1, 2] == 5 and v[1, 2] == 10
    r, v = projection.points_to_range_map(tie.flip(0), model, intensity=T([30, 20, 10]))
    assert r[1, 2] == 5 and v[1, 2] == 20


def test_P12_range_thresholds():
    model = toy()
    ray = np.array([1.0, 0, 0])
    azimuth_cell = int(COLUMNS / 2)  # +x lies on the leading edge of this column
    one = lambda d: T([[d, 0.0, 0.0]], torch.float64)  # noqa: E731
    for d, hit in (
        (99.999, True),
        (100.0, False),
        (math.nextafter(100.0, 200.0), False),
        (math.nextafter(100.0, 0.0), True),
        (0.0, False),
    ):
        ranges, _ = projection.points_to_range_map(one(d), model)
        assert bool(ranges.any()) is hit, d
    assert not projection.points_to_range_map(
        torch.empty(0, 3, dtype=torch.float64), model
    )[0].any()
    assert ray.shape == (3,) and azimuth_cell == 4


def test_P13_elevation_acceptance_margin():
    model = toy()
    margin = math.radians(0.1)
    inside = (
        T([[math.cos(0.25 + margin * 0.99), 0.0, math.sin(0.25 + margin * 0.99)]]) * 10
    )
    outside = (
        T([[math.cos(0.25 + margin * 1.01), 0.0, math.sin(0.25 + margin * 1.01)]]) * 10
    )
    assert projection.points_to_range_map(inside, model)[0][0].any()
    assert not projection.points_to_range_map(outside, model)[0].any()


@pytest.mark.parametrize(
    "call",
    [
        lambda m: projection.points_to_range_map(torch.zeros(2, 2), m),
        lambda m: projection.points_to_range_map(torch.zeros(3), m),
        lambda m: projection.points_to_range_map(
            torch.zeros(2, 3, dtype=torch.int64), m
        ),
        lambda m: projection.points_to_range_map(
            torch.tensor([[1.0, float("nan"), 0]]), m
        ),
        lambda m: projection.points_to_range_map(
            torch.tensor([[1.0, float("inf"), 0]]), m
        ),
        lambda m: projection.points_to_range_map(
            torch.ones(2, 3), m, intensity=torch.ones(1)
        ),
        lambda m: projection.points_to_range_map(
            torch.ones(2, 3), m, intensity=torch.ones(2, 1)
        ),
        lambda m: projection.points_to_range_map(
            torch.ones(2, 3), m, intensity=torch.ones(2, device="meta")
        ),
        lambda m: projection.points_to_range_map(torch.ones(2, 3), m, max_range=0),
        lambda m: projection.points_to_range_map(
            torch.ones(2, 3), m, max_range=float("nan")
        ),
        lambda m: projection.points_to_range_map(torch.ones(2, 3), m, max_range=-1),
    ],
)
def test_P14_malformed_inputs_rejected(call):
    with pytest.raises(ValueError):
        call(toy())


def test_P16_noncontiguous_inputs_match_contiguous_copy():
    model = toy()
    base = torch.as_tensor(model.ray_directions().reshape(-1, 3) * 9)
    wide = torch.zeros(len(base), 6, dtype=torch.float64)
    wide[:, ::2] = base
    view = wide[:, ::2]
    assert not view.is_contiguous()
    before = view.clone()
    a = projection.points_to_range_map(view, model)
    b = projection.points_to_range_map(base.clone(), model)
    assert torch.equal(a[0], b[0]) and torch.equal(view, before)


def test_T01_T02_T03_interpolation():
    start, end = pose(), pose(math.pi / 2, (2, 0, 0))
    t = T([0, 0.5, 1])
    rotations, translations = projection.interpolate_rigid(T(start), T(end), t)
    for r, yaw in zip(rotations, (0, math.pi / 4, math.pi / 2)):
        np.testing.assert_allclose(r.numpy(), pose(yaw)[:3, :3], atol=1e-12)
        assert abs(float(torch.linalg.det(r)) - 1) < 1e-12
    np.testing.assert_allclose(
        translations.numpy(), [[0, 0, 0], [1, 0, 0], [2, 0, 0]], atol=1e-12
    )
    same = projection.interpolate_rigid(T(start), T(start), t)
    np.testing.assert_allclose(same[0].numpy(), np.stack([np.eye(3)] * 3), atol=1e-12)
    near = pose(1e-12)
    assert torch.isfinite(projection.interpolate_rigid(T(start), T(near), t)[0]).all()
    # near 180 degrees: shortest arc, proper rotations, no reflection
    r180 = projection.interpolate_rigid(T(start), T(pose(math.pi - 1e-9)), t)[0]
    assert torch.isfinite(r180).all()
    np.testing.assert_allclose(torch.linalg.det(r180).numpy(), 1, atol=1e-9)
    assert projection.interpolate_rigid(T(start), T(end), T([]))[0].shape == (0, 3, 3)
    batched = projection.interpolate_rigid(T(start), T(end), T([1, 0, 0.5]))
    np.testing.assert_allclose(batched[1][:, 0].numpy(), [2, 0, 1], atol=1e-12)


def test_T04_apply_rigid_inverse_round_trip():
    rotations = T(np.stack([pose(0.3)[:3, :3], pose(-1.0)[:3, :3]]))
    translations = T([[1, 2, 3], [-4, 0, 5]])
    sensor = T([[1, 0, 0], [0, 2, -1]])
    world = torch.einsum("nij,nj->ni", rotations, sensor) + translations
    torch.testing.assert_close(
        projection.apply_rigid_inverse(rotations, translations, world), sensor
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: projection.interpolate_rigid(T(np.eye(3)), T(pose()), T([0.5])),
        lambda: projection.interpolate_rigid(
            T(np.diag([1, 1, -1, 1.0])), T(pose()), T([0.5])
        ),
        lambda: projection.interpolate_rigid(
            T(np.full((4, 4), np.nan)), T(pose()), T([0.5])
        ),
        lambda: projection.interpolate_rigid(T(pose()), T(pose()), T([1.5])),
        lambda: projection.interpolate_rigid(T(pose()), T(pose()), T([-0.1])),
        lambda: projection.interpolate_rigid(T(pose()), T(pose()), T([float("nan")])),
        lambda: projection.apply_rigid_inverse(
            torch.zeros(2, 3, 3), torch.zeros(2, 3), torch.zeros(3, 3)
        ),
    ],
)
def test_T05_invalid_rigid_inputs_rejected(call):
    with pytest.raises(ValueError):
        call()


def world_cloud(model, length=12.0):
    return T(model.ray_directions().reshape(-1, 3) * length)


def test_T06_T07_stationary_sweeps():
    model = synthetic_128(columns=64)
    cloud = world_cloud(model)
    out = projection.world_points_to_shutter_sensor_frame(model, cloud, pose(), pose())
    assert torch.equal(out.valid_indices, torch.arange(len(cloud)))
    torch.testing.assert_close(out.points_sensor, cloud)
    expected, _ = model.relative_frame_times(cloud)
    torch.testing.assert_close(out.relative_time, expected)
    moved = pose(0.7, (10, -3, 2))
    world = cloud @ T(moved[:3, :3]).T + T(moved[:3, 3])
    out = projection.world_points_to_shutter_sensor_frame(model, world, moved, moved)
    torch.testing.assert_close(out.points_sensor, cloud, atol=1e-9, rtol=0)
    common = pose(-1.1, (100, 50, -7))

    def rigid(p):
        return p @ T(common[:3, :3]).T + T(common[:3, 3])

    both = projection.world_points_to_shutter_sensor_frame(
        model, rigid(world), common @ moved, common @ moved
    )
    torch.testing.assert_close(both.points_sensor, out.points_sensor, atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "start,end",
    [
        (pose(), pose(0, (3, 0, 0))),
        (pose(), pose(0.4)),
        (pose(0, (2, 1, 0)), pose(-0.3, (0, 0, 0))),
        (pose(), pose(0, (-3, 0, 0))),
    ],
)
def test_T08_reconstruction_at_reported_time(start, end):
    model = synthetic_128(columns=64)
    cloud = world_cloud(model, 15.0)
    out = projection.world_points_to_shutter_sensor_frame(
        model, cloud, start, end, max_error=True
    )
    assert len(out.valid_indices) > 0
    rotations, translations = projection.interpolate_rigid(
        T(start), T(end), out.relative_time
    )
    forward = torch.einsum("nij,nj->ni", rotations, out.points_sensor) + translations
    torch.testing.assert_close(forward, cloud[out.valid_indices], atol=1e-9, rtol=0)
    static = projection.world_points_to_shutter_sensor_frame(model, cloud, start, start)
    if not np.array_equal(start, end):
        n = min(len(static.points_sensor), len(out.points_sensor))
        assert n > 0 and not torch.equal(
            static.points_sensor[:n], out.points_sensor[:n]
        )


def test_T09_self_consistent_columns_against_slow_oracle():
    model = toy(columns=16)
    start, end = pose(), pose(0.1, (1.5, 0, 0))
    cloud = T([direction(0.0, a, 8) for a in np.linspace(-3, 3, 13)])
    out = projection.world_points_to_shutter_sensor_frame(
        model,
        cloud,
        start,
        end,
        max_iterations=50,
        stop_mean_relative_time_error=1e-9,
        stop_delta_mean_relative_time_error=1e-12,
        max_error=True,
    )
    for index, time, sensor in zip(
        out.valid_indices, out.relative_time, out.points_sensor
    ):
        # oracle: scan candidate times; a self-consistent one reproduces its own firing time
        grid = torch.linspace(0, 1, 20001, dtype=torch.float64)
        r, p = projection.interpolate_rigid(T(start), T(end), grid)
        local = projection.apply_rigid_inverse(r, p, cloud[index].expand(len(grid), 3))
        implied, _ = model.relative_frame_times(local)
        crossings = grid[(implied - grid).abs() < 2e-4]
        assert len(crossings) and (crossings - time).abs().min() < 2e-3


def test_T10_field_of_view_changes_during_motion():
    model = toy()
    cloud = T([[10, 0, 2.7], [10, 0, 2.5], [10, 0, -0.1]])
    start, end = pose(), pose(0, (6, 0, 0))
    out = projection.world_points_to_shutter_sensor_frame(
        model, cloud, start, end, max_error=True
    )
    rotations, translations = projection.interpolate_rigid(
        T(start), T(end), out.relative_time
    )
    forward = torch.einsum("nij,nj->ni", rotations, out.points_sensor) + translations
    torch.testing.assert_close(forward, cloud[out.valid_indices], atol=1e-9, rtol=0)
    assert out.valid_indices.dtype == torch.int64 and (out.valid_indices < 3).all()


def test_T11_empty_and_out_of_view():
    model = toy()
    for cloud in (torch.empty(0, 3), T([[0, 0, 50.0]], torch.float32)):
        out = projection.world_points_to_shutter_sensor_frame(
            model, cloud, pose(), pose(0, (1, 0, 0))
        )
        assert out.points_sensor.shape == (0, 3) and out.relative_time.shape == (0,)
        assert (
            out.valid_indices.shape == (0,) and out.valid_indices.dtype == torch.int64
        )
        assert out.points_sensor.dtype == cloud.dtype


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_iterations=0),
        dict(max_iterations=-1),
        dict(max_iterations=2.5),
        dict(stop_mean_relative_time_error=0),
        dict(stop_mean_relative_time_error=-1e-4),
        dict(stop_mean_relative_time_error=float("nan")),
        dict(stop_delta_mean_relative_time_error=0),
        dict(stop_delta_mean_relative_time_error=float("nan")),
    ],
)
def test_T12_invalid_accuracy_arguments(kwargs):
    with pytest.raises(ValueError):
        projection.world_points_to_shutter_sensor_frame(
            toy(), world_cloud(toy()), pose(), pose(), **kwargs
        )


def test_T12_budget_is_respected_and_nonconvergence_reported():
    model = synthetic_128(columns=64)
    cloud = world_cloud(model, 15.0)
    with pytest.warns(RuntimeWarning):
        out = projection.world_points_to_shutter_sensor_frame(
            model,
            cloud,
            pose(),
            pose(0, (30, 0, 0)),
            max_iterations=1,
            stop_mean_relative_time_error=1e-12,
            stop_delta_mean_relative_time_error=1e-15,
        )
    assert len(out.valid_indices) > 0


def test_T13_reuse_quaternions_does_not_change_results():
    model = synthetic_128(columns=64)
    cloud = world_cloud(model, 15.0)
    for dtype in (torch.float32, torch.float64):
        c = cloud.to(dtype)
        a = projection.world_points_to_shutter_sensor_frame(
            model, c, pose(), pose(0.2, (2, 0, 0))
        )
        b = projection.world_points_to_shutter_sensor_frame(
            model, c, pose(), pose(0.2, (2, 0, 0)), reuse_quaternions=True
        )
        assert torch.equal(a.valid_indices, b.valid_indices)
        torch.testing.assert_close(a.points_sensor, b.points_sensor)


def test_T14_max_error_mode_is_independent_of_distractors():
    model = synthetic_128(columns=64)
    near = world_cloud(model, 15.0)[::7]
    far = T(np.random.default_rng(17).normal(size=(500, 3)) * 1e6)
    start, end = pose(), pose(0.1, (2, 0, 0))
    alone = projection.world_points_to_shutter_sensor_frame(
        model, near, start, end, max_error=True
    )
    mixed = projection.world_points_to_shutter_sensor_frame(
        model, torch.cat((near, far)), start, end, max_error=True
    )
    keep = mixed.valid_indices < len(near)
    assert torch.equal(mixed.valid_indices[keep], alone.valid_indices)
    torch.testing.assert_close(mixed.points_sensor[keep], alone.points_sensor)
    torch.testing.assert_close(mixed.relative_time[keep], alone.relative_time)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        projection.world_points_to_shutter_sensor_frame(model, near, start, end)


def test_T15_no_stale_state_between_calls():
    model = synthetic_128(columns=64)
    cloud = world_cloud(model, 15.0)
    first = projection.world_points_to_shutter_sensor_frame(
        model, cloud, pose(), pose(0, (2, 0, 0))
    )
    other = projection.world_points_to_shutter_sensor_frame(
        model, cloud, pose(0.3), pose(0.5, (0, 3, 0))
    )
    again = projection.world_points_to_shutter_sensor_frame(
        model, cloud, pose(), pose(0, (2, 0, 0))
    )
    assert torch.equal(first.valid_indices, again.valid_indices)
    assert torch.equal(first.points_sensor, again.points_sensor)
    assert not torch.equal(
        first.points_sensor, other.points_sensor[: len(first.points_sensor)]
    ) or len(other.points_sensor) != len(first.points_sensor)


@pytest.mark.parametrize("max_error", [False, True])
def test_T09_T10_rear_seam_gap_returns_only_self_consistent_points(max_error):
    model = synthetic_128(columns=360)
    behind = np.linspace(np.pi - 0.2, np.pi + 0.2, 801)  # straddles the rear seam
    cloud = T([direction(0.0, a, 12) for a in behind])
    out = projection.world_points_to_shutter_sensor_frame(
        model, cloud, pose(), pose(0.1), max_error=max_error
    )
    implied, valid = model.relative_frame_times(out.points_sensor)
    assert valid.all()
    assert ((implied - out.relative_time).abs() <= 0.5 / 359).all()
    # yawing left by 0.1 rad opens a scan gap behind the sensor: some points drop
    assert 0 < len(out.valid_indices) < len(cloud)
    rotations, translations = projection.interpolate_rigid(
        T(pose()), T(pose(0.1)), out.relative_time
    )
    forward = torch.einsum("nij,nj->ni", rotations, out.points_sensor) + translations
    torch.testing.assert_close(forward, cloud[out.valid_indices], atol=1e-9, rtol=0)
