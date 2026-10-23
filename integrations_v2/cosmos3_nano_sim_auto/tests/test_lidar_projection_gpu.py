# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA cases I18/I19 for the independent LiDAR geometry/projection."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    _lidar_geometry as geometry,
)
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    _lidar_projection as projection,
)
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    prepared_lidar,
)

pytestmark = [
    pytest.mark.ci_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def profiles():
    toy = projection.SpinningLidarModel(np.array([0.25, 0, -0.25]), np.zeros(3), 8)
    rings = np.linspace(0.25, -0.45, 128)
    big = projection.SpinningLidarModel(rings, np.full(128, 0.01), 1800)
    return {"F05": toy, "synthetic128": big}


def yaw_pose(yaw, x):
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    out[0, 3] = x
    return out


def cloud(dtype, n=200_000):
    gen = torch.Generator().manual_seed(17)
    direction = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    direction[:, 2] *= 0.2
    return (
        direction * (torch.rand(n, 1, generator=gen, dtype=torch.float64) * 120)
    ).to(dtype)


@pytest.mark.parametrize("name", ["F05", "synthetic128"])
def test_I18_cpu_and_cuda_agree(name):
    model = profiles()[name]
    for sorted_lookup in (False, True):
        sensor = projection.SpinningLidarModel(
            model.row_elevations_rad,
            model.row_azimuth_offsets_rad,
            model.n_columns,
            sorted_ring_lookup=sorted_lookup,
        )
        points = cloud(torch.float64)
        gpu = points.cuda()
        elevation = sensor.sensor_angles(points)[0]
        rows = sensor.nearest_rows(elevation)
        rows_gpu = sensor.nearest_rows(elevation.cuda())
        assert rows_gpu.is_cuda and torch.equal(rows_gpu.cpu(), rows)
        time, valid = sensor.relative_frame_times(points)
        time_gpu, valid_gpu = sensor.relative_frame_times(gpu)
        assert torch.equal(valid_gpu.cpu(), valid)
        torch.testing.assert_close(time_gpu.cpu(), time, atol=2e-6, rtol=0)
        intensity = torch.arange(len(points), dtype=torch.float64)
        cpu_map = projection.points_to_range_map(points, sensor, intensity=intensity)
        gpu_map = projection.points_to_range_map(
            gpu, sensor, intensity=intensity.cuda()
        )
        assert all(t.is_cuda and t.dtype == torch.float32 for t in gpu_map)
        assert torch.equal(gpu_map[0].cpu() > 0, cpu_map[0] > 0)
        torch.testing.assert_close(gpu_map[0].cpu(), cpu_map[0], atol=1e-4, rtol=1e-5)
        assert torch.equal(gpu_map[1].cpu(), cpu_map[1])  # same winning point per cell


@pytest.mark.filterwarnings(
    "ignore:Shutter projection:RuntimeWarning"
)  # sub-metre points
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_I18_moving_shutter_cpu_and_cuda_agree(dtype):
    model = profiles()["synthetic128"]
    points = cloud(dtype, 50_000)
    first, last = yaw_pose(0.0, 0.0), yaw_pose(0.05, 1.5)
    for max_error in (False, True):
        cpu = projection.world_points_to_shutter_sensor_frame(
            model, points, first, last, max_error=max_error
        )
        gpu = projection.world_points_to_shutter_sensor_frame(
            model, points.cuda(), first, last, max_error=max_error
        )
        assert gpu.points_sensor.is_cuda and gpu.valid_indices.is_cuda
        # exact away from boundaries: allow a handful of points within float error
        # of the seam/field-of-view/self-consistency limits
        shared = np.intersect1d(
            cpu.valid_indices.numpy(), gpu.valid_indices.cpu().numpy()
        )
        differ = len(cpu.valid_indices) + len(gpu.valid_indices) - 2 * len(shared)
        assert differ <= (0 if dtype == torch.float64 else 10)
        position = {i: k for k, i in enumerate(gpu.valid_indices.cpu().tolist())}
        pick = torch.tensor(
            [position[i] for i in cpu.valid_indices.tolist() if i in position]
        )
        keep = torch.tensor([i in position for i in cpu.valid_indices.tolist()])
        torch.testing.assert_close(
            gpu.points_sensor.cpu()[pick], cpu.points_sensor[keep], atol=1e-4, rtol=1e-5
        )


def renderer(device, monkeypatch, rings):
    model = projection.SpinningLidarModel(rings, np.full(128, 0.01), 1800, 10)
    walls = [
        [[20, -4, -4], [20, 4, -4], [20, 4, 4], [20, -4, 4]],
        [[-12, -6, -3], [-12, 6, -3], [-12, 6, 3], [-12, -6, 3]],
        [[2, -0.1, -0.1], [2, 0.1, -0.1], [2, 0.1, 0.1], [2, -0.1, 0.1]],
    ]
    layers = {name: [] for name in geometry.DEFAULT_MINIMAP_TYPES}
    layers["road_markings"] = [np.array(w, float) for w in walls]
    layers["poles"] = [np.array([[8, 3, -1], [8, 3, 1]], float)]
    car = (
        np.array([0, 2]),
        np.array([[4, 2, 1.5]] * 2),
        np.stack([yaw_pose(0, 30), yaw_pose(0, -9)]),
    )
    lidar = prepared_lidar.PreparedLidar(
        model, np.eye(4), np.eye(4), "front", layers, [car]
    )
    scene = SimpleNamespace(lidar=lidar, frame_mapping=lambda *_: np.arange(9) // 3)
    return prepared_lidar.Renderer(scene, device=device)


@pytest.mark.parametrize("options", [{}, {"optimized": True}, {"cull": True}])
def test_I18_renderer_on_cuda_matches_cpu(monkeypatch, options):
    rings = np.linspace(0.25, -0.45, 128)
    cpu, gpu = renderer("cpu", monkeypatch, rings), renderer("cuda", monkeypatch, rings)
    for target in (cpu, gpu):
        for key, value in options.items():
            setattr(target, key, value)
    for index, (first, last) in enumerate(
        [
            (np.eye(4), np.eye(4)),
            (np.eye(4), yaw_pose(0.05, 1.5)),
            (yaw_pose(0.05, -1.5), yaw_pose(0.1, -3)),
        ]
    ):
        expected, actual = cpu(index, first, last), gpu(index, first, last)
        assert actual.is_cuda and actual.shape == (3, 128, 1800)
        mismatch = int((actual[2].cpu() != expected[2]).sum())
        assert mismatch <= 2, mismatch  # float32 points: only boundary cells may flip
        same = actual[2].cpu() == expected[2]
        torch.testing.assert_close(
            actual[0].cpu()[same], expected[0][same], atol=1e-3, rtol=1e-5
        )


def test_I19_no_memory_growth_across_sweeps_and_profiles(monkeypatch):
    a = renderer("cuda", monkeypatch, np.linspace(0.25, -0.45, 128))
    b = renderer("cuda", monkeypatch, np.linspace(0.2, -0.4, 128))
    # reverse away from the 2 m occluder so no geometry passes through the sensor
    poses = [
        (yaw_pose(0.05 * k, -1.5 * k), yaw_pose(0.05 * (k + 1), -1.5 * (k + 1)))
        for k in range(6)
    ]
    reference = [a(k, *p).cpu() for k, p in enumerate(poses)]  # warmup + reference
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for repeat in range(3):
        for k, p in enumerate(poses):
            assert torch.equal(a(k, *p).cpu(), reference[k])  # no cross-profile state
            b(k, *p)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == baseline, repeat
    peak = torch.cuda.max_memory_allocated()
    print(
        f"I19 allocated {baseline / 2**20:.1f} MiB, sweep peak {peak / 2**20:.1f} MiB, "
        f"reserved {torch.cuda.memory_reserved() / 2**20:.1f} MiB"
    )


def test_capacity_fallback_renders_on_cuda(monkeypatch):
    monkeypatch.setattr(geometry, "MAX_SCENE_POINTS", 100_000)
    rings = np.linspace(0.25, -0.45, 128)
    cpu = renderer("cpu", monkeypatch, rings)
    gpu = renderer("cuda", monkeypatch, rings)
    assert cpu.density == gpu.density > geometry.DEFAULT_DENSITY_M
    assert len(gpu.static) + gpu.data.actor_point_capacity(gpu.density) <= 100_000
    for index in (0, 2, 50):
        expected = cpu(index, np.eye(4), np.eye(4))
        actual = gpu(index, np.eye(4), np.eye(4))
        assert actual.is_cuda and torch.isfinite(actual).all()
        assert torch.equal(actual[2].cpu(), expected[2])
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=1e-3, rtol=1e-5)
