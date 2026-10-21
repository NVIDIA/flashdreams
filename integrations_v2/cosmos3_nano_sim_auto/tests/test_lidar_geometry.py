# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytic G/B cases for semantic LiDAR geometry (IDs follow the specification)."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    _lidar_geometry as geometry,
)

pytestmark = pytest.mark.ci_cpu

DTYPES = [np.float32, np.float64]
SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
TRIANGLE = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
L_SHAPE = np.array(
    [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], float
)
X90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # 90 degrees about +x


def tilt(points):
    return points @ X90.T + [3, -2, 5]


def untilt(points):
    return (points - [3, -2, 5]) @ X90


def max_gap_1d(values, lo, hi):
    values = np.sort(np.concatenate(([lo], values, [hi])))
    return np.diff(values).max()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("moved", [False, True])
def test_G01_line_support_and_endpoints(dtype, moved):
    start, end = np.zeros(3, dtype), np.array([1, 0, 0], dtype)
    if moved:
        start, end = (
            tilt(start.astype(float)).astype(dtype),
            tilt(end.astype(float)).astype(dtype),
        )
    points = geometry.sample_points_along_line(start, end, 0.1)
    assert points.shape[1] == 3 and np.isfinite(points).all()
    canonical = untilt(points.astype(float)) if moved else points.astype(float)
    tol = 1e-4 if dtype == np.float32 else 1e-9
    assert np.abs(canonical[:, 1:]).max() < tol
    assert canonical[:, 0].min() < tol and canonical[:, 0].max() > 1 - tol


def test_G02_short_and_degenerate_segments():
    for end in ([0, 0, 0], [0.009, 0, 0]):
        out = geometry.sample_points_along_line(
            np.zeros(3), np.array(end, float), 0.002
        )
        assert out.shape == (0, 3)
    out = geometry.sample_points_along_line(np.zeros(3), np.array([0.01, 0, 0]), 0.002)
    assert len(out) >= 2 and np.isfinite(out).all()


def test_G03_density_refinement():
    gaps = []
    for density in (0.2, 0.1, 0.05):
        x = geometry.sample_points_along_line(
            np.zeros(3), np.array([1.0, 0, 0]), density
        )[:, 0]
        gaps.append(max_gap_1d(x, 0, 1))
        assert gaps[-1] <= 2 * density
    assert gaps == sorted(gaps, reverse=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "start,end",
    [
        ([0, 0, 0], [0, 0, 2]),
        ([0, 0, 2], [0, 0, 0]),
        ([0, 0, 0], [2, 0, 0]),
        ([0, 0, 0], [0, 2, 0]),
        ([0, 0, 0], [1, 1, 1]),
    ],
)
def test_G04_cylinder_envelope_and_ends(dtype, start, end):
    a, b = np.array(start, float), np.array(end, float)
    points = geometry.sample_points_in_cylinder(
        a.astype(dtype), b.astype(dtype), 0.05, 0.2
    )
    p = points.astype(float)
    axis = (b - a) / np.linalg.norm(b - a)
    t = (p - a) @ axis
    radial = np.linalg.norm((p - a) - t[:, None] * axis, axis=1)
    tol = 1e-4 if dtype == np.float32 else 1e-9
    length = np.linalg.norm(b - a)
    assert t.min() > -tol and t.max() < length + tol and radial.max() <= 0.2 + tol
    assert t.min() < 0.05 and t.max() > length - 0.05
    assert radial.max() > 0.1


def test_G05_radius_zero_and_positive():
    line = geometry.sample_points_in_cylinder(
        np.zeros(3), np.array([0, 0, 1.0]), 0.1, 0.0
    )
    assert np.abs(line[:, :2]).max() == 0
    solid = geometry.sample_points_in_cylinder(
        np.zeros(3), np.array([0, 0, 1.0]), 0.1, 0.3
    )
    assert np.abs(solid[:, :2]).max() > 0.2
    assert np.hypot(solid[:, 0], solid[:, 1]).max() <= 0.3 + 1e-12


def test_G04_default_density_covers_disk_within_bounded_capacity():
    budget = geometry.PointBudget(64)
    points = geometry.sample_points_in_cylinder(
        np.zeros(3), np.array([0, 0, 0.04]), budget=budget
    )
    assert len(points) <= 64 and budget.remaining == 64 - len(points)
    section = points[np.isclose(points[:, 2], 0), :2]
    angle = np.linspace(0, 2 * np.pi, 33)
    probes = np.concatenate(
        [
            radius * np.stack((np.cos(angle), np.sin(angle)), axis=1)
            for radius in np.linspace(0, 0.05, 7)
        ]
    )
    distance = np.linalg.norm(probes[:, None] - section[None], axis=-1)
    assert distance.min(axis=1).max() < 2 * geometry.DEFAULT_DENSITY_M
    assert np.linalg.norm(section, axis=1).max() <= 0.05 + 1e-12


def test_G05_subpitch_disk_keeps_off_axis_samples():
    points = geometry.sample_points_in_cylinder(
        np.zeros(3), np.array([0, 0, 1]), density=0.1, radius=0.01
    )
    radius = np.linalg.norm(points[:, :2], axis=1)
    assert radius.max() == pytest.approx(0.01)
    assert radius.min() == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_G06_G18_empty_and_mixed_inputs(dtype):
    budget = geometry.PointBudget(5)
    out = geometry.points_from_polylines([], dtype=dtype, budget=budget)
    assert out.shape == (0, 3) and out.dtype == dtype and budget.remaining == 5
    mixed = [np.empty((0, 3)), np.array([[0, 0, 0], [1, 0, 0.0]]), np.empty((1, 3))]
    alone = geometry.points_from_polylines(mixed[1:2], 0.1, 0.0, dtype)
    both = geometry.points_from_polylines(mixed, 0.1, 0.0, dtype)
    assert both.dtype == dtype and np.array_equal(both, alone)


def test_G07_disconnected_polylines_are_not_joined():
    a = np.array([[0, 0, 0], [1, 0, 0.0]])
    b = np.array([[0, 5, 0], [1, 5, 0.0]])
    points = geometry.points_from_polylines([a, b], 0.1, 0.0)
    assert set(np.round(points[:, 1], 9)) == {0.0, 5.0}


@pytest.mark.parametrize("repeat_first", [False, True])
def test_G08_closed_path_closing_edge_once(repeat_first):
    path = SQUARE if not repeat_first else np.vstack((SQUARE, SQUARE[:1]))
    points = geometry.points_from_closed_polylines([path], 0.1, 0.0)
    assert np.isfinite(points).all()
    on_boundary = (
        np.isclose(points[:, 0], 0)
        | np.isclose(points[:, 0], 1)
        | np.isclose(points[:, 1], 0)
        | np.isclose(points[:, 1], 1)
    )
    assert on_boundary.all()  # no diagonal
    reference = geometry.points_from_closed_polylines([SQUARE], 0.1, 0.0)
    assert len(points) == len(reference)


@pytest.mark.parametrize("shape", ["square", "triangle"])
def test_G09_convex_polygon_fill(shape):
    hull = SQUARE if shape == "square" else TRIANGLE
    density = 0.05
    points = geometry.points_from_polygons([hull], density)
    assert np.abs(points[:, 2]).max() == 0
    assert points[:, :2].min() >= -1e-9 and points[:, :2].max() <= 1 + 1e-9
    if shape == "triangle":
        assert (points[:, 0] + points[:, 1]).max() <= 1 + 1e-9
    probes = (
        [(0.3, 0.3), (0.2, 0.1), (0.1, 0.2)]
        if shape == "triangle"
        else [(0.5, 0.5), (0.2, 0.8)]
    )
    for probe in probes:
        assert np.linalg.norm(points[:, :2] - probe, axis=1).min() <= 2 * density


def test_G10_concave_polygon_has_no_fill_in_missing_quadrant():
    points = geometry.points_from_polygons([L_SHAPE], 0.05)
    outside = (points[:, 0] > 1 + 1e-9) & (points[:, 1] > 1 + 1e-9)
    assert not outside.any()
    assert (points[:, 0] > 1.5).any() and (points[:, 1] > 1.5).any()
    assert np.linalg.norm(points[:, :2] - (0.5, 0.5), axis=1).min() <= 0.1


def test_G11_winding_closure_and_repeats_cover_same_region():
    base = geometry.points_from_polygons([L_SHAPE], 0.1)
    variants = [
        L_SHAPE[::-1],
        np.vstack((L_SHAPE, L_SHAPE[:1])),
        np.insert(L_SHAPE, 2, L_SHAPE[1], axis=0),
    ]
    for variant in variants:
        again = geometry.points_from_polygons([variant], 0.1)
        assert np.array_equal(again, geometry.points_from_polygons([variant], 0.1))
        order = np.lexsort(again.T)
        assert np.allclose(
            np.unique(np.round(again, 9), axis=0), np.unique(np.round(base, 9), axis=0)
        )
        assert order.size == len(again)


def test_G12_rotated_polygon_stays_in_transformed_plane():
    points = geometry.points_from_polygons([tilt(L_SHAPE)], 0.1)
    local = untilt(points)
    assert np.abs(local[:, 2]).max() < 1e-9
    assert not ((local[:, 0] > 1 + 1e-9) & (local[:, 1] > 1 + 1e-9)).any()


@pytest.mark.parametrize(
    "hull", [[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0]], [[0, 0, 0]]]
)
def test_G13_degenerate_polygon_is_boundary_only(hull):
    points = geometry.points_from_polygons([np.array(hull, float)], 0.1)
    assert np.isfinite(points).all() and np.abs(points[:, 1:]).max(initial=0) == 0


def test_G14_self_intersecting_polygon_rejected():
    hull = [[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]]
    with pytest.raises(ValueError):
        geometry.points_from_polygons([np.array(hull, float)])


@pytest.mark.parametrize(
    "reverse,shift", [(False, 0), (True, 0), (False, 2), (True, 2)]
)
def test_G14_nonplanar_quad_preserves_piecewise_terrain_height(reverse, shift):
    hull = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.3], [0, 1, 0]], float)
    hull = np.roll(hull[::-1] if reverse else hull, shift, axis=0)
    points = geometry.points_from_polygons([hull], 0.05)
    assert points[:, :2].min() >= -1e-9 and points[:, :2].max() <= 1 + 1e-9
    height = 0.3 * np.maximum(points[:, 0] + points[:, 1] - 1, 0)
    np.testing.assert_allclose(points[:, 2], height, atol=1e-9)
    for vertex in hull:
        assert np.linalg.norm(points - vertex, axis=1).min() < 1e-9


def test_G14_nonplanar_concave_polygon_does_not_fill_missing_quadrant():
    hull = L_SHAPE.copy()
    hull[1, 2], hull[-1, 2] = 0.4, 0.3
    points = geometry.points_from_polygons([hull], 0.05)
    assert np.isfinite(points).all()
    assert not ((points[:, 0] > 1 + 1e-9) & (points[:, 1] > 1 + 1e-9)).any()
    assert points[:, 2].min() >= -1e-9 and points[:, 2].max() <= 0.4 + 1e-9
    for probe in [[0.5, 0.5], [1.8, 0.5], [0.5, 1.8]]:
        assert np.linalg.norm(points[:, :2] - probe, axis=1).min() < 0.1


def test_B03_nonplanar_triangles_share_original_primitive_limit(monkeypatch):
    hull = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.3], [0, 1, 0]], float)
    count = len(geometry.points_from_polygons([hull], 0.1))
    budget = geometry.PointBudget(count * 2)
    monkeypatch.setattr(geometry, "MAX_PRIMITIVE_POINTS", count - 1)
    with pytest.raises(ValueError):
        geometry.points_from_polygons([hull], 0.1, budget=budget)
    assert budget.remaining == count * 2


@pytest.mark.parametrize("moved", [False, True])
def test_G15_cuboid_edges(moved):
    pose = np.eye(4)
    if moved:
        pose[:3, :3], pose[:3, 3] = X90, [3, -2, 5]
    path = geometry.cuboid_polyline(np.array([4, 2, 2.0]), pose)
    local = untilt(path) if moved else path
    corners = {
        tuple(c)
        for c in np.array(np.meshgrid([-2, 2], [-1, 1], [-1, 1])).reshape(3, -1).T
    }
    assert {tuple(np.round(p, 9)) for p in local} <= corners
    edges = set()
    for a, b in zip(local[:-1], local[1:]):
        assert (np.abs(a - b) > 1e-9).sum() == 1
        edges.add(frozenset((tuple(np.round(a, 9)), tuple(np.round(b, 9)))))
    assert len(edges) == 12


def test_G16_layer_dispatch():
    line = [np.array([[0, 0, 0], [1, 0, 0.0]])]
    box = [geometry.cuboid_polyline(np.ones(3), np.eye(4))]
    for name in ("lanelines", "road_boundaries", "wait_lines", "poles"):
        assert np.array_equal(
            geometry.minimap_points(name, line, 0.1, 0.0),
            geometry.points_from_polylines(line, 0.1, 0.0),
        )
    for name in ("crosswalks", "road_markings"):
        assert np.array_equal(
            geometry.minimap_points(name, [SQUARE], 0.1),
            geometry.points_from_polygons([SQUARE], 0.1),
        )
    for name in ("traffic_signs", "traffic_lights"):
        assert np.array_equal(
            geometry.minimap_points(name, box, 0.1, 0.0),
            geometry.points_from_closed_polylines(box, 0.1, 0.0),
        )
    assert set(geometry.DEFAULT_MINIMAP_TYPES) == set(geometry.LAYER_TYPES)
    with pytest.raises(ValueError):
        geometry.minimap_points("trees", line)


@pytest.mark.parametrize(
    "call",
    [
        lambda: geometry.sample_points_along_line(np.array([np.nan, 0, 0]), np.ones(3)),
        lambda: geometry.sample_points_along_line(np.zeros(2), np.ones(3)),
        lambda: geometry.sample_points_along_line(np.zeros(3), np.ones(3), 0),
        lambda: geometry.sample_points_along_line(np.zeros(3), np.ones(3), -1),
        lambda: geometry.sample_points_in_cylinder(np.zeros(3), np.ones(3), 0.1, -0.1),
        lambda: geometry.points_from_polylines([np.ones((3, 2))]),
        lambda: geometry.points_from_polylines([np.array([[np.inf, 0, 0], [1, 0, 0]])]),
        lambda: geometry.points_from_polylines([SQUARE], dtype=np.int32),
        lambda: geometry.cuboid_polyline(np.array([1, 0, 1.0]), np.eye(4)),
        lambda: geometry.cuboid_polyline(np.ones(3), 2 * np.eye(4)),
        lambda: geometry.cuboid_polyline(np.ones(3), np.diag([1, 1, -1, 1.0])),
    ],
)
def test_G17_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_G17_inputs_not_mutated():
    polygon = L_SHAPE.copy()
    geometry.points_from_polygons([polygon], 0.1)
    assert np.array_equal(polygon, L_SHAPE)
    pose = np.eye(4)
    geometry.cuboid_polyline(np.ones(3), pose)
    assert np.array_equal(pose, np.eye(4))


def test_G19_determinism_and_reordering():
    prims = [SQUARE, tilt(TRIANGLE)]
    a = geometry.points_from_polygons(prims, 0.1)
    assert np.array_equal(a, geometry.points_from_polygons(prims, 0.1))
    b = geometry.points_from_polygons(prims[::-1], 0.1)
    assert np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)])


def test_B01_B02_budget_semantics():
    budget = geometry.PointBudget(10)
    remaining = []
    for count in (0, 4, 6):
        budget.reserve(count)
        remaining.append(budget.remaining)
    assert remaining == [10, 6, 0]
    budget = geometry.PointBudget(3)
    for bad in (4, -1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            budget.reserve(bad)  # ty: ignore[invalid-argument-type]  # Exercise invalid-input validation.
        assert budget.remaining == 3


def test_B03_B04_shared_budget_exact_fit_and_empty_free(monkeypatch):
    seg = [np.array([[0, 0, 0], [1, 0, 0.0]])]
    n = len(geometry.points_from_polylines(seg, 0.25, 0.0))
    budget = geometry.PointBudget(2 * n)
    geometry.points_from_polylines(seg, 0.25, 0.0, budget=budget)
    geometry.minimap_points("lanelines", seg, 0.25, 0.0, budget=budget)
    assert budget.remaining == 0
    assert len(geometry.points_from_polylines([np.empty((0, 3))], budget=budget)) == 0
    assert len(geometry.points_from_polygons([np.zeros((3, 3))], budget=budget)) == 0
    with pytest.raises(ValueError):
        geometry.points_from_polylines(seg, 0.25, 0.0, budget=budget)
    assert budget.remaining == 0


def test_B03_failed_multi_primitive_call_rolls_back():
    seg = np.array([[0, 0, 0], [1, 0, 0.0]])
    n = len(geometry.points_from_polylines([seg], 0.25, 0.0))
    budget = geometry.PointBudget(n + 1)
    with pytest.raises(ValueError):
        geometry.points_from_polylines([seg, seg], 0.25, 0.0, budget=budget)
    assert budget.remaining == n + 1


def test_B05_over_limit_refused_before_allocation():
    with pytest.raises(ValueError):
        geometry.sample_points_along_line(np.zeros(3), np.array([1e9, 0, 0]), 1e-3)
    with pytest.raises(ValueError):
        geometry.sample_points_along_line(np.zeros(3), np.ones(3), 1e-300)
    huge = np.array([[0, 0, 0], [1e6, 0, 0], [1e6, 1e6, 0], [0, 1e6, 0]], float)
    with pytest.raises(ValueError):
        geometry.points_from_polygons([huge], 0.02)
    with pytest.raises(ValueError):
        geometry.sample_points_in_cylinder(np.zeros(3), np.ones(3), 1e-3, 1e6)


def test_B05_memory_capped_process_rejects_cleanly():
    code = textwrap.dedent(
        """
        import resource, sys, numpy as np
        sys.path[:0] = sys.argv[1:]
        resource.setrlimit(resource.RLIMIT_AS, (3 * 2**30, 3 * 2**30))
        from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import _lidar_geometry as g
        for call in (
            lambda: g.sample_points_along_line(np.zeros(3), np.array([1e7, 0, 0])),
            lambda: g.points_from_polygons([np.array([[0, 0, 0], [1e5, 0, 0], [1e5, 1e5, 0], [0, 1e5, 0.]])]),
        ):
            try:
                call()
            except ValueError:
                continue
            raise SystemExit("not rejected")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code, *sys.path], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_B06_ten_thousand_vertex_boundary_is_bounded():
    angle = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
    ring = 100 * np.stack((np.cos(angle), np.sin(angle), 0 * angle), axis=1)
    points = geometry.points_from_closed_polylines([ring], 0.05, 0.01, np.float32)
    assert (
        np.isfinite(points).all() and np.hypot(points[:, 0], points[:, 1]).max() < 101
    )


def test_B07_narrow_concave_polygon_bounded_work(monkeypatch):
    sliver = np.array([[0, 0, 0], [1000, 0, 0], [1000, 0.01, 0], [0, 0.01, 0]], float)
    points = geometry.points_from_polygons([sliver], 0.5)
    assert np.isfinite(points).all() and points[:, 1].max() <= 0.01 + 1e-9
    monkeypatch.setattr(geometry, "MAX_PRIMITIVE_POINTS", 100)
    with pytest.raises(ValueError):
        geometry.points_from_polygons([sliver], 0.5)
