# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic LiDAR geometry sampling written independently from the handoff specification.

Policies (see ``tests/specs/lidar_geometry_projection.md``):

* Everything is computed in float64 and cast to the requested dtype once.
* Segments shorter than 1 cm contribute no samples; empty results are ``[0, 3]``.
* Budgets count returned points (duplicates included). A call reserves its total
  once, after every primitive is validated and sized, so a failed multi-primitive
  call never charges the budget (all-or-nothing). ``MAX_PRIMITIVE_POINTS`` and
  ``MAX_SCENE_POINTS`` are read at call time. Over-limit requests are refused from
  closed-form counts before any output-sized array exists.
* Cylinder cross-sections use a triangular lattice of pitch ``density`` clipped
  to the disk. Sub-pitch disks retain their centre and four boundary samples.
* Planar polygons use even-odd scanlines plus boundary samples. Non-planar
  terrain polygons use ear clipping in their best-fit plane and fill each
  resulting triangle in 3D. Canonical winding and starting vertex make that
  surface independent of boundary winding or cyclic vertex order. Self-intersections reject.
  Fewer than three distinct vertices, or collinear vertices, yield boundary
  samples only. Workspace is bounded by the point limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MINIMAP_TYPES = (
    "lanelines",
    "road_boundaries",
    "crosswalks",
    "road_markings",
    "wait_lines",
    "poles",
    "traffic_signs",
    "traffic_lights",
)
"""Semantic layer names accepted by the prepared-scene metadata schema."""

LAYER_TYPES = {
    "lanelines": "polyline",
    "road_boundaries": "polyline",
    "crosswalks": "polygon",
    "road_markings": "polygon",
    "wait_lines": "polyline",
    "poles": "polyline",
    "traffic_signs": "cuboid3d",
    "traffic_lights": "cuboid3d",
}
"""Primitive categories used to validate metadata and choose a sampler."""

DEFAULT_DENSITY_M = 0.02
DEFAULT_RADIUS_M = 0.05
MAX_PRIMITIVE_POINTS = 10_000_000
MAX_SCENE_POINTS = 60_000_000

IMPLEMENTATION_AVAILABLE = True
"""Renderer gate; set False to disable prepared-scene LiDAR rendering again."""

UNAVAILABLE_REASON = (
    "Prepared-scene LiDAR rendering is unavailable in this build. Use RGB-only mode."
)

_MIN_SEGMENT_M = 0.01
_PLANE_TOLERANCE = 1e-2  # fraction of the polygon's largest extent
_POSE_TOLERANCE = 1e-3
_CHUNK_ELEMENTS = 500_000


def require_implementation() -> None:
    """Reject rendering while ``IMPLEMENTATION_AVAILABLE`` is False."""
    if not IMPLEMENTATION_AVAILABLE:
        raise NotImplementedError(UNAVAILABLE_REASON)


def _count(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


class PointLimitError(ValueError):
    """Sampling capacity exceeded; callers may retry at a coarser spacing."""


@dataclass
class PointBudget:
    """Shared point capacity; ``reserve`` is atomic and never increases capacity."""

    remaining: int
    """Points still available."""

    def __post_init__(self) -> None:
        self.remaining = _count(self.remaining, "remaining")

    def reserve(self, count: int) -> None:
        """Charge ``count`` points, or raise ``ValueError`` and change nothing."""
        count = _count(count, "count")
        if count > self.remaining:
            raise PointLimitError(
                f"LiDAR point budget exceeded: {count} requested, "
                f"{self.remaining} remaining"
            )
        self.remaining -= count


def _positive(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not (math.isfinite(result) and result > 0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _radius(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"radius must be a number, got {value!r}") from None
    if not (math.isfinite(result) and result >= 0):
        raise ValueError(f"radius must be finite and non-negative, got {value!r}")
    return result


def _dtype(dtype) -> np.dtype:
    try:
        result = np.dtype(dtype)
    except TypeError:
        raise ValueError(f"unsupported dtype {dtype!r}") from None
    if result not in (np.float32, np.float64):
        raise ValueError(f"dtype must be float32 or float64, got {result}")
    return result


def _vec3(value, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite 3-vector") from None
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector, got shape {result.shape}")
    return result


def _path(value, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite [K, 3] array") from None
    if result.size == 0:
        return np.empty((0, 3))
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [K, 3] array, got {result.shape}")
    return result


def _primitives(items, name: str) -> list[np.ndarray]:
    if isinstance(items, (str, bytes)) or not hasattr(items, "__iter__"):
        raise ValueError(f"{name} must be a sequence of [K, 3] arrays")
    return [_path(item, name) for item in items]


def _scene_budget(budget: PointBudget | None) -> PointBudget:
    if budget is None:
        return PointBudget(MAX_SCENE_POINTS)
    if not isinstance(budget, PointBudget):
        raise ValueError("budget must be a PointBudget or None")
    return budget


def _gather(items, make, budget: PointBudget | None, dtype) -> np.ndarray:
    """Sample each item under the remaining capacity, then charge the total once."""
    budget = _scene_budget(budget)
    parts, used = [], 0
    for item in items:
        points = make(item, min(MAX_PRIMITIVE_POINTS, budget.remaining - used))
        used += len(points)
        parts.append(points)
    budget.reserve(used)
    return (np.concatenate(parts) if parts else np.empty((0, 3))).astype(dtype)


def _too_many(cap: int) -> PointLimitError:
    return PointLimitError(f"LiDAR primitive needs more than the {cap} points allowed")


def _disk(radius: float, density: float, cap: int) -> np.ndarray:
    """Cross-section offsets on a density-spaced triangular lattice."""
    if radius == 0:
        return np.zeros((1, 2))
    # Work in density units to avoid squaring very large physical radii.
    r = radius / density
    pitch = math.sqrt(3) / 2
    rows = math.floor(r / pitch) if r < cap else cap + 1
    if 2 * rows + 1 > cap:
        raise _too_many(cap)
    row = np.arange(-rows, rows + 1)
    y = row * pitch
    shift = (row % 2) / 2
    width = np.sqrt(np.maximum((r - np.abs(y)) * (r + np.abs(y)), 0))
    first = np.ceil(-width - shift)
    count = np.maximum(np.floor(width - shift) - first + 1, 0).astype(np.int64)
    total = int(count.sum())
    if total == 1:
        if cap < 5:
            raise _too_many(cap)
        return radius * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    if total > cap:
        raise _too_many(cap)
    index = np.repeat(np.arange(len(row)), count)
    x = (
        np.arange(total)
        - (np.cumsum(count) - count)[index]
        + first[index]
        + shift[index]
    )
    return density * np.stack((x, y[index]), axis=1)


def _segment_points(
    a: np.ndarray, b: np.ndarray, density: float, radius: float, cap: int
) -> np.ndarray:
    """Sample the cylinders (``radius`` 0: lines) around segments ``a[i] -> b[i]``."""
    delta = b - a
    length = np.linalg.norm(delta, axis=1)
    keep = length >= _MIN_SEGMENT_M
    a, delta, length = a[keep], delta[keep], length[keep]
    if not len(length):
        return np.empty((0, 3))
    offsets = _disk(radius, density, cap)
    axial = np.ceil(length / density) + 1
    if axial.sum() * len(offsets) > cap:
        raise _too_many(cap)
    axial = axial.astype(np.int64)
    segment = np.repeat(np.arange(len(axial)), axial)
    first = np.cumsum(axial) - axial
    t = (np.arange(len(segment)) - first[segment]) / (axial[segment] - 1)
    centres = a[segment] + t[:, None] * delta[segment]
    if len(offsets) == 1:
        return centres
    axis = delta / length[:, None]
    helper = np.eye(3)[np.argmin(np.abs(axis), axis=1)]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(axis, u)
    points = (
        centres[:, None, :]
        + offsets[None, :, 0, None] * u[segment][:, None, :]
        + offsets[None, :, 1, None] * v[segment][:, None, :]
    )
    return points.reshape(-1, 3)


def _float_dtype(*values) -> np.dtype:
    result = np.result_type(*(np.asarray(value).dtype for value in values))
    return result if result in (np.float32, np.float64) else np.dtype(np.float64)


def sample_points_along_line(
    start: np.ndarray,
    end: np.ndarray,
    density: float = DEFAULT_DENSITY_M,
    *,
    budget: PointBudget | None = None,
) -> np.ndarray:
    """Evenly sample ``start -> end`` including both endpoints (gap <= ``density``)."""
    dtype = _float_dtype(start, end)
    a, b = _vec3(start, "start"), _vec3(end, "end")
    density = _positive(density, "density")
    return _gather(
        [None],
        lambda _, cap: _segment_points(a[None], b[None], density, 0.0, cap),
        budget,
        dtype,
    )


def sample_points_in_cylinder(
    start: np.ndarray,
    end: np.ndarray,
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    *,
    budget: PointBudget | None = None,
) -> np.ndarray:
    """Sample the solid cylinder of ``radius`` around ``start -> end``."""
    dtype = _float_dtype(start, end)
    a, b = _vec3(start, "start"), _vec3(end, "end")
    density, radius = _positive(density, "density"), _radius(radius)
    return _gather(
        [None],
        lambda _, cap: _segment_points(a[None], b[None], density, radius, cap),
        budget,
        dtype,
    )


def points_from_polylines(
    polylines: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    dtype: type = np.float64,
    *,
    budget: PointBudget | None = None,
) -> np.ndarray:
    """Sample each polyline's segments; separate polylines are never joined."""
    paths = _primitives(polylines, "polyline")
    density, radius, output_dtype = (
        _positive(density, "density"),
        _radius(radius),
        _dtype(dtype),
    )
    return _gather(
        paths,
        lambda path, cap: _segment_points(path[:-1], path[1:], density, radius, cap),
        budget,
        output_dtype,
    )


def _distinct(path: np.ndarray) -> np.ndarray:
    """Drop consecutive repeats and a repeated closing vertex."""
    if len(path) > 1:
        path = path[np.concatenate(([True], np.any(path[1:] != path[:-1], axis=1)))]
    if len(path) > 1 and np.array_equal(path[0], path[-1]):
        path = path[:-1]
    return path


def _closed_edges(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = _distinct(path)
    if len(path) < 2:
        return np.empty((0, 3)), np.empty((0, 3))
    if len(path) == 2:  # out-and-back is one geometric edge
        return path[:1], path[1:]
    return path, np.roll(path, -1, axis=0)


def points_from_closed_polylines(
    hulls: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    dtype: type = np.float64,
    *,
    budget: PointBudget | None = None,
) -> np.ndarray:
    """Sample closed edge paths; a repeated first vertex does not duplicate the closing edge."""
    paths = _primitives(hulls, "closed polyline")
    density, radius, output_dtype = (
        _positive(density, "density"),
        _radius(radius),
        _dtype(dtype),
    )
    return _gather(
        paths,
        lambda path, cap: _segment_points(*_closed_edges(path), density, radius, cap),
        budget,
        output_dtype,
    )


def _self_intersects(xy: np.ndarray) -> bool:
    """True if two non-adjacent edges properly cross.

    ponytail: O(K^2) time (chunked, O(1) memory); a sweep line would scale past ~1e5 edges.
    """
    n = len(xy)
    a, b = xy, np.roll(xy, -1, axis=0)
    scale = float(np.ptp(xy, axis=0).max())
    eps = 1e-12 * scale * scale
    index = np.arange(n)

    def side(p, q, r):
        value = (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]
        ) * (r[..., 0] - p[..., 0])
        return np.where(np.abs(value) <= eps, 0.0, np.sign(value))

    step = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, n, step):
        i = index[start : start + step, None]
        j = index[None, :]
        neighbours = (j == i) | (j == (i + 1) % n) | (j == (i - 1) % n)
        pa, pb, pc, pd = a[i], b[i], a[j], b[j]
        crossing = (
            (side(pa, pb, pc) * side(pa, pb, pd) < 0)
            & (side(pc, pd, pa) * side(pc, pd, pb) < 0)
            & ~neighbours
        )
        if crossing.any():
            return True
    return False


def _fill(xy: np.ndarray, density: float, cap: int) -> np.ndarray:
    """Even-odd scanline fill: lattice points ``(xmin + i h, ymin + j h)`` inside ``xy``."""
    xmin, ymin = xy.min(axis=0)
    nxt = np.roll(xy, -1, axis=0)
    flip = xy[:, 1] > nxt[:, 1]
    low, high = np.where(flip[:, None], nxt, xy), np.where(flip[:, None], xy, nxt)
    first_row = np.ceil((low[:, 1] - ymin) / density)
    last_row = np.ceil((high[:, 1] - ymin) / density) - 1  # half-open: rows in [ya, yb)
    rows = np.maximum(last_row - first_row + 1, 0)
    if rows.sum() > cap:
        raise _too_many(cap)
    rows = rows.astype(np.int64)
    edge = np.repeat(np.arange(len(rows)), rows)
    row = (
        np.arange(len(edge)) - (np.cumsum(rows) - rows)[edge] + first_row[edge]
    ).astype(np.int64)
    y = ymin + row * density
    x = low[edge, 0] + (y - low[edge, 1]) * (
        (high[edge, 0] - low[edge, 0]) / (high[edge, 1] - low[edge, 1])
    )
    order = np.lexsort((x, row))
    row, x = row[order], x[order]
    left, right, row = x[0::2], x[1::2], row[0::2]
    first_col = np.ceil((left - xmin) / density)
    count = np.maximum(np.floor((right - xmin) / density) - first_col + 1, 0)
    if count.sum() > cap:
        raise _too_many(cap)
    count = count.astype(np.int64)
    span = np.repeat(np.arange(len(count)), count)
    col = np.arange(len(span)) - (np.cumsum(count) - count)[span] + first_col[span]
    return np.stack((xmin + col * density, ymin + row[span] * density), axis=1)


def _triangles(vertices: np.ndarray, xy: np.ndarray) -> list[np.ndarray]:
    """Triangulate a simple boundary without filling its concavities."""
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    order = list(range(len(vertices)))
    if area < 0:
        order.reverse()
    anchor = int(np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))[0])
    start = order.index(anchor)
    order = order[start:] + order[:start]
    eps = 1e-12 * float(np.ptp(xy, axis=0).max()) ** 2

    def side(a, b, p):
        return (b[0] - a[0]) * (p[..., 1] - a[1]) - (b[1] - a[1]) * (p[..., 0] - a[0])

    triangles = []
    # ponytail: O(K^3) worst-case time, O(K) workspace; use a maintained ear
    # queue if large non-planar boundaries become a supported workload.
    while len(order) > 3:
        for i, current in enumerate(order):
            previous, following = order[i - 1], order[(i + 1) % len(order)]
            a, b, c = xy[[previous, current, following]]
            if side(a, b, c) <= eps:
                continue
            other = xy[[j for j in order if j not in (previous, current, following)]]
            if np.any(
                (side(a, b, other) >= -eps)
                & (side(b, c, other) >= -eps)
                & (side(c, a, other) >= -eps)
            ):
                continue
            triangles.append(vertices[[previous, current, following]])
            del order[i]
            break
        else:
            raise ValueError("polygon boundary cannot be triangulated")
    triangles.append(vertices[order])
    return triangles


def _polygon_points(hull: np.ndarray, density: float, cap: int) -> np.ndarray:
    vertices = _distinct(hull)
    if len(vertices) == 0:
        return np.empty((0, 3))
    outline = _segment_points(*_closed_edges(vertices), density, 0.0, cap)
    if len(vertices) < 3:
        return outline
    centre = vertices.mean(axis=0)
    offsets = vertices - centre
    _, sigma, vt = np.linalg.svd(offsets, full_matrices=False)
    if sigma[1] <= 1e-9 * sigma[0]:  # collinear: boundary only
        return outline
    normal = vt[2]
    extent = float(np.ptp(vertices, axis=0).max())
    nonplanar = np.abs(offsets @ normal).max() > _PLANE_TOLERANCE * extent + 1e-12
    if nonplanar and normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    e1 = np.eye(3)[np.argmin(np.abs(normal))]
    u = e1 - (e1 @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    xy = np.stack((offsets @ u, offsets @ v), axis=1)
    if _self_intersects(xy):
        raise ValueError("polygon boundary self-intersects")
    if nonplanar:
        parts, used = [], 0
        for triangle in _triangles(vertices, xy):
            points = _polygon_points(triangle, density, cap - used)
            used += len(points)
            parts.append(points)
        return np.concatenate(parts)
    inside = _fill(xy, density, cap - len(outline))
    interior = centre + inside[:, :1] * u + inside[:, 1:] * v
    return np.concatenate((outline, interior))


def points_from_polygons(
    hulls: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    dtype: type = np.float64,
    *,
    budget: PointBudget | None = None,
) -> np.ndarray:
    """Sample planar or triangulated terrain polygons, preserving concavities."""
    paths = _primitives(hulls, "polygon")
    density, output_dtype = _positive(density, "density"), _dtype(dtype)
    return _gather(
        paths,
        lambda path, cap: _polygon_points(path, density, cap),
        budget,
        output_dtype,
    )


def minimap_points(
    minimap_name: str,
    elements: Sequence[np.ndarray],
    density: float = DEFAULT_DENSITY_M,
    radius: float = DEFAULT_RADIUS_M,
    dtype: type = np.float64,
    *,
    budget: PointBudget | None = None,
) -> np.ndarray:
    """Sample one semantic layer: lines, polygon interiors or closed box edges."""
    kind = LAYER_TYPES.get(minimap_name)
    if kind is None:
        raise ValueError(
            f"Unsupported map layer {minimap_name!r}; expected one of "
            f"{DEFAULT_MINIMAP_TYPES}"
        )
    if kind == "polyline":
        return points_from_polylines(elements, density, radius, dtype, budget=budget)
    if kind == "polygon":
        return points_from_polygons(elements, density, dtype, budget=budget)
    return points_from_closed_polylines(elements, density, radius, dtype, budget=budget)


_CORNER = {
    name: np.array(sign, dtype=np.float64)
    for name, sign in {
        "b0": (-1, -1, -1),
        "b1": (1, -1, -1),
        "b2": (1, 1, -1),
        "b3": (-1, 1, -1),
        "t0": (-1, -1, 1),
        "t1": (1, -1, 1),
        "t2": (1, 1, 1),
        "t3": (-1, 1, 1),
    }.items()
}
# One walk over all twelve edges; the three pillars b1-t1, b2-t2, b3-t3 are retraced.
_CUBOID_WALK = np.stack(
    [
        _CORNER[name]
        for name in ("b0 b1 b2 b3 b0 t0 t1 b1 t1 t2 b2 t2 t3 b3 t3 t0".split())
    ]
)


def cuboid_polyline(lwh: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Edge path ``[16, 3]`` over a box of ``lwh`` (x, y, z extents) at ``pose`` (object to world)."""
    size = _vec3(lwh, "lwh")
    if (size <= 0).any():
        raise ValueError(f"lwh must be positive, got {size}")
    try:
        matrix = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("pose must be a finite rigid [4, 4] transform") from None
    rotation = matrix[:3, :3] if matrix.shape == (4, 4) else None
    if (
        rotation is None
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=_POSE_TOLERANCE)
        or not np.linalg.det(rotation) > 0
    ):
        raise ValueError("pose must be a finite rigid [4, 4] transform")
    return (_CUBOID_WALK * size / 2) @ rotation.T + matrix[:3, 3]
