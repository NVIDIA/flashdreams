# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spinning-LiDAR rays, firing time, rigid motion and range-map projection.

Written independently from the handoff specification.

Conventions: metres, radians, right-handed forward/left/up sensor frame, poses map
sensor to world. Azimuth is ``atan2(y, x)``. A scan starts at the rear (+pi) and
runs clockwise: column ``c`` has leading edge ``pi - c * 2pi/C`` and its cell
extends one column clockwise from it. A ring's firing offset is added to the
physical azimuth at which each column fires, so a point at leading edge + offset
fires at that column's time. Firing time is continuous in azimuth: ``u`` columns
into the scan gives ``u / (C - 1)``, clamped to ``[0, 1]``. Rings are matched by
minimum angular distance, exact ties choosing the lower row index.

Policies: equal-depth returns in one cell resolve to the lowest point index
(range and intensity from the same point); ``interpolate_rigid`` is a shortest-arc
slerp (a 180 degree relative rotation takes the axis with its largest component
positive); the ``quaternions`` and ``reuse_quaternions`` hints are accepted but
results are always derived from the poses; if the iteration budget is exhausted
without meeting the stop criteria, the last iterate is returned (it still
reconstructs its world point at its reported time) and a ``RuntimeWarning`` is
issued. Only self-consistent points are returned: the firing time implied by the
returned sensor-frame point must be within half a column of the reported time.
A point whose direction crosses the rear seam during the sweep may have no such
time (it lies in the scan gap) and is dropped; where two solutions exist (scan
overlap) the one reached from the start-pose firing time is kept. Default mode stops on the mean time change over all finite points;
``max_error=True`` freezes each point at its own convergence, so a point's result
never depends on other points. Float16/bfloat16 are rejected.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch

IMPLEMENTATION_AVAILABLE = True
"""Renderer gate; set False to disable prepared-scene LiDAR rendering again."""

UNAVAILABLE_REASON = (
    "Prepared-scene LiDAR rendering is unavailable in this build. Use RGB-only mode."
)

_SPAN_SLACK_RAD = 1e-6
_RANGE_MAP_MARGIN_RAD = math.radians(0.1)
_POSE_TOLERANCE = 1e-3
_FLOATS = (torch.float32, torch.float64)


def require_implementation() -> None:
    """Reject rendering while ``IMPLEMENTATION_AVAILABLE`` is False."""
    if not IMPLEMENTATION_AVAILABLE:
        raise NotImplementedError(UNAVAILABLE_REASON)


def _calibration(value, name: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric 1-D array") from None
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SpinningLidarModel:
    """Validated spinning-LiDAR calibration with nominal ray/lookup geometry."""

    row_elevations_rad: np.ndarray
    """Ring elevations (radians), one per row."""
    row_azimuth_offsets_rad: np.ndarray
    """Per-ring firing offsets (radians)."""
    n_columns: int = 3600
    """Range-map width."""
    spinning_frequency_hz: float = 10.0
    """Scan frequency."""
    sensor_name: str | None = None
    """Optional sensor identifier."""
    sorted_ring_lookup: bool = False
    """Use binary search for ring lookup (needs strictly descending rings)."""

    def __post_init__(self) -> None:
        elevations = _calibration(self.row_elevations_rad, "row_elevations_rad")
        offsets = _calibration(self.row_azimuth_offsets_rad, "row_azimuth_offsets_rad")
        if elevations.shape != offsets.shape:
            raise ValueError(
                "ring elevations and azimuth offsets must have equal length"
            )
        if np.abs(elevations).max() > math.pi / 2:
            raise ValueError("ring elevations must lie within [-pi/2, pi/2]")
        columns = self.n_columns
        if (
            isinstance(columns, (bool, np.bool_))
            or not isinstance(columns, (int, np.integer))
            or columns < 2
        ):
            raise ValueError(f"n_columns must be an integer >= 2, got {columns!r}")
        frequency = self.spinning_frequency_hz
        if isinstance(frequency, (bool, np.bool_)) or not isinstance(
            frequency, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"invalid spinning frequency {frequency!r}")
        if not (math.isfinite(frequency) and frequency > 0):
            raise ValueError(f"invalid spinning frequency {frequency!r}")
        if not isinstance(self.sorted_ring_lookup, (bool, np.bool_)):
            raise ValueError("sorted_ring_lookup must be a bool")
        if self.sensor_name is not None and not isinstance(self.sensor_name, str):
            raise ValueError("sensor_name must be a string or None")
        if self.sorted_ring_lookup and not np.all(
            np.diff(elevations.astype(np.float32)) < 0
        ):
            raise ValueError(
                "sorted ring lookup requires distinct, strictly descending rings"
            )
        object.__setattr__(self, "row_elevations_rad", elevations)
        object.__setattr__(self, "row_azimuth_offsets_rad", offsets)
        object.__setattr__(self, "n_columns", int(columns))
        object.__setattr__(self, "spinning_frequency_hz", float(frequency))

    @property
    def n_rows(self) -> int:
        return len(self.row_elevations_rad)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_columns)

    @property
    def column_azimuths_rad(self) -> np.ndarray:
        """Nominal leading-edge azimuth of each column (clockwise from +pi)."""
        return math.pi - np.arange(self.n_columns) * (2 * math.pi / self.n_columns)

    @property
    def row_elevations_deg(self) -> np.ndarray:
        return np.degrees(self.row_elevations_rad)

    @property
    def sweep_duration_us(self) -> float:
        return 1e6 / self.spinning_frequency_hz

    def ray_directions(self) -> np.ndarray:
        """Unit rays ``[R, C, 3]`` through the centres of the nominal cells."""
        azimuth = self.column_azimuths_rad - math.pi / self.n_columns
        elevation = self.row_elevations_rad[:, None]
        flat = np.cos(elevation)
        return np.stack(
            (
                flat * np.cos(azimuth)[None, :],
                flat * np.sin(azimuth)[None, :],
                np.broadcast_to(np.sin(elevation), self.shape),
            ),
            axis=-1,
        )

    def _rings(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.row_elevations_rad, dtype=like.dtype, device=like.device
        )

    def nearest_rows(
        self, elevation_rad: torch.Tensor, chunk_size: int = 131072
    ) -> torch.Tensor:
        """Row of minimum angular distance for each elevation (ties: lower row)."""
        _check_tensor(elevation_rad, 1, "elevation_rad")
        if (
            isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1
        ):
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )
        rings = self._rings(elevation_rad)
        if self.sorted_ring_lookup:
            ascending = rings.flip(0)
            above = torch.searchsorted(ascending, elevation_rad)
            hi = above.clamp(max=len(ascending) - 1)
            lo = (above - 1).clamp(min=0)
            pick_lo = (elevation_rad - ascending[lo]).abs() < (
                ascending[hi] - elevation_rad
            ).abs()
            return len(ascending) - 1 - torch.where(pick_lo, lo, hi)
        rows = torch.empty(
            len(elevation_rad), dtype=torch.int64, device=elevation_rad.device
        )
        for start in range(0, len(elevation_rad), chunk_size):
            block = elevation_rad[start : start + chunk_size]
            rows[start : start + chunk_size] = (
                (block[:, None] - rings[None, :]).abs().argmin(dim=1)
            )
        return rows

    def sensor_angles(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Elevation, azimuth and validity (finite, nonzero, inside the ring span)."""
        _check_tensor(points, 2, "points", width=3)
        x, y, z = points.unbind(dim=1)
        flat = torch.hypot(x, y)
        usable = torch.isfinite(points).all(dim=1) & ((flat > 0) | (z != 0))
        zero = torch.zeros_like(flat)
        elevation = torch.where(usable, torch.atan2(z, flat), zero)
        azimuth = torch.where(usable, torch.atan2(y, x), zero)
        rings = self.row_elevations_rad
        valid = (
            usable
            & (elevation >= float(rings.min()) - _SPAN_SLACK_RAD)
            & (elevation <= float(rings.max()) + _SPAN_SLACK_RAD)
        )
        return elevation, azimuth, valid

    def relative_frame_times(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized firing time in ``[0, 1]`` and validity for sensor-frame points."""
        elevation, azimuth, valid = self.sensor_angles(points)
        offsets = torch.tensor(
            self.row_azimuth_offsets_rad, dtype=points.dtype, device=points.device
        )[self.nearest_rows(elevation)]
        step = 2 * math.pi / self.n_columns
        columns = torch.remainder(
            (math.pi - (azimuth - offsets)) / step, self.n_columns
        )
        time = columns.clamp(0, self.n_columns - 1) / (self.n_columns - 1)
        usable = torch.isfinite(points).all(dim=1) & points.any(dim=1)
        return torch.where(usable, time, torch.zeros_like(time)), valid


@dataclass
class ShutterProjection:
    """Sensor-frame points that can return, with firing times and original indices."""

    points_sensor: torch.Tensor
    """Sensor-frame points ``[M, 3]``."""
    relative_time: torch.Tensor
    """Normalized scan times ``[M]``."""
    valid_indices: torch.Tensor
    """Original point indices ``[M]`` (int64)."""


def _check_tensor(value, ndim: int, name: str, width: int | None = None) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or (width is not None and value.shape[-1] != width)
        or value.dtype not in _FLOATS
    ):
        raise ValueError(
            f"{name} must be a float32/float64 tensor of "
            f"{ndim} dims{'' if width is None else f' ending in {width}'}"
        )


def _pose(value, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Validated rotation (projected onto SO(3)) and translation of a 4x4 pose."""
    try:
        matrix = torch.as_tensor(value).detach().cpu().double().numpy()
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(f"{name} must be a [4, 4] pose") from None
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite [4, 4] pose")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=_POSE_TOLERANCE)
        or not np.linalg.det(rotation) > 0
    ):
        raise ValueError(f"{name} must be a proper rigid transform")
    u, _, vt = np.linalg.svd(rotation)
    return u @ vt, matrix[:3, 3].copy()


def _log_rotation(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis and angle in ``[0, pi]`` of a rotation matrix."""
    sine_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    sine_vector /= 2
    norm = float(np.linalg.norm(sine_vector))
    angle = math.atan2(norm, (np.trace(rotation) - 1) / 2)
    if norm > 1e-6:
        return sine_vector / norm, angle
    if angle < 1:  # identity
        return (sine_vector / norm if norm > 0 else np.array([0.0, 0.0, 1.0])), angle
    outer = (rotation + np.eye(3)) / 2  # half-turn: axis * axis^T
    axis = outer[:, int(np.argmax(np.diag(outer)))]
    axis = axis / np.linalg.norm(axis)
    if norm > 0 and axis @ sine_vector < 0:
        axis = -axis
    elif norm == 0 and axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return axis, angle


class _Motion:
    """Shortest-arc rigid interpolation between two sensor-to-world poses."""

    def __init__(self, start, end, dtype: torch.dtype, device: torch.device) -> None:
        r0, p0 = _pose(start, "transform_start")
        r1, p1 = _pose(end, "transform_end")
        axis, self.angle = _log_rotation(r0.T @ r1)
        as_tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        self.r0, self.p0, self.dp, self.axis = map(as_tensor, (r0, p0, p1 - p0, axis))
        k = torch.zeros(3, 3, dtype=dtype, device=device)
        k[0, 1], k[0, 2] = -self.axis[2], self.axis[1]
        k[1, 0], k[1, 2] = self.axis[2], -self.axis[0]
        k[2, 0], k[2, 1] = -self.axis[1], self.axis[0]
        self.k = k

    def rotations(self, time: torch.Tensor) -> torch.Tensor:
        theta = time * self.angle
        sine, cosine = theta.sin()[:, None, None], theta.cos()[:, None, None]
        eye = torch.eye(3, dtype=time.dtype, device=time.device)
        return self.r0 @ (eye + sine * self.k + (1 - cosine) * (self.k @ self.k))

    def translations(self, time: torch.Tensor) -> torch.Tensor:
        return self.p0 + time[:, None] * self.dp

    def to_sensor(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """``R(t)^T (p - t(t))`` without building per-point matrices."""
        local = (points - self.translations(time)) @ self.r0
        theta = time * self.angle
        sine, cosine = theta.sin()[:, None], theta.cos()[:, None]
        axis = self.axis.expand_as(local)
        return (
            local * cosine
            - torch.linalg.cross(axis, local, dim=1) * sine
            + axis * (local @ self.axis)[:, None] * (1 - cosine)
        )


def interpolate_rigid(
    transform_start: torch.Tensor,
    transform_end: torch.Tensor,
    relative_time: torch.Tensor,
    quaternions: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shortest-arc slerp / lerp between two poses at times in ``[0, 1]``.

    Results use ``relative_time``'s dtype and device.
    """
    _check_tensor(relative_time, 1, "relative_time")
    if quaternions is not None and len(quaternions) != 2:
        raise ValueError("quaternions must be a pair of tensors")
    if not bool(torch.isfinite(relative_time).all()) or bool(
        ((relative_time < 0) | (relative_time > 1)).any()
    ):
        raise ValueError("relative_time must be finite and within [0, 1]")
    motion = _Motion(
        transform_start, transform_end, relative_time.dtype, relative_time.device
    )
    return motion.rotations(relative_time), motion.translations(relative_time)


def apply_rigid_inverse(
    rotations: torch.Tensor, translations: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Per-point ``R^T (p - t)``."""
    _check_tensor(rotations, 3, "rotations")
    _check_tensor(translations, 2, "translations", width=3)
    _check_tensor(points, 2, "points", width=3)
    n = len(points)
    if rotations.shape != (n, 3, 3) or translations.shape != (n, 3):
        raise ValueError("rotations/translations must match the number of points")
    return torch.einsum("nji,nj->ni", rotations, points - translations)


def _positive_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


def world_points_to_shutter_sensor_frame(
    model: SpinningLidarModel,
    world_points: torch.Tensor,
    lidar_to_world_start: torch.Tensor | np.ndarray,
    lidar_to_world_end: torch.Tensor | np.ndarray,
    max_iterations: int = 10,
    stop_mean_relative_time_error: float = 1e-4,
    stop_delta_mean_relative_time_error: float = 1e-6,
    reuse_quaternions: bool = False,
    max_error: bool = False,
) -> ShutterProjection:
    """Find each point's firing time and sensor-frame position during a moving sweep.

    Fixed-point iteration: time -> sensor pose at that time -> sensor-frame firing
    time of the point.

    ponytail: plain fixed-point iteration stalls or diverges for points within about
    ``motion per sweep / 2pi`` of the sensor (sub-metre at road speeds), where the
    point's apparent azimuth moves nearly as fast as the beam. Those points fail the
    self-consistency check and are dropped with the non-convergence warning; a
    per-point bracketed secant solve would recover them if ever needed. Returns the self-consistent points inside the vertical field
    of view at their reported time, with original indices.
    """
    if not isinstance(model, SpinningLidarModel):
        raise ValueError("model must be a SpinningLidarModel")
    _check_tensor(world_points, 2, "world_points", width=3)
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError(
            f"max_iterations must be a positive integer, got {max_iterations!r}"
        )
    stop_mean = _positive_number(
        stop_mean_relative_time_error, "stop_mean_relative_time_error"
    )
    stop_delta = _positive_number(
        stop_delta_mean_relative_time_error, "stop_delta_mean_relative_time_error"
    )
    device, dtype = world_points.device, world_points.dtype
    motion = _Motion(lidar_to_world_start, lidar_to_world_end, dtype, device)

    finite = torch.isfinite(world_points).all(dim=1)
    points = torch.where(finite[:, None], world_points, torch.zeros_like(world_points))
    time = model.relative_frame_times(
        motion.to_sensor(points, torch.zeros(len(points), dtype=dtype, device=device))
    )[0]
    active = finite.clone()
    previous = None
    converged = not bool(finite.any())
    for _ in range(0 if converged else max_iterations):
        updated = model.relative_frame_times(motion.to_sensor(points, time))[0]
        error = (updated - time).abs()
        time = torch.where(active, updated, time)
        # A jump of over half a sweep is a wrap across the rear seam, not progress;
        # such points keep iterating but do not hold up the stop criteria. The
        # self-consistency filter below decides whether they return.
        settling = finite & (error <= 0.5)
        if max_error:
            done = error < stop_mean
            if previous is not None:
                done |= (previous - error).abs() < stop_delta
            previous = error
            active = active & ~done
            converged = not bool((active & settling).any())
        else:
            mean = float(error[settling].mean()) if bool(settling.any()) else 0.0
            converged = mean < stop_mean or (
                previous is not None and abs(previous - mean) < stop_delta
            )
            previous = mean
        if converged:
            break
    if not converged:
        warnings.warn(
            f"Shutter projection did not meet its stop criteria within "
            f"{max_iterations} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    sensor = motion.to_sensor(points, time)
    implied, valid = model.relative_frame_times(sensor)
    # Keep only self-consistent firings: the beam that fires at ``time`` must point
    # within half a column of the point. A point whose direction crosses the rear
    # seam during the sweep may have no such time (it falls in the scan gap).
    consistent = (implied - time).abs() <= 0.5 / (model.n_columns - 1)
    valid &= finite & consistent
    indices = valid.nonzero().squeeze(1)
    return ShutterProjection(sensor[indices], time[indices], indices)


def points_to_range_map(
    points: torch.Tensor,
    model: SpinningLidarModel,
    max_range: float = 100.0,
    intensity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-return range and intensity images ``[R, C]`` (float32, empty cells 0).

    A point is eligible when ``0 < range < max_range`` and its elevation is within
    0.1 degrees of the ring span; it lands in its nearest ring and nominal column.
    """
    _check_tensor(points, 2, "points", width=3)
    max_range = _positive_number(max_range, "max_range")
    if not isinstance(model, SpinningLidarModel):
        raise ValueError("model must be a SpinningLidarModel")
    if intensity is not None:
        _check_tensor(intensity, 1, "intensity")
        if intensity.shape != (len(points),) or intensity.device != points.device:
            raise ValueError("intensity must be [N] on the same device as points")
        if not bool(torch.isfinite(intensity).all()):
            raise ValueError("intensity must be finite")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")

    rows_n, columns = model.shape
    cells = rows_n * columns
    best = torch.full((cells,), math.inf, dtype=points.dtype, device=points.device)
    chosen = torch.full((cells,), len(points), dtype=torch.int64, device=points.device)

    distance = torch.linalg.norm(points, dim=1)
    elevation = torch.atan2(points[:, 2], torch.hypot(points[:, 0], points[:, 1]))
    rings = model.row_elevations_rad
    eligible = (
        (distance > 0)
        & (distance.double() < max_range)
        & (elevation >= float(rings.min()) - _RANGE_MAP_MARGIN_RAD)
        & (elevation <= float(rings.max()) + _RANGE_MAP_MARGIN_RAD)
    )
    index = eligible.nonzero().squeeze(1)
    if len(index):
        row = model.nearest_rows(elevation[index])
        azimuth = torch.atan2(points[index, 1], points[index, 0])
        step = 2 * math.pi / columns
        column = ((math.pi - azimuth.double()) / step).floor().clamp(0, columns - 1)
        cell = row * columns + column.long()
        reach = distance[index]
        best.scatter_reduce_(0, cell, reach, "amin")
        nearest = reach == best[cell]
        chosen.scatter_reduce_(0, cell[nearest], index[nearest], "amin")
    hit = chosen < len(points)
    ranges = (
        torch.where(hit, best, torch.zeros_like(best)).float().reshape(rows_n, columns)
    )
    values = torch.zeros(cells, dtype=torch.float32, device=points.device)
    if intensity is not None and bool(hit.any()):
        values[hit] = intensity[chosen[hit]].float()
    return ranges, values.reshape(rows_n, columns)
