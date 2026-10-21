# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated plain-text LiDAR assets and controls for supplied driving poses."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import torch

from . import _lidar_geometry as geometry
from . import _lidar_projection as projection
from ._lidar_projection import (
    SpinningLidarModel,
    points_to_range_map,
    world_points_to_shutter_sensor_frame,
)

SETTINGS = {
    "columns": 1800,
    "frequency_hz": 10,
    "rgb_fps": 30,
    "density_m": 0.02,
    "radius_m": 0.05,
    "min_range_m": 5,
    "max_range_m": 100,
    "sweeps": "opening instantaneous; subsequent trailing 3 RGB frames",
    "objects": "at sweep endpoint; hold final source frame after scene end",
    "projection": "clockwise rear-start; nearest ring; nominal azimuth bins",
}


def require_implementation() -> None:
    """Reject rendering before loading weights, allocating points or using caches."""
    geometry.require_implementation()
    projection.require_implementation()


def rigid(
    value, shape: tuple[int, ...], label: str, *, rotation_tolerance: float = 1e-4
) -> np.ndarray:
    """Read finite, proper rigid transforms without silently repairing a mount."""
    from .prepared_scene import _array

    result = _array(value, shape, np.float64)
    rotation = result[..., :3, :3]
    if (
        not np.allclose(result[..., 3, :], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(
            rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=rotation_tolerance
        )
        or not np.allclose(np.linalg.det(rotation), 1, atol=rotation_tolerance)
    ):
        raise ValueError(f"{label} must contain proper rigid transforms")
    return result


@dataclass
class PreparedLidar:
    """CPU geometry and explicit calibration, loaded once with the RGB scene."""

    model: SpinningLidarModel
    sensor_to_ego: np.ndarray
    camera_to_ego: np.ndarray
    reference_camera: str
    layers: dict[str, list[np.ndarray]]
    objects: list[tuple[np.ndarray, np.ndarray, np.ndarray]]

    def sensor_to_world(self, scene, frame: int) -> np.ndarray:
        """Locate the mount independently of the selected RGB anchor camera."""
        return (
            scene.tracks[self.reference_camera][frame].astype(np.float64)
            @ np.linalg.inv(self.camera_to_ego)
            @ self.sensor_to_ego
        )

    def object_points(
        self,
        frame: int,
        *,
        density: float = geometry.DEFAULT_DENSITY_M,
        budget: geometry.PointBudget | None = None,
    ) -> np.ndarray:
        """Sample present actors within ``budget`` (a new scene budget if omitted)."""
        lines = []
        for frames, sizes, poses in self.objects:
            index = int(np.searchsorted(frames, frame))
            if index < len(frames) and frames[index] == frame:
                lines.append(geometry.cuboid_polyline(sizes[index], poses[index]))
        # Match reference float64 sampling followed by one float32 conversion.
        return geometry.points_from_polylines(
            lines, density=density, budget=budget
        ).astype(np.float32)

    def actor_point_capacity(self, density: float) -> int:
        """Bound the largest actor frame without allocating sampled point clouds."""
        if not self.objects:
            return 0
        offsets = len(
            geometry._disk(
                geometry.DEFAULT_RADIUS_M, density, geometry.MAX_PRIMITIVE_POINTS
            )
        )
        totals: dict[int, int] = defaultdict(int)
        for frames, sizes, poses in self.objects:
            lengths = sizes * np.linalg.norm(poses[:, :3, :3], axis=1)
            # The cuboid walk has 4 length, 4 width and 7 height edges.
            # Reserve an extra station per edge for world-coordinate rounding.
            counts = ((np.ceil(lengths / density) + 2) * [4, 4, 7]).sum(
                axis=1
            ) * offsets
            if np.any(counts > geometry.MAX_PRIMITIVE_POINTS):
                raise geometry.PointLimitError(
                    "LiDAR actor exceeds primitive point cap"
                )
            for frame, count in zip(frames, counts):
                totals[int(frame)] += int(count)
        return max(totals.values(), default=0)


def load(data: dict, manifest: dict) -> PreparedLidar:
    """Validate the v2 LiDAR extension before any model or GPU allocation."""
    from .prepared_scene import _array

    if (
        data.get("settings") != SETTINGS
        or data.get("coordinates") != "metres; world and ego FLU"
    ):
        raise ValueError("Unsupported prepared LiDAR rendering settings/coordinates")
    profile = data["profile"]
    elevations = _array(profile["row_elevations_rad"], (128,), np.float64)
    offsets = _array(profile["row_azimuth_offsets_rad"], (128,), np.float64)
    if (
        not np.all(np.diff(elevations) < 0)
        or np.max(np.abs(elevations)) >= np.pi / 2
        or np.max(np.abs(offsets)) > np.pi
    ):
        raise ValueError("Invalid LiDAR ring calibration")
    for name in ("source_url", "source_revision", "source_sha256", "mount_description"):
        if (
            not isinstance(data["provenance"].get(name), str)
            or not data["provenance"][name].strip()
        ):
            raise ValueError(f"LiDAR provenance is missing {name}")
    camera = data["reference_camera"]
    if camera not in manifest["camera_order"]:
        raise ValueError("LiDAR reference camera is absent from the scene")
    sensor = rigid(data["sensor_to_ego"], (4, 4), "sensor_to_ego")
    camera_mount = rigid(data["camera_to_ego"], (4, 4), "camera_to_ego")
    if set(data["layers"]) != set(geometry.DEFAULT_MINIMAP_TYPES):
        raise ValueError("Prepared LiDAR requires all eight semantic map layers")
    layers = {}
    for name, records in data["layers"].items():
        minimum = 2 if geometry.LAYER_TYPES[name] == "polyline" else 3
        layers[name] = []
        for record in records:
            if not isinstance(record, list) or len(record) < minimum:
                raise ValueError(f"Invalid LiDAR primitive in {name}")
            vertices = _array(record, (len(record), 3), np.float64)
            layers[name].append(vertices)
    objects = []
    ids = []
    for track in data["objects"]:
        ids.append(track["id"])
        frames = _array(track["frames"], (len(track["frames"]),), np.int64)
        if (
            not len(frames)
            or np.any(np.diff(frames) <= 0)
            or frames[0] < 0
            or frames[-1] >= manifest["frames"]
        ):
            raise ValueError("LiDAR actor frames must be ordered and inside the scene")
        sizes = _array(track["lwh"], (len(frames), 3), np.float64)
        if np.any(sizes <= 0):
            raise ValueError("LiDAR actor dimensions must be positive")
        # Reference static-object averaging normalizes axes independently, leaving
        # slight non-orthogonality. Preserve those boxes instead of rotating them.
        poses = rigid(
            track["poses"],
            (len(frames), 4, 4),
            "LiDAR actor poses",
            rotation_tolerance=1e-3,
        )
        objects.append((frames, sizes, poses))
    if any(not isinstance(key, str) for key in ids) or ids != sorted(set(ids)):
        raise ValueError("LiDAR actor IDs must be unique and sorted")
    return PreparedLidar(
        SpinningLidarModel(
            elevations, offsets, n_columns=1800, spinning_frequency_hz=10
        ),
        sensor,
        camera_mount,
        camera,
        layers,
        objects,
    )


class Renderer:
    """Keep static points on device and render completed scans from live poses."""

    def __init__(
        self,
        scene,
        *,
        device: torch.device | str,
        optimized: bool = False,
        cull: bool = False,
        settle: bool = False,
    ) -> None:
        self.data = scene.lidar
        if self.data is None:
            raise ValueError(
                "LiDAR requires a v2 prepared scene; use the LiDAR scene ZIP"
            )
        require_implementation()
        self.device = torch.device(device)
        self.optimized = optimized
        self.cull = cull
        # Per-point settling without the cull: the reference the culled path
        # should match exactly.
        self.settle = settle
        self.mapping = scene.frame_mapping(0, None)
        self.density = geometry.DEFAULT_DENSITY_M
        for attempt in range(32):
            try:
                actors = self.data.actor_point_capacity(self.density)
                if actors > geometry.MAX_SCENE_POINTS:
                    raise geometry.PointLimitError(
                        "LiDAR actors exceed scene point cap"
                    )
                budget = geometry.PointBudget(geometry.MAX_SCENE_POINTS - actors)
                layers = [
                    geometry.minimap_points(
                        name,
                        self.data.layers[name],
                        density=self.density,
                        dtype=np.float32,
                        budget=budget,
                    )
                    for name in geometry.DEFAULT_MINIMAP_TYPES
                ]
                break
            except geometry.PointLimitError:
                if attempt == 31:
                    raise
                self.density *= 2
        if self.density != geometry.DEFAULT_DENSITY_M:
            logging.getLogger(__name__).warning(
                "LiDAR point cap: using %.3g m spacing for the entire scene "
                "(default %.3g m); all map primitives and actors retained",
                self.density,
                geometry.DEFAULT_DENSITY_M,
            )
        self.static = torch.as_tensor(np.concatenate(layers), device=self.device)

    @torch.no_grad()
    def __call__(self, index: int, first: np.ndarray, last: np.ndarray) -> torch.Tensor:
        frame = int(self.mapping[min(index * 3, len(self.mapping) - 1)])
        budget = geometry.PointBudget(geometry.MAX_SCENE_POINTS - len(self.static))
        objects = torch.as_tensor(
            self.data.object_points(frame, density=self.density, budget=budget),
            device=self.device,
        )
        points = torch.cat((objects, self.static))
        fast = self.optimized or self.cull
        model = (
            replace(self.data.model, sorted_ring_lookup=True)
            if fast
            else self.data.model
        )
        stationary = np.array_equal(first, last)
        if (self.optimized and stationary) or self.cull:
            # Keep only points that can return within 100 m from *some* sensor
            # position of the sweep: translation is lerped between endpoints,
            # so every firing position lies within |end - start| of the start.
            # Stationary scans are time-independent, so this cull is exact
            # there. Moving scans also settle each point on its own below: the
            # default mean-error stop depends on the culled far points. Keep
            # <5 m occluders and pad for float32.
            center = torch.as_tensor(
                first[:3, 3], device=self.device, dtype=points.dtype
            )
            motion = float(np.linalg.norm(last[:3, 3] - first[:3, 3]))
            radius = 100.01 + motion
            points = points[((points - center).square().sum(dim=1) <= radius**2)]
        projected = world_points_to_shutter_sensor_frame(
            model,
            points,
            first,
            last,
            reuse_quaternions=fast,
            max_error=(self.cull or self.settle) and not stationary,
        )
        ranges, _ = points_to_range_map(projected.points_sensor, model, max_range=100)
        # Apply the training validity threshold AFTER nearest-return occlusion.
        valid = (ranges >= 5) & (ranges < 100)
        return torch.stack((ranges * valid, torch.zeros_like(ranges), valid.float()))
