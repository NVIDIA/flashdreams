# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render recorded or simulated drives into a synchronized comparison video."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from .camera import FThetaCameraModel
from .config import VehicleConfig
from .types import CameraCalibration
from .vehicle_mesh import chase_projection, render_vehicle

CAMERAS = {
    "camera_front_wide_120fov": "CAM_FRONT",
    "camera_cross_right_120fov": "CAM_FRONT_RIGHT",
    "camera_rear_right_70fov": "CAM_BACK_RIGHT",
    "camera_rear_tele_30fov": "CAM_BACK",
    "camera_rear_left_70fov": "CAM_BACK_LEFT",
    "camera_cross_left_120fov": "CAM_FRONT_LEFT",
    "camera_front_tele_30fov": "CAM_FRONT_TELE",
}
LABELS = {
    "CAM_FRONT": "FRONT WIDE",
    "CAM_FRONT_LEFT": "LEFT FRONT",
    "CAM_FRONT_RIGHT": "RIGHT FRONT",
    "CAM_BACK_LEFT": "LEFT REAR",
    "CAM_BACK_RIGHT": "RIGHT REAR",
    "CAM_BACK": "REAR",
    "CAM_FRONT_TELE": "FRONT TELE",
}
BG, PANEL, LINE, TEXT, MUTED = "#080f15", "#0d171f", "#243540", "#e6edf3", "#8195a5"
FONTS = {
    size: ImageFont.truetype(
        "DejaVuSansCondensed-Bold.ttf" if size >= 16 else "DejaVuSansCondensed.ttf",
        size,
    )
    for size in (11, 12, 14, 16, 18, 20, 22, 24, 26, 30, 38, 42, 54)
}


def asset(root: Path, name: str) -> Path:
    """Resolve a recording asset without escaping its directory."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes the recording directory")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def nearest(times: np.ndarray, timestamp: int) -> int:
    """Select the closest sample, preferring the earlier one on ties."""
    index = int(np.searchsorted(times, timestamp))
    if index == 0:
        return 0
    if index == len(times):
        return index - 1
    return (
        index - 1 if timestamp - times[index - 1] <= times[index] - timestamp else index
    )


def project_raw_camera(
    points: np.ndarray, pose: np.ndarray, calibration: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with the reference's eight Brown distortion terms."""
    transform = np.linalg.inv(pose @ np.asarray(calibration["sensor_to_ego_matrix"]))
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    x, y = (camera[:, :2] / np.maximum(camera[:, 2:3], 0.05)).T
    r2 = x * x + y * y
    k1, k2, p1, p2, k3, k4, k5, k6 = calibration["D"]
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
        1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    )
    distorted = np.stack(
        (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            np.ones_like(x),
        ),
        axis=-1,
    )
    pixels = distorted @ np.asarray(calibration["K"]).T
    return pixels[:, :2] / [calibration["width"], calibration["height"]], camera[:, 2]


def raw_camera_radius(calibration: dict[str, Any]) -> float:
    """Find the first radial fold or pole of the Brown projection."""
    k1, k2, _, _, k3, k4, k5, k6 = calibration["D"]
    polynomial = np.polynomial.Polynomial
    numerator, denominator = polynomial([1, k1, k2, k3]), polynomial([1, k4, k5, k6])
    slope = numerator * denominator + polynomial([0, 2]) * (
        numerator.deriv() * denominator - numerator * denominator.deriv()
    )
    roots = np.concatenate([p.roots() for p in (numerator, denominator, slope)])
    positive = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
    return math.sqrt(min(positive)) * 0.999 if positive else math.inf


def future_edges(source: dict[str, Any], index: int) -> np.ndarray:
    """Place the recorded four-second ribbon at the configured vehicle floor."""
    last = np.searchsorted(source["times"], source["times"][index] + 4_000_000)
    poses = np.array(
        [frame["pose"] for frame in source["frames"][index : max(index + 1, last)]]
    )
    # ponytail: The rig uses its configured half-height as the floor offset;
    # surveyed road elevation would improve the overlay on uneven surfaces.
    centers = poses[:, :3, 3] - poses[:, :3, 2] * source["floor_offset"]
    side = poses[:, :3, 1] * source["ribbon_half_width"]
    return np.stack((centers + side, centers - side), axis=1)


def clip_polygon(vertices: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Clip a polygon against a plane whose interior has nonnegative distance."""
    result = []
    for i, point in enumerate(vertices):
        previous, distance, previous_distance = (
            vertices[i - 1],
            distances[i],
            distances[i - 1],
        )
        if (distance >= 0) != (previous_distance >= 0):
            result.append(
                previous
                + (point - previous)
                * previous_distance
                / (previous_distance - distance)
            )
        if distance >= 0:
            result.append(point)
    return np.asarray(result, dtype=float).reshape(-1, vertices.shape[-1])


def stabilize_reference_speed(
    times_s: np.ndarray,
    speeds_mps: np.ndarray,
    positions_xy: np.ndarray,
    vehicle: VehicleConfig,
    smoothing_s: float = 0.6,
) -> np.ndarray:
    """Fit a smooth speed curve within native acceleration and travel limits."""
    from scipy.optimize import Bounds, LinearConstraint, minimize

    times_s, speeds_mps, positions_xy = map(
        np.asarray, (times_s, speeds_mps, positions_xy)
    )
    n = len(times_s)
    if (
        n < 3
        or times_s.shape != (n,)
        or speeds_mps.shape != (n,)
        or positions_xy.shape != (n, 2)
        or not all(np.isfinite(a).all() for a in (times_s, speeds_mps, positions_xy))
        or np.any(np.diff(times_s) <= 0)
        or np.any(speeds_mps < 0)
        or not math.isfinite(smoothing_s)
        or smoothing_s <= 0
    ):
        raise ValueError(
            "Speed fitting requires finite forward states at increasing timestamps"
        )
    if not all(
        math.isfinite(v) and v > 0
        for v in (vehicle.max_accel_mps2, vehicle.max_brake_mps2)
    ):
        raise ValueError(
            "Reference speed fitting requires positive acceleration limits"
        )
    dt = np.diff(times_s)
    acceleration = np.diff(np.eye(n), axis=0) / dt[:, None]
    jerk = np.diff(acceleration, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
    weights = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    distance = float(np.linalg.norm(np.diff(positions_xy, axis=0), axis=1).sum())
    maximum_speed = vehicle.max_speed_mps if vehicle.speed_limit_enabled else np.inf
    if distance > maximum_speed * (times_s[-1] - times_s[0]):
        raise ValueError("Reference travel exceeds the configured speed limit")
    hessian = np.eye(n) / n + smoothing_s**4 * jerk.T @ jerk / len(jerk)
    target = speeds_mps / n
    # ponytail: Dense offline fitting targets short presentation clips;
    # hour-long recordings need a sparse or overlapping-window solver.
    result = minimize(
        lambda v: 0.5 * v @ hessian @ v - target @ v,
        np.clip(np.linalg.solve(hessian, target), 0, maximum_speed),
        jac=lambda v: hessian @ v - target,
        method="SLSQP",
        bounds=Bounds(0, maximum_speed),
        constraints=[
            LinearConstraint(
                acceleration, -vehicle.max_brake_mps2, vehicle.max_accel_mps2
            ),
            LinearConstraint(weights, distance, distance),
        ],
        options={"ftol": 1e-10, "maxiter": 300},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Reference speed fit failed: {result.message}")
    return result.x


def load_source(path: Path, scene: Path) -> dict[str, Any]:
    """Read raw RGB references or a finalized Web recording."""
    path = path.resolve()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = []
    speed_estimation = None
    front_camera = None
    vehicle_config = manifest.get("config", {}).get("vehicle", {})
    if manifest.get("format") == "flashdreams.drive-recording":
        if manifest.get("status") != "ready":
            raise ValueError(
                "Use a completed recording; partial sessions are retained for recovery"
            )
        for line in asset(path, manifest["frames_path"]).read_text().splitlines():
            frame = json.loads(line)
            vehicle = frame["vehicle"]
            records.append(
                {
                    "timestamp_us": int(frame["timestamp_us"]),
                    "images": {
                        CAMERAS.get(k, k): asset(path, v)
                        for k, v in frame["images"].items()
                    },
                    "pose": np.asarray(frame["rig_to_world"], dtype=float),
                    "speed": float(vehicle["speed_mps"]),
                    "steer": float(vehicle["steer_rad"]),
                    "yaw_rate": float(
                        vehicle.get(
                            "yaw_rate_radps",
                            float(vehicle["speed_mps"])
                            * math.tan(float(vehicle["steer_rad"]))
                            / float(vehicle_config.get("wheel_base_m", 2.8)),
                        )
                    ),
                    "throttle": float(frame["command"]["throttle"]),
                    "brake": float(frame["command"]["brake"]),
                    "bev": asset(path, frame["bev"]) if frame.get("bev") else None,
                    "actors": frame.get("actors", []),
                }
            )
        for camera in manifest.get("camera_calibration", []):
            if camera["logical_name"] == "camera_front_wide_120fov":
                camera = dict(camera)
                for key in ("polynomial", "linear_cde", "sensor_to_rig_flu"):
                    camera[key] = np.asarray(camera[key], dtype=np.float32)
                front_camera = FThetaCameraModel(CameraCalibration(**camera))
                break
        estimated = False
    else:
        sample = json.loads((path / "sample.json").read_text())
        if sample.get("format") != "flashdreams.jlu-raw-multiview":
            raise ValueError(
                "Expected a raw multiview package or a Web drive recording"
            )
        tables = {
            view["camera_key"]: read_csv(asset(path, view["timestamps_path"]))
            for view in sample["multiview"]["views"]
        }
        controls = read_csv(asset(path, sample["estimated_controls_path"]))
        states = read_csv(asset(path, sample["estimated_vehicle_state_path"]))
        source_poses = read_csv(asset(path, "ego_poses.csv"))
        calibration = json.loads(asset(path, sample["calibration_path"]).read_text())
        front_camera = calibration["cameras"]["CAM_FRONT"]
        scene_manifest = json.loads((scene / "scene.json").read_text())
        camera_poses = np.asarray(
            json.loads((scene / "poses.json").read_text())["camera_front_wide_120fov"]
        )
        mount = np.asarray(calibration["cameras"]["CAM_FRONT"]["sensor_to_ego_matrix"])
        poses = camera_poses @ np.linalg.inv(mount)
        pose_times = int(scene_manifest["start_us"]) + np.arange(
            len(poses), dtype=np.int64
        ) * round(1e6 / float(scene_manifest.get("fps", 10)))
        raw_times = np.array([int(row["timestamp_ns"]) // 1000 for row in source_poses])
        raw_speed = np.array([float(row["metadata_speed_mps"]) for row in source_poses])
        dynamics = json.loads(
            asset(
                path, sample.get("dynamics_model_path", "dynamics_model.json")
            ).read_text()
        )
        vehicle = VehicleConfig(**dynamics["vehicle_config"])
        vehicle_config = dynamics["vehicle_config"]
        state_ns = np.array(
            [int(row["timestamp_ns"]) for row in states], dtype=np.int64
        )
        state_times_s = (state_ns - state_ns[0]) / 1e9
        reference_speeds = stabilize_reference_speed(
            state_times_s,
            np.array([float(row["speed_est_mps"]) for row in states]),
            np.array(
                [[float(row[k]) for k in ("x_fit_m", "y_fit_m")] for row in states]
            ),
            vehicle,
        )
        reference_acceleration = np.diff(reference_speeds) / np.diff(state_times_s)
        speed_estimation = {
            "method": "Pose speed fit with jerk regularization and FlashDreams acceleration bounds",
            "smoothing_s": 0.6,
            "distance_constraint": "Total fitted XY trajectory length; trapezoidal speed integral",
            "max_accel_mps2": vehicle.max_accel_mps2,
            "max_brake_mps2": vehicle.max_brake_mps2,
            "raw_metadata_preserved": True,
        }
        for index, row in enumerate(tables["CAM_FRONT"]):
            timestamp = int(row["index_unix_timestamp_ns"]) // 1000
            if timestamp > pose_times[-1] + 50_000:
                break
            control = controls[index]
            records.append(
                {
                    "timestamp_us": timestamp,
                    "images": {
                        camera: asset(path, rows[index]["image_path"])
                        for camera, rows in tables.items()
                    },
                    "pose": poses[nearest(pose_times, timestamp)],
                    "speed": float(reference_speeds[index]),
                    "speed_raw": float(
                        np.interp(
                            timestamp - raw_times[0],
                            raw_times - raw_times[0],
                            raw_speed,
                        )
                    ),
                    "acceleration": float(reference_acceleration[index]),
                    "steer": float(control["front_wheel_angle_est_rad"]),
                    # Native coast drag applies only when both pedals are zero.
                    "throttle": max(
                        1e-10,
                        float(reference_acceleration[index]) / vehicle.max_accel_mps2,
                    )
                    if reference_acceleration[index] >= 0
                    else 0.0,
                    "brake": max(
                        0.0,
                        -float(reference_acceleration[index]) / vehicle.max_brake_mps2,
                    ),
                    "bev": None,
                }
            )
        estimated = True
    if not records:
        raise ValueError("Source contains no usable frames")
    times = np.array([record["timestamp_us"] for record in records], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be strictly increasing")
    for record in records:
        if record["pose"].shape != (4, 4) or not np.isfinite(record["pose"]).all():
            raise ValueError("Invalid frame pose")
        if not all(
            math.isfinite(record[key])
            for key in ("speed", "steer", "throttle", "brake")
        ):
            raise ValueError("Non-finite vehicle state")
    speeds = np.array([record["speed"] for record in records])
    acceleration = (
        np.gradient(speeds, (times - times[0]) / 1e6) if len(times) > 1 else np.zeros(1)
    )
    for record, value in zip(records, acceleration, strict=True):
        record.setdefault("acceleration", float(value))
    acceleration = np.array([record["acceleration"] for record in records])
    jerk = (
        np.gradient(acceleration, (times - times[0]) / 1e6)
        if len(times) > 1
        else np.zeros(1)
    )
    distance = np.r_[
        0,
        np.cumsum(
            np.linalg.norm(np.diff([r["pose"][:2, 3] for r in records], axis=0), axis=1)
        ),
    ]
    for record, j, travel in zip(records, jerk, distance, strict=True):
        record["jerk"] = float(j)
        record["travel"] = float(travel)
        record["lateral_acceleration"] = record["speed"] * record.get(
            "yaw_rate",
            record["speed"]
            * math.tan(record["steer"])
            / float(vehicle_config.get("wheel_base_m", 2.8)),
        )
    return {
        "frames": records,
        "times": times,
        "estimated_controls": estimated,
        "speed_estimation": speed_estimation,
        "front_camera": front_camera,
        "floor_offset": float(vehicle_config.get("aabb_height_m", 1.6)) / 2,
        "ribbon_half_width": float(vehicle_config.get("aabb_width_m", 1.9)) / 2,
        "vehicle_config": vehicle_config,
        "name": path.name,
    }


def start_from_motion(source: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Copy a source onto a time axis beginning at its first moving frame."""
    first = next(
        (
            i
            for i, frame in enumerate(source["frames"])
            if abs(frame["speed"]) > threshold
        ),
        0,
    )
    origin = int(source["times"][first])
    travel = source["frames"][first].get("travel", 0.0)
    return {
        **source,
        "time_origin_us": origin,
        "trimmed_start_s": (origin - int(source["times"][0])) / 1e6,
        "times": source["times"][first:] - origin,
        "frames": [
            dict(
                frame,
                source_timestamp_us=frame["timestamp_us"],
                timestamp_us=int(frame["timestamp_us"]) - origin,
                travel=frame.get("travel", 0.0) - travel,
            )
            for frame in source["frames"][first:]
        ],
    }


class ComparisonRenderer:
    """Compose camera views, real scene geometry, telemetry, and data checks."""

    def __init__(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        scene: Path,
        template: dict[str, Any],
        ego_model: Path | None = None,
    ) -> None:
        self.start_speed_mps = float(template.get("start_speed_mps", 0.1))
        if not math.isfinite(self.start_speed_mps) or self.start_speed_mps < 0:
            raise ValueError("Start speed threshold must be finite and nonnegative")
        self.left = start_from_motion(left, self.start_speed_mps)
        self.right = (
            self.left
            if right is left
            else start_from_motion(right, self.start_speed_mps)
        )
        self.template = template
        self.steering_ratio = float(template.get("steering_ratio", 15.0))
        if not math.isfinite(self.steering_ratio) or not 1 <= self.steering_ratio <= 50:
            raise ValueError("Steering ratio must be finite and between 1 and 50")
        self.comfort_limits = {
            "acceleration": 3.0,
            "braking": 4.0,
            "lateral_acceleration": 3.0,
            "jerk": 5.0,
            "path_jump": 1.0,
        }
        self.comfort_limits.update(template.get("comfort_limits", {}))
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            for v in self.comfort_limits.values()
        ):
            raise ValueError("Comfort display limits must be finite and positive")
        self.speed_scale = max(
            60,
            math.ceil(
                max(abs(f["speed"]) * 3.6 for s in (left, right) for f in s["frames"])
                / 20
            )
            * 20,
        )
        self.ego_sprite = None
        if ego_model is not None:
            cached = ego_model.with_suffix(".png")
            if cached.exists() and cached.stat().st_mtime >= ego_model.stat().st_mtime:
                with Image.open(cached) as image:
                    self.ego_sprite = image.convert("RGBA")
            else:
                self.ego_sprite = render_vehicle(ego_model, (604, 330))
        geometry = json.loads((scene / "map.json").read_text())
        manifest = json.loads((scene / "scene.json").read_text())
        self.lines = [
            (
                np.asarray(layer["vertices"], dtype=float),
                tuple(manifest["palette"].get(str(layer["prim"]), [200, 210, 220])[:3]),
            )
            for layer in geometry["lines"]
        ]
        self.tracks = [
            track
            for group in json.loads((scene / "tracks.json").read_text())
            for track in group["tracks"]
        ]
        self.start = 0
        self.end = int(min(self.left["times"][-1], self.right["times"][-1]))
        self.reference_path = np.array(
            [frame["pose"][:2, 3] for frame in left["frames"]]
        )

    def text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        value: str,
        size: int = 16,
        color: Any = TEXT,
    ) -> None:
        draw.text(xy, value, font=FONTS[size], fill=color)

    def fitted_text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        value: str,
        width: int,
        size: int = 24,
        color: Any = TEXT,
    ) -> None:
        while size > 12 and FONTS[size].getlength(value) > width:
            size = max(s for s in FONTS if s < size)
        self.text(draw, xy, value, size, color)

    def panel(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
        draw.rounded_rectangle(box, radius=8, fill=PANEL, outline=LINE)

    def camera(
        self,
        canvas: Image.Image,
        box: tuple[int, int, int, int],
        path: Path | None,
        label: str,
        accent: str,
        crop: bool = False,
        zoom: float = 1,
        ribbon: list[np.ndarray] | None = None,
    ) -> None:
        draw = ImageDraw.Draw(canvas)
        x, y, width, height = box
        draw.rectangle((x, y, x + width, y + height), fill="#050a0e", outline=LINE)
        if path is not None:
            with Image.open(path) as image:
                if zoom > 1:
                    w, h = image.size
                    image = image.crop(
                        (
                            w / 2 - w / zoom / 2,
                            h / 2 - h / zoom / 2,
                            w / 2 + w / zoom / 2,
                            h / 2 + h / zoom / 2,
                        )
                    )
                resize = ImageOps.fit if crop else ImageOps.contain
                image = resize(image.convert("RGB"), (width - 2, height - 2))
                if ribbon is not None:
                    overlay = Image.new("RGBA", (image.width * 2, image.height * 2))
                    tint = tuple(int(accent[i : i + 2], 16) for i in (1, 3, 5))
                    mask = Image.new("L", overlay.size)
                    painter = ImageDraw.Draw(mask)
                    for quad in ribbon:
                        painter.polygon(
                            [tuple(p) for p in quad * overlay.size], fill=255
                        )
                    # Outline the union, so adjacent samples have no crossbars.
                    border = ImageChops.subtract(
                        mask, mask.filter(ImageFilter.MinFilter(5))
                    )
                    overlay = Image.new("RGBA", overlay.size, (*tint, 0))
                    overlay.putalpha(
                        ImageChops.add(
                            mask.point(lambda value: value * 60 // 255),
                            border.point(lambda value: value * 130 // 255),
                        )
                    )
                    overlay = overlay.resize(image.size, Image.Resampling.LANCZOS)
                    image = Image.alpha_composite(
                        image.convert("RGBA"), overlay
                    ).convert("RGB")
                canvas.paste(
                    image,
                    (x + (width - image.width) // 2, y + (height - image.height) // 2),
                )
        draw.rectangle(
            (
                x + 1,
                y + 1,
                x + min(width - 1, round(FONTS[11].getlength(label)) + 16),
                y + 19,
            ),
            fill=BG,
        )
        draw.rectangle((x + 1, y + 1, x + 4, y + 19), fill=accent)
        self.text(draw, (x + 9, y + 3), label, 11)

    def front_ribbon(
        self, source: dict[str, Any], index: int
    ) -> list[np.ndarray] | None:
        camera = source["front_camera"]
        if camera is None:
            return None
        edges = future_edges(source, index)
        if len(edges) < 2:
            return None
        pose = source["frames"][index]["pose"]
        transform, radius = None, math.inf
        if not isinstance(camera, FThetaCameraModel):
            transform = np.linalg.inv(pose @ np.asarray(camera["sensor_to_ego_matrix"]))
            if "projection_radius" not in camera:
                camera["projection_radius"] = raw_camera_radius(camera)
            radius = camera["projection_radius"]

        def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if isinstance(camera, FThetaCameraModel):
                pixels, depth, _ = camera.project_world(points, pose)
                return pixels / [
                    camera.calibration.width,
                    camera.calibration.height,
                ], depth
            return project_raw_camera(points, pose, camera)

        polygons = []
        for first, second in pairwise(edges):
            quad = np.array([first[0], second[0], second[1], first[1]])
            _, depths = project(quad)
            quad = clip_polygon(quad, depths - 0.1)
            if len(quad) < 3:
                continue
            if transform is not None and math.isfinite(radius):
                camera_points = quad @ transform[:3, :3].T + transform[:3, 3]
                if np.any(
                    np.linalg.norm(camera_points[:, :2], axis=1)
                    > radius * math.cos(math.pi / 32) * camera_points[:, 2]
                ):
                    # ponytail: The inscribed 32-sided cone stays on the valid
                    # radial branch; exact conic clipping removes its <0.5% margin.
                    for angle in np.arange(32) * (2 * math.pi / 32):
                        distance = radius * math.cos(math.pi / 32) * camera_points[
                            :, 2
                        ] - camera_points[:, :2] @ np.array(
                            [math.cos(angle), math.sin(angle)]
                        )
                        quad = clip_polygon(quad, distance)
                        if len(quad) < 3:
                            break
                        camera_points = quad @ transform[:3, :3].T + transform[:3, 3]
                    if len(quad) < 3:
                        continue
            pixels, _ = project(quad)
            if not np.isfinite(pixels).all():
                continue
            for axis, bound, sign in ((0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, -1)):
                pixels = clip_polygon(pixels, (pixels[:, axis] - bound) * sign)
                if len(pixels) < 3:
                    break
            if len(pixels) >= 3:
                polygons.append(pixels)
        return polygons

    def map_view(
        self,
        source: dict[str, Any],
        index: int,
        size: tuple[int, int],
        accent: str,
        *,
        overhead: bool = False,
    ) -> Image.Image:
        width, height = size
        image = Image.new("RGB", size, PANEL)
        draw = ImageDraw.Draw(image)
        frame = source["frames"][index]
        pose = frame["pose"]
        origin = pose[:3, 3]
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0])

        def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            relative = np.asarray(points) - origin
            x, y, z = relative @ right, relative @ forward, relative[..., 2]
            if overhead:
                return np.stack(
                    (width / 2 + x * height / 52, height * 0.82 - y * height / 52),
                    axis=-1,
                ), np.ones(x.shape, dtype=bool)
            xy, depth = chase_projection(np.stack((x, y, z), axis=-1), size)
            return xy, depth > 1

        for segments, color in self.lines:
            middle = segments.mean(axis=1) - origin
            mask = np.linalg.norm(middle[:, :2], axis=1) < (65 if overhead else 95)
            selected = segments[mask]
            pixels, valid = project(selected)
            for segment in pixels[np.all(valid, axis=1)]:
                draw.line(
                    [tuple(point) for point in segment],
                    fill=color,
                    width=1 if overhead else 2,
                )
        edges = future_edges(source, index)
        if len(edges) > 1:
            perimeter = np.concatenate((edges[:, 0], edges[::-1, 1]))
            pixels, valid = project(perimeter)
            if valid.all():
                overlay = Image.new("RGBA", size)
                tint = tuple(int(accent[i : i + 2], 16) for i in (1, 3, 5))
                ImageDraw.Draw(overlay).polygon(
                    [tuple(p) for p in pixels], fill=(*tint, 55), outline=(*tint, 155)
                )
                image = Image.alpha_composite(image.convert("RGBA"), overlay).convert(
                    "RGB"
                )
                draw = ImageDraw.Draw(image)

        def box(
            center: np.ndarray, scale: np.ndarray, heading: float, ego: bool = False
        ) -> None:
            axis = np.array([math.cos(heading), math.sin(heading), 0])
            side = np.array([-math.sin(heading), math.cos(heading), 0])
            base = np.array(
                [
                    center + axis * a * scale[0] / 2 + side * b * scale[1] / 2
                    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
            vertices = np.concatenate((base, base + [0, 0, scale[2]]))
            pixels, valid = project(vertices)
            if not valid.all():
                return
            colors = (
                ("#bccbd2", "#e8eff1", "#658e9c")
                if ego
                else ("#2164bb", "#66b9df", "#16448c")
            )
            faces = (
                ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7))
                if not overhead
                else ((4, 5, 6, 7),)
            )
            camera_position = origin - forward * 14 + [0, 0, 12]
            for face in faces:
                corners = vertices[list(face)]
                normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                if not overhead and normal @ (camera_position - corners[0]) <= 0:
                    continue
                color = colors[2 if face[0] == 4 else face[0] % 2]
                draw.polygon(
                    [tuple(pixels[i]) for i in face],
                    fill=color,
                    outline="#b9d5e0" if ego else "#76b7e6",
                )
            if ego:
                roof = pixels[4:] * 0.65 + pixels[4:].mean(axis=0) * 0.35
                draw.polygon([tuple(p) for p in roof], fill="#183c4a", outline=accent)

        actors = []
        actor_tracks = self.tracks
        if frame.get("actors"):
            actor_tracks = [
                {
                    "timestamps_us": [
                        frame.get("source_timestamp_us", frame["timestamp_us"])
                    ],
                    "translations": [actor["translation_world"]],
                    "quaternions_xyzw": [actor["orientation_xyzw"]],
                    "scale": actor["dimensions_lwh"],
                }
                for actor in frame["actors"]
            ]
        for track in actor_tracks:
            times = np.asarray(track["timestamps_us"], dtype=np.int64)
            timestamp = frame.get("source_timestamp_us", frame["timestamp_us"])
            if not times[0] - 50_000 <= timestamp <= times[-1] + 50_000:
                continue
            i = nearest(times, timestamp)
            center = np.asarray(track["translations"][i], dtype=float)
            if np.linalg.norm(center[:2] - origin[:2]) > 65:
                continue
            qx, qy, qz, qw = track["quaternions_xyzw"][i]
            heading = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            scale = np.asarray(track["scale"], dtype=float)
            actors.append((float((center - origin) @ forward), center, scale, heading))
        for _, center, scale, heading in sorted(
            actors, key=lambda actor: actor[0], reverse=True
        ):
            box(center - [0, 0, scale[2] / 2], scale, heading)
        if self.ego_sprite is not None and not overhead:
            scale = width / self.ego_sprite.width
            sprite = self.ego_sprite.resize(
                (width, round(self.ego_sprite.height * scale)), Image.Resampling.LANCZOS
            )
            image.paste(sprite, (0, round((height - sprite.height) * 0.6)), sprite)
        else:
            box(
                origin - [0, 0, source["floor_offset"]],
                np.array([4.8, 2.0, 1.6]),
                yaw,
                True,
            )
        if overhead:
            draw.rectangle((0, 0, width, 22), fill=BG)
            self.text(draw, (10, 4), "LOCAL MAP", 12, MUTED)
            self.text(draw, (width - 43, 4), "50 m", 12, MUTED)
        else:
            draw.rounded_rectangle((12, 12, 148, 36), radius=4, fill=BG)
            self.text(
                draw,
                (22, 17),
                "HUMAN DRIVER" if source.get("estimated_controls") else "SIMULATION",
                12,
                MUTED,
            )
            draw.rounded_rectangle(
                (12, height - 42, 342, height - 12), radius=4, fill=BG
            )
            draw.rectangle((24, height - 31, 52, height - 22), fill=accent)
            self.text(
                draw,
                (62, height - 36),
                "RECORDED PATH"
                if source.get("estimated_controls")
                else "SIMULATED PATH",
                14,
            )
            self.text(draw, (236, height - 34), "NEXT 4 s", 12, MUTED)
        return image

    def instruments(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        frame: dict[str, Any],
        accent: str,
    ) -> None:
        draw.rectangle((x, y, x + 292, y + 208), fill=PANEL, outline=LINE)
        self.text(draw, (x + 12, y + 10), "SPEED", 12, MUTED)
        cx, cy, radius = x + 80, y + 88, 58
        bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.arc(bounds, 140, 400, fill=LINE, width=4)
        draw.arc(
            bounds,
            140,
            140 + min(abs(frame["speed"]) * 3.6 / self.speed_scale, 1) * 260,
            fill=accent,
            width=4,
        )
        for i in range(13):
            a = math.radians(140 + i * 260 / 12)
            outer, inner = radius, radius - (7 if i % 4 == 0 else 4)
            draw.line(
                (
                    cx + inner * math.cos(a),
                    cy + inner * math.sin(a),
                    cx + outer * math.cos(a),
                    cy + outer * math.sin(a),
                ),
                fill=MUTED,
                width=1,
            )
            if i % 4 == 0:
                draw.text(
                    (
                        cx + (radius + 12) * math.cos(a),
                        cy + (radius + 12) * math.sin(a),
                    ),
                    f"{self.speed_scale * i / 12:.0f}",
                    font=FONTS[12],
                    fill=MUTED,
                    anchor="mm",
                )
        value = f"{abs(frame['speed']) * 3.6:.0f}"
        draw.text(
            (cx, cy - 2),
            value,
            font=FONTS[38 if len(value) < 3 else 30],
            fill=TEXT,
            anchor="mm",
        )
        draw.text((cx, cy + 28), "km/h", font=FONTS[12], fill=MUTED, anchor="mm")
        self.text(draw, (x + 176, y + 10), "STEERING", 12, MUTED)
        cx, cy, radius = x + 228, y + 78, 28
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=accent,
            width=3,
        )
        # ponytail: A fixed presentation ratio approximates the steering wheel;
        # replace it with measured steering-wheel telemetry when available.
        angle = frame["steer"] * self.steering_ratio
        for delta in (-150, -30, 90):
            a = math.radians(delta) - angle
            draw.line(
                (cx, cy, cx + radius * math.cos(a), cy + radius * math.sin(a)),
                fill=accent,
                width=2,
            )
        draw.ellipse(
            (cx - 6, cy - 6, cx + 6, cy + 6), fill=PANEL, outline=accent, width=2
        )
        direction = "L" if angle > 0 else "R"
        value = (
            f"{direction} ≈{abs(math.degrees(angle)):.1f}°"
            if abs(angle) > 0.0001
            else "0.0°"
        )
        draw.text((cx, y + 122), value, font=FONTS[16], fill=accent, anchor="mm")
        draw.line((x + 12, y + 145, x + 280, y + 145), fill=LINE)
        for i, key in enumerate(("throttle", "brake")):
            yy = y + 158 + i * 24
            color = accent if frame[key] > 0.001 else MUTED
            iw, ih = (9, 15) if i == 0 else (14, 10)
            draw.rounded_rectangle(
                (x + 13, yy - 2, x + 13 + iw, yy - 2 + ih),
                radius=2,
                outline=color,
                width=2,
            )
            for offset in (3, 6, 9):
                if i == 0:
                    draw.line((x + 15, yy + offset, x + 20, yy + offset), fill=color)
                else:
                    draw.line(
                        (x + 13 + offset, yy, x + 13 + offset, yy + 6), fill=color
                    )
            self.text(draw, (x + 34, yy - 1), key.upper(), 12, MUTED)
            draw.rounded_rectangle(
                (x + 112, yy + 4, x + 278, yy + 9), radius=2, fill=LINE
            )
            if frame[key] > 0.001:
                draw.rounded_rectangle(
                    (x + 112, yy + 4, x + 112 + min(1, frame[key]) * 166, yy + 9),
                    radius=2,
                    fill=accent,
                )

    def chart(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        key: str,
        title: str,
        timestamp: int,
    ) -> None:
        x, y, width, height = box
        self.text(draw, (x + 6, y), title, 12, MUTED)
        plot = x + 38, y + 30, width - 56, height - 58
        px, py, pw, ph = plot
        values = [
            frame[key] * (3.6 if key == "speed" else 1)
            for source in (self.left, self.right)
            for frame in source["frames"]
        ]
        high = (
            self.speed_scale
            if key == "speed"
            else max(3, math.ceil(max(abs(v) for v in values) / 3) * 3)
        )
        low = 0 if key == "speed" else -high
        draw.rectangle((x, py - 8, x + width, py + ph + 25), outline=LINE)
        for value in np.linspace(high, low, 3):
            yy = py + (high - value) / (high - low) * ph
            if value == 0:
                draw.line((px, yy, px + pw, yy), fill=LINE)
            self.text(
                draw,
                (x + 5, yy - 7),
                f"{value:+.0f}" if key != "speed" else f"{value:.0f}",
                12,
                MUTED,
            )
        duration = (self.end - self.start) / 1e6
        for elapsed in np.arange(
            0, duration + 0.001, max(1, 2 * math.ceil(duration / 12))
        ):
            xx = px + elapsed / max(duration, 0.1) * pw
            draw.line((xx, py + ph + 4, xx, py + ph + 8), fill=LINE)
            draw.text(
                (xx, py + ph + 15),
                f"{elapsed:g}s",
                font=FONTS[12],
                fill=MUTED,
                anchor="mm",
            )
        for source, accent in (
            (self.left, self.template["left_accent"]),
            (self.right, self.template["right_accent"]),
        ):
            full_points = []
            for frame in source["frames"]:
                t = frame["timestamp_us"]
                if self.start <= t <= self.end:
                    value = frame[key] * (3.6 if key == "speed" else 1)
                    point = (
                        px + (t - self.start) / max(1, self.end - self.start) * pw,
                        py + (high - value) / (high - low) * ph,
                    )
                    full_points.append(point)
            if len(full_points) > 1:
                draw.line(full_points, fill=accent, width=3)
                frame = source["frames"][nearest(source["times"], timestamp)]
                xx = px + (timestamp - self.start) / max(1, self.end - self.start) * pw
                value = frame[key] * (3.6 if key == "speed" else 1)
                yy = py + (high - value) / (high - low) * ph
                draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=accent)
        cursor = px + (timestamp - self.start) / max(1, self.end - self.start) * pw
        draw.line((cursor, py, cursor, py + ph), fill=TEXT)
        draw.polygon(
            ((cursor - 4, py - 6), (cursor + 4, py - 6), (cursor, py)), fill=TEXT
        )
        draw.text(
            (cursor, py - 14),
            f"{(timestamp - self.start) / 1e6:.1f}s",
            font=FONTS[12],
            fill=MUTED,
            anchor="mm",
        )

    def evaluation(self, index: int) -> list[tuple[str, list[tuple[str, str | None]]]]:
        """Summarize recorded motion; leave checks requiring missing evidence unset."""
        frames = self.right["frames"]
        frame = frames[index]
        segments = np.diff(self.reference_path, axis=0)
        relative = frame["pose"][:2, 3] - self.reference_path[:-1]
        weight = np.clip(
            np.sum(relative * segments, axis=1)
            / np.maximum(np.sum(segments**2, axis=1), 1e-12),
            0,
            1,
        )
        offset = (
            float(np.linalg.norm(relative - segments * weight[:, None], axis=1).min())
            if len(segments)
            else float(np.linalg.norm(frame["pose"][:2, 3] - self.reference_path[0]))
        )
        first_idle = index
        while (
            first_idle > 0
            and abs(frames[first_idle - 1]["speed"]) < 0.3
            and frames[first_idle - 1]["throttle"] > 0.2
        ):
            first_idle -= 1
        idle = (
            (frame["timestamp_us"] - frames[first_idle]["timestamp_us"]) / 1e6
            if abs(frame["speed"]) < 0.3 and frame["throttle"] > 0.2
            else 0
        )
        previous = frames[max(0, index - 1)]
        dt = (frame["timestamp_us"] - previous["timestamp_us"]) / 1e6
        step = np.linalg.norm(frame["pose"][:2, 3] - previous["pose"][:2, 3])
        jump = abs(step - (abs(frame["speed"]) + abs(previous["speed"])) * dt / 2)

        def check(label: str, value: float, key: str) -> tuple[str, str]:
            limit = self.comfort_limits[key]
            return (
                label,
                "fail" if value > limit else "warn" if value > limit * 0.8 else "pass",
            )

        # ponytail: These are explicit presentation thresholds, not a certified
        # evaluator; map rules, occupancy and planner intent need recorded results.
        return [
            (
                "SAFETY",
                [
                    ("Collision Risk", None),
                    ("Curb / Barrier", None),
                    ("OCC Collision", None),
                ],
            ),
            (
                "COMPLIANCE",
                [
                    ("Solid-line Crossing", None),
                    ("Wrong-way", None),
                    ("Overspeed", None),
                    ("Traffic Light", None),
                ],
            ),
            (
                "EFFICIENCY",
                [
                    ("Follow-too-far", None),
                    ("Stuck", "fail" if idle >= 5 else "pass"),
                    (f"Travel {frame['travel']:.1f} m", "value"),
                    ("Lane Change", None),
                ],
            ),
            ("NAVIGATION", [(f"GT Path Offset {offset:.2f} m", "value")]),
            (
                "COMFORT",
                [
                    check("Path Jump", float(jump), "path_jump"),
                    check(
                        "Harsh Acceleration",
                        max(0, frame["acceleration"]),
                        "acceleration",
                    ),
                    check("Emergency Brake", max(0, -frame["acceleration"]), "braking"),
                    check("Cruise Jerk", abs(frame["jerk"]), "jerk"),
                    check(
                        "Lateral Accel",
                        abs(frame["lateral_acceleration"]),
                        "lateral_acceleration",
                    ),
                    ("Weaving", None),
                    ("Path Oscillation", None),
                    ("Phantom Brake", None),
                ],
            ),
        ]

    def evaluation_panel(self, draw: ImageDraw.ImageDraw, index: int) -> None:
        groups = self.evaluation(index)
        colors = {
            None: (MUTED, "#131e26", LINE),
            "value": ("#39c7aa", "#102923", "#20493f"),
            "pass": ("#39c7aa", "#102923", "#20493f"),
            "warn": ("#efb542", "#34270f", "#67501f"),
            "fail": ("#f07868", "#351915", "#673029"),
        }
        self.panel(draw, (958, 776, 1886, 1244))
        self.text(draw, (970, 786), "EVALUATION", 20)
        states = [state for _, badges in groups for _, state in badges]
        for i, (state, suffix) in enumerate(
            (("pass", "passed"), ("warn", "warned"), ("fail", "failed"))
        ):
            color, fill, outline = colors[state]
            x = 1520 + i * 118
            draw.rounded_rectangle(
                (x, 785, x + 110, 809), radius=9, fill=fill, outline=outline
            )
            draw.text(
                (x + 55, 797),
                f"{states.count(state)} {suffix}",
                font=FONTS[12],
                fill=color,
                anchor="mm",
            )
        for (name, badges), (y, height) in zip(
            groups,
            ((824, 72), (902, 72), (980, 72), (1058, 62), (1126, 108)),
            strict=True,
        ):
            draw.rounded_rectangle(
                (970, y, 1874, y + height), radius=6, fill=BG, outline=LINE
            )
            status = (
                "fail"
                if any(s == "fail" for _, s in badges)
                else "warn"
                if any(s == "warn" for _, s in badges)
                else "pass"
                if any(s in ("pass", "value") for _, s in badges)
                else None
            )
            draw.line(
                (971, y + 7, 971, y + height - 7), fill=colors[status][0], width=3
            )
            self.text(draw, (982, y + 6), name, 18)
            x, yy = 982, y + 34
            for label, state in badges:
                width = round(FONTS[16].getlength(label)) + 38
                if x + width > 1862:
                    x, yy = 982, yy + 33
                color, fill, outline = colors[state]
                draw.rounded_rectangle(
                    (x, yy, x + width, yy + 28), radius=14, fill=fill, outline=outline
                )
                draw.ellipse(
                    (x + 9, yy + 10, x + 17, yy + 18),
                    fill=color if state is not None else None,
                    outline=color,
                )
                self.text(draw, (x + 24, yy + 4), label, 16, color)
                x += width + 8

    def render(self, timestamp: int) -> Image.Image:
        """Draw one deterministic presentation frame at a source timestamp."""
        canvas = Image.new("RGB", (1920, 1280), BG)
        draw = ImageDraw.Draw(canvas)
        indices = [
            nearest(source["times"], timestamp) for source in (self.left, self.right)
        ]
        current = []
        for side, source in enumerate((self.left, self.right)):
            x = 24 + side * 946
            index = indices[side]
            frame = source["frames"][index]
            current.append(frame)
            accent = (
                self.template["left_accent"]
                if side == 0
                else self.template["right_accent"]
            )
            title = (
                self.template["left_title"]
                if side == 0
                else self.template["right_title"]
            )
            draw.rounded_rectangle((x, 14, x + 24, 38), radius=5, outline=accent)
            self.text(draw, (x + 5, 17), "//", 14, accent)
            self.fitted_text(draw, (x + 34, 10), title, 890)
            draw.line((x, 46, x + 924, 46), fill=LINE)
            draw.line((x, 46, x + 92, 46), fill=accent, width=2)
            self.camera(
                canvas,
                (x, 62, 534, 286),
                frame["images"].get("CAM_FRONT"),
                "FRONT WIDE",
                accent,
                ribbon=self.front_ribbon(source, index),
            )
            auxiliary = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_FRONT_TELE",
            ]
            for i, camera in enumerate(auxiliary):
                xx, yy = x + 546 + (i % 2) * 189, 62 + (i // 2) * 96
                derived = (
                    camera == "CAM_FRONT_TELE"
                    and camera not in frame["images"]
                    and "CAM_FRONT" in frame["images"]
                )
                if camera in frame["images"] or derived:
                    self.camera(
                        canvas,
                        (xx, yy, 187, 94),
                        frame["images"]["CAM_FRONT" if derived else camera],
                        "FRONT TELE · CROP" if derived else LABELS[camera],
                        accent,
                        crop=derived,
                        zoom=2 if derived else 1,
                    )
            self.text(draw, (x, 351), "DRIVING SCENE", 12, MUTED)
            self.text(draw, (x + 480, 351), "CHASE CAMERA", 12, MUTED)
            canvas.paste(self.map_view(source, index, (632, 350), accent), (x, 368))
            canvas.paste(
                self.map_view(source, index, (292, 142), accent, overhead=True),
                (x + 632, 368),
            )
            self.instruments(draw, x + 632, 510, frame, accent)
            draw.rectangle((x, 368, x + 924, 718), outline=LINE)
            draw.line((x + 632, 368, x + 632, 718), fill=LINE)
            for xx, yy, dx, dy in (
                (x, 368, 1, 1),
                (x + 924, 368, -1, 1),
                (x, 718, 1, -1),
                (x + 924, 718, -1, -1),
            ):
                draw.line(
                    (xx, yy + dy * 16, xx, yy, xx + dx * 16, yy), fill=accent, width=2
                )
        draw.rounded_rectangle((24, 730, 1896, 1256), radius=9, outline="#3a515f")
        self.text(draw, (38, 738), "CLOSED-LOOP ANALYSIS", 26)
        caption = str(self.template.get("caption", "")).strip()
        if caption:
            self.fitted_text(draw, (430, 747), caption, 650, 14, MUTED)
        deviation = float(
            np.linalg.norm(current[0]["pose"][:2, 3] - current[1]["pose"][:2, 3])
        )
        self.text(
            draw,
            (1310, 746),
            f"t = {(timestamp - self.start) / 1e6:.1f}s    frame {round((timestamp - self.start) / 1e6 * int(self.template.get('fps', 10)))}",
            14,
            MUTED,
        )
        draw.text(
            (1876, 752),
            f"DIVERGENCE  {deviation:.2f} m",
            font=FONTS[14],
            fill=MUTED,
            anchor="rm",
        )
        self.panel(draw, (34, 776, 946, 1244))
        self.text(draw, (46, 786), "MONITORING", 20)
        for i, (label, color) in enumerate(
            (
                ("Human driver", self.template["left_accent"]),
                (
                    "Simulation" if self.right is not self.left else "GT replay",
                    self.template["right_accent"],
                ),
            )
        ):
            x = 432 + i * 190
            draw.line((x, 800, x + 30, 800), fill=color, width=3)
            self.text(draw, (x + 38, 792), label, 12, MUTED)
        self.chart(
            draw,
            (46, 824, 888, 229),
            "acceleration",
            "LONGITUDINAL ACCEL  m/s²",
            timestamp,
        )
        self.chart(draw, (46, 1060, 888, 173), "speed", "SPEED  km/h", timestamp)
        self.evaluation_panel(draw, indices[1])
        return canvas


def main() -> None:
    """Render a GT test or compare GT with an exported simulation directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--ego-model", type=Path)
    args = parser.parse_args()
    template = json.loads(args.template.read_text())
    fps = int(template.get("fps", 10))
    if not 1 <= fps <= 60 or args.frames is not None and args.frames < 1:
        parser.error("FPS must be 1..60 and frame limit must be positive")
    left = load_source(args.gt, args.scene)
    right = load_source(args.right, args.scene) if args.right else left
    renderer = ComparisonRenderer(left, right, args.scene, template, args.ego_model)
    count = (renderer.end - renderer.start) * fps // 1_000_000 + 1
    if args.frames is not None:
        count = min(count, args.frames)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from flashdreams.runtime_v2.video_encoder import Mp4Encoder

    temporary = args.output.with_suffix(".partial.mp4")
    if temporary.exists():
        raise FileExistsError(temporary)
    encoder = Mp4Encoder(temporary, width=1920, height=1280, frames_per_second=fps)
    try:
        for i in range(count):
            image = renderer.render(int(renderer.start + round(i * 1e6 / fps)))
            if i in (0, min(round(fps * 4.5), count // 2)):
                image.save(args.output.with_suffix(".jpg"), quality=92)
            encoder.write(np.asarray(image)[None])
            print(json.dumps({"frame": i + 1, "frames": count}), flush=True)
    finally:
        encoder.close()
    temporary.replace(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "template": template,
                "gt": str(args.gt.resolve()),
                "right": str(args.right.resolve()) if args.right else "GT duplicate",
                "ego_model": str(args.ego_model.resolve()) if args.ego_model else None,
                "reference_speed_estimation": left["speed_estimation"],
                "steering_wheel": {
                    "ratio": renderer.steering_ratio,
                    "method": "front-wheel angle multiplied by a configurable fixed ratio; approximate, not Haval calibration",
                    "positive_direction": "left; wheel rotates counterclockwise",
                },
                "evaluation": {
                    "method": "Motion display heuristics, not certified safety evaluation; gray checks have no recorded evidence",
                    "comfort_limits": renderer.comfort_limits,
                    "comfort_units": "m/s² for acceleration, braking and lateral acceleration; m/s³ for jerk; m for path jump residual",
                    "warning_fraction": 0.8,
                    "stuck": "Speed below 0.3 m/s with throttle above 0.2 for at least 5 seconds",
                    "gt_path_offset": "Nearest distance to the reference XY polyline, not a navigation route compliance check",
                    "trajectory": "Observed next four seconds; not a planner prediction",
                },
                "frames": count,
                "fps": fps,
                "width": 1920,
                "height": 1280,
                "start_timestamp_us": str(renderer.left["time_origin_us"]),
                "time_alignment": "Each side starts at its first moving frame; original timestamps remain in the source recording",
                "right_start_timestamp_us": str(renderer.right["time_origin_us"]),
                "trimmed_stationary_start_s": {
                    "left": renderer.left["trimmed_start_s"],
                    "right": renderer.right["trimmed_start_s"],
                },
                "start_speed_threshold_mps": renderer.start_speed_mps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
