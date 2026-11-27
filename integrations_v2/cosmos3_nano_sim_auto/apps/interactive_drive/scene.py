# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load prepared scene bundles into Interactive Drive's scene contract.

The bundle supplies the model's palette, geometry, calibration and opening RGB.
Explicit camera-to-vehicle calibration defines the rig frame when available;
otherwise the anchor camera does. Source conversion happens offline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from interactive_drive.config import AppConfig, RasterConfig
from interactive_drive.math3d import euler_xyz_degrees_to_matrix, transform_from_rt
from interactive_drive.types import CameraCalibration, SceneBundle

from cosmos3_nano_sim_auto.impl.conditioning import camera_rig
from cosmos3_nano_sim_auto.impl.conditioning.controls import loaded_control_clips
from cosmos3_nano_sim_auto.impl.conditioning.driving import recorded_speed
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import prepared_scene
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.camera import (
    RDF_TO_FLU,
    frame_rate,
    smoothable,
)
from cosmos3_nano_sim_auto.inputs import MultiviewSample

__all__ = ["load_scene_bundle"]

_FLU_FROM_RECORDED = np.linalg.inv(RDF_TO_FLU)


def load_scene_bundle(
    scene_path: Path,
    camera_names: Sequence[str],
    variant: str,
    prompt_override: str | None,
    raster: RasterConfig,
    *,
    config: AppConfig | None = None,
    road_height_primitives: Sequence[int] = (),
) -> SceneBundle:
    """Load a prepared scene with calibrated cameras and matching opening RGB."""
    if not camera_names:
        raise ValueError("A scene needs at least one camera.")
    if not (scene_path / "scene.json").is_file():
        raise ValueError(
            "Expected an extracted prepared scene bundle containing scene.json; "
            "pass its directory with --scene."
        )
    return _prepared_bundle(
        scene_path, camera_names, prompt_override, raster, config, road_height_primitives
    )


class _RoadMarkingElevation:
    """Fit local ground height only where surrounding road markings support one surface."""

    def __init__(self, points: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        self.points = np.unique(
            np.asarray(points, dtype=np.float64).reshape(-1, 3), axis=0
        )
        self.tree = cKDTree(self.points[:, :2])

    def __call__(self, x: float, y: float) -> float | None:
        # ponytail: infer the local surface within 6 m of surveyed markings;
        # use a registered ground mesh where available for gaps and multiple decks.
        nearby = self.points[self.tree.query_ball_point((x, y), 6.0)]
        if len(nearby) < 3:
            return None
        xy = nearby[:, :2] - (x, y)
        angles = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
        if np.diff(angles, append=angles[0] + 2 * np.pi).max() >= np.pi:
            return None  # The query is outside the local convex hull.
        design = np.column_stack((xy, np.ones(len(xy))))
        plane, _, rank, singular = np.linalg.lstsq(design, nearby[:, 2], rcond=None)
        if (
            rank < 3
            or singular[-1] < 0.25 * np.sqrt(len(xy))
            or np.hypot(*plane[:2]) > np.tan(np.deg2rad(10))
            or np.abs(design @ plane - nearby[:, 2]).max() > 0.15
        ):
            return None  # Nearly collinear, too steep, or incompatible elevations.
        return float(plane[2])


def _calibration(
    clip: Path,
    name: str,
    track: np.ndarray,
    rig_to_world: np.ndarray,
    opening: int,
    intrinsics: np.ndarray,
) -> CameraCalibration:
    """One camera, as the app's calibration carries it.

    The fourteen numbers a rig records per sensor are exactly the fields of
    ``CameraCalibration``, so the lens needs no translation. The placement does:
    the app wants a sensor-to-rig in its forward-left-up frame, and a clip
    records a pose per camera in the renderer's right-down-front one.
    """
    intrinsics = np.asarray(
        intrinsics,
        dtype=np.float64,
    ).reshape(-1)
    if intrinsics.size < 11:
        raise ValueError(
            f"camera {name!r} records {intrinsics.size} intrinsics; at least 11 are "
            "needed for the optical centre, the size, the polynomial and its flag."
        )
    distortion = (
        intrinsics[11:14] if intrinsics.size >= 14 else np.array([1.0, 0.0, 0.0])
    )
    sensor_to_rig = np.linalg.inv(rig_to_world) @ track[opening] @ _FLU_FROM_RECORDED
    return CameraCalibration(
        clipgt_name=name,
        logical_name=name,
        width=int(intrinsics[2]),
        height=int(intrinsics[3]),
        cx=float(intrinsics[0]),
        cy=float(intrinsics[1]),
        polynomial=intrinsics[4:10].astype(np.float32),
        is_backward_polynomial=bool(intrinsics[10] > 0),
        linear_cde=np.asarray(distortion, dtype=np.float32),
        sensor_to_rig_flu=sensor_to_rig.astype(np.float32),
        ludus_draw_size=smoothable((int(intrinsics[2]), int(intrinsics[3]))),
    )


def _prepared_bundle(
    root: Path,
    cameras: Sequence[str],
    prompt_override: str | None,
    raster: RasterConfig,
    config: AppConfig | None,
    road_height_primitives: Sequence[int] = (),
) -> SceneBundle:
    """Open the generated RGB set at its recorded scene state."""
    camera_rig.view_ids(cameras)
    prepared = prepared_scene.load(root)
    if any(name not in prepared.tracks for name in cameras):
        raise ValueError("Requested camera is absent from the prepared scene")
    sampled = 0 if config is None else config.start_frame
    posed = int(prepared.frame_mapping(sampled, None)[0])
    manifest = prepared.manifest
    fps = frame_rate(
        manifest["frames"], int(manifest["start_us"]), int(manifest["end_us"])
    )
    if config is not None and not np.isclose(config.chunk.fps, fps, rtol=0, atol=1e-3):
        raise ValueError(f"Prepared scene requires {fps:.6f} FPS")
    sample = MultiviewSample.load(root / "sample.json")
    anchor = prepared.tracks[cameras[0]]
    rig_track = anchor @ _FLU_FROM_RECORDED
    mount = (
        manifest["cameras"][cameras[0]]
        .get("virtual_camera", {})
        .get("nominalSensor2Rig_FLU")
    )
    if mount is not None:
        sensor_to_rig = transform_from_rt(
            euler_xyz_degrees_to_matrix(mount["roll-pitch-yaw"]), mount["t"]
        )
        if not np.isfinite(sensor_to_rig).all():
            raise ValueError("Camera-to-vehicle calibration must be finite")
        rig_track = rig_track @ np.linalg.inv(sensor_to_rig)
    rig_to_world = rig_track[posed]
    elevation_at = None
    ground_metadata = {}
    if road_height_primitives:
        registration = manifest.get("provenance", {}).get("height_registration", {})
        if (
            registration.get("mode") != "per_frame_vertical_display_translation"
            or not registration.get("dem_sha256")
        ):
            raise ValueError("Road-marking height requires a DEM-registered prepared map")
        pools = {
            pool.prim_type_id: pool.vertices.numpy().reshape(-1, 3)
            for pool in prepared.scene.polyline_pools
            if pool.prim_type_id in road_height_primitives
        }
        if set(road_height_primitives) != pools.keys():
            raise ValueError("Requested road height primitives are absent from the map")
        elevation_at = _RoadMarkingElevation(np.concatenate(list(pools.values())))
        opening_ground = elevation_at(*rig_to_world[:2, 3])
        if opening_ground is None:
            raise ValueError("Road markings do not establish ground at the opening pose")
        ground_metadata["rig_ground_clearance_m"] = (
            float(rig_to_world[2, 3]) - opening_ground
        )
    return SceneBundle(
        scene_path=root,
        scene_id=sample.name,
        metadata={
            **ground_metadata,
            "map": str(root),
            "sample": str(root),
            "start_frame": 0,
            "pose_frame": posed,
            "source_fps": fps,
            "prepared_scene": prepared,
            "scene_geometry": {
                "root": str(prepared.root),
                "manifest_sha256": prepared_scene.sha256(prepared.root / "scene.json"),
            },
        },
        cameras=tuple(
            _calibration(
                root,
                name,
                prepared.tracks[name],
                rig_to_world,
                posed,
                prepared.intrinsics(name),
            )
            for name in cameras
        ),
        initial_rig_to_world=rig_to_world.astype(np.float32),
        initial_timestamp_us=int(manifest["start_us"]) + round(posed * 1e6 / fps),
        initial_yaw_rad=_yaw_of(rig_to_world),
        initial_speed_mps=recorded_speed(
            rig_track, fps, first=posed, over=max(1, round(fps))
        ),
        initial_rgbs=_opening_frames(sample, cameras, start_frame=0, raster=raster),
        prompt=("" if sample.view_captions else sample.prompt)
        if prompt_override is None
        else prompt_override,
        line_layers=(),
        triangle_layers=(),
        road_elevation_at=elevation_at,
    )


def _opening_frames(
    sample: MultiviewSample,
    camera_names: Sequence[str],
    *,
    start_frame: int,
    raster: RasterConfig,
) -> tuple[Any, ...]:
    """The frame each camera opens on, as the app holds images.

    One pixel frame a camera, because one latent conditioning frame is one pixel
    frame. The app presents these as well as conditioning on them, so they come
    back as height-width-channel bytes rather than as the encoder's range.
    """
    if not sample.image_paths:
        raise ValueError(
            f"{sample.name!r} carries no image_path, so there is no footage for the "
            "frames the drive opens on."
        )
    wanted = [sample.cameras.index(name) for name in camera_names]
    clips = loaded_control_clips(
        [sample.image_paths[view] for view in wanted],
        pixel_frames=1,
        pixel_height=raster.height,
        pixel_width=raster.width,
        start_frame=start_frame,
    )
    return tuple(_as_bytes(clip[0]) for clip in clips)


def _as_bytes(frame: Any) -> Any:
    """``[3, H, W]`` in the encoder's range to ``[H, W, 3]`` bytes."""
    pixels = frame.permute(1, 2, 0).add(1.0).mul(127.5).round().clamp(0, 255)
    return pixels.to("cpu").numpy().astype(np.uint8)


def _yaw_of(rig_to_world: np.ndarray) -> float:
    """Where the car points, as the app's vehicle state counts it.

    The rig is forward-left-up, so its first column is where forward goes, and
    the app's yaw is that bearing in the world's ground plane.
    """
    forward = rig_to_world[:3, 0]
    return float(math.atan2(float(forward[1]), float(forward[0])))
