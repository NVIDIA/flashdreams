# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load prepared scene bundles into Interactive Drive's scene contract.

The bundle supplies the model's palette, geometry, calibration and opening RGB.
The anchor camera defines the rig frame, with other camera offsets derived from
the recorded tracks. Source conversion and image generation happen offline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from interactive_drive.config import AppConfig, RasterConfig
from interactive_drive.types import CameraCalibration, SceneBundle

from cosmos3_nano_sim_auto.impl.conditioning import camera_rig
from cosmos3_nano_sim_auto.impl.conditioning.controls import loaded_control_clips
from cosmos3_nano_sim_auto.impl.conditioning.driving import recorded_speed
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import prepared_scene
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.camera import (
    RDF_TO_FLU,
    frame_rate,
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
) -> SceneBundle:
    """Load a prepared scene with calibrated cameras and matching opening RGB."""
    if not camera_names:
        raise ValueError("A scene needs at least one camera.")
    if not (scene_path / "scene.json").is_file():
        raise ValueError(
            "Expected an extracted prepared scene bundle containing scene.json; "
            "pass its directory with --scene."
        )
    return _prepared_bundle(scene_path, camera_names, prompt_override, raster, config)


def _calibration(
    clip: Path,
    name: str,
    track: np.ndarray,
    anchor: np.ndarray,
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
    sensor_to_rig = np.linalg.inv(anchor[opening]) @ track[opening]
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
        sensor_to_rig_flu=(RDF_TO_FLU @ sensor_to_rig @ _FLU_FROM_RECORDED).astype(
            np.float32
        ),
    )


def _prepared_bundle(
    root: Path,
    cameras: Sequence[str],
    prompt_override: str | None,
    raster: RasterConfig,
    config: AppConfig | None,
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
    rig_to_world = anchor[posed] @ _FLU_FROM_RECORDED
    return SceneBundle(
        scene_path=root,
        scene_id=sample.name,
        metadata={
            "map": str(root),
            "sample": str(root),
            "start_frame": 0,
            "pose_frame": posed,
            "source_fps": fps,
            "prepared_scene": prepared,
        },
        cameras=tuple(
            _calibration(
                root,
                name,
                prepared.tracks[name],
                anchor,
                posed,
                prepared.intrinsics(name),
            )
            for name in cameras
        ),
        initial_rig_to_world=rig_to_world.astype(np.float32),
        initial_timestamp_us=int(manifest["start_us"]) + round(posed * 1e6 / fps),
        initial_yaw_rad=_yaw_of(rig_to_world),
        initial_speed_mps=recorded_speed(anchor, fps, first=posed),
        initial_rgbs=_opening_frames(sample, cameras, start_frame=0, raster=raster),
        prompt=("" if sample.view_captions else sample.prompt)
        if prompt_override is None
        else prompt_override,
        line_layers=(),
        triangle_layers=(),
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
