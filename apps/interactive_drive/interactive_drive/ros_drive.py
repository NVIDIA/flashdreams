# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

"""Map a simulator step and its native camera frames to the ROS socket protocol."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import AppConfig, VehicleConfig
from .ros_endpoint import DriveEndpoint
from .types import (
    MANUAL_STEER_RAD,
    DriverCommand,
    FrameChunk,
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)


@contextmanager
def report_model_errors(endpoint: DriveEndpoint) -> Iterator[None]:
    """Let the bridge discard an incomplete operation before this session closes."""
    try:
        yield
    except Exception as error:
        endpoint.send(
            {
                "type": "error",
                "code": "MODEL_ERROR",
                "message": str(error),
                "fatal": True,
            }
        )
        raise


def driver_command(request: dict, config: AppConfig) -> DriverCommand:
    """Translate a validated wire command using this vehicle's capabilities."""
    limit = config.vehicle.max_steer_rad
    angle = request["steering_angle_rad"]
    if "manual_gear" in request:
        if abs(angle) > MANUAL_STEER_RAD:
            raise ValueError(f"Manual steering exceeds {MANUAL_STEER_RAD} rad")
        return DriverCommand.from_manual_controls(
            steer=max(-limit, min(angle, limit)) / MANUAL_STEER_RAD,
            throttle=request["throttle"],
            brake=request["brake"],
            reverse=request["reverse"],
            gear=request["manual_gear"],
        )
    if abs(angle) > limit:
        raise ValueError(f"steering_angle_rad exceeds this vehicle's {limit} rad limit")
    reverse = request["reverse"]
    if reverse and not config.reverse:
        raise ValueError("The configured driving model does not support reverse")
    return DriverCommand(
        throttle=0.0 if request["brake"] > 0 else request["throttle"],
        brake=request["brake"],
        steer=angle / limit,
        steer_is_direct=True,
        reverse=reverse,
        gear="R" if reverse else "D",
    )


def publish_frame(
    endpoint: DriveEndpoint,
    episode: str,
    step: int,
    timestamp: int,
    pose: np.ndarray,
    vehicle: VehicleState,
    names: Sequence[str] = (),
    images: Sequence[Any] = (),
    *,
    calibration: list[dict] | None = None,
    scene_geometry: dict | None = None,
    scene_time_us: int | None = None,
    ego_dimensions_lwh: tuple[float, float, float] | None = None,
    rig_ground_clearance_m: float | None = None,
    condition: Any | None = None,
) -> None:
    header: dict[str, Any] = {
        "type": "frame",
        "episode_id": episode,
        "step": step,
        "timestamp_us": timestamp,
        "rig_to_world": pose.tolist(),
        "vehicle": asdict(vehicle),
        "images": [],
    }
    if calibration is not None:
        header["calibration"] = calibration
    if scene_geometry is not None:
        header["scene_geometry"] = scene_geometry
    if scene_time_us is not None:
        header["scene_time_us"] = scene_time_us
    if ego_dimensions_lwh is not None:
        header["ego_dimensions_lwh"] = ego_dimensions_lwh
    if rig_ground_clearance_m is not None:
        header["rig_ground_clearance_m"] = rig_ground_clearance_m
    buffers = []
    payloads = list(zip(names, images, strict=True))
    if condition is not None:
        payloads.append((f"condition/{names[0]}", condition))
    for name, pixels in payloads:
        array = np.asarray(pixels)
        if (
            array.ndim != 3
            or array.shape[2] not in (3, 4)
            or min(array.shape[:2]) <= 0
            or array.dtype != np.uint8
        ):
            raise ValueError(f"Expected a uint8 RGB or RGBA image for {name}")
        rgb = memoryview(np.ascontiguousarray(array[..., :3])).cast("B")
        height, width = array.shape[:2]
        header["images"].append(
            {"name": name, "width": width, "height": height, "nbytes": rgb.nbytes}
        )
        buffers.append(rgb)
    if condition is not None:
        header["images"][-1]["role"] = "condition"
    endpoint.send(header, buffers)


def publish_reset(
    endpoint: DriveEndpoint,
    episode: str,
    scene: SceneBundle,
    vehicle: VehicleState,
    vehicle_config: VehicleConfig | None = None,
) -> None:
    """Publish the source image and state without advancing simulation time."""
    names = [camera.clipgt_name for camera in scene.cameras]
    calibration = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(camera).items()
        }
        for camera in scene.cameras
    ]
    publish_initial_observation(
        endpoint,
        episode,
        int(scene.initial_timestamp_us),
        scene.initial_rig_to_world,
        vehicle,
        names,
        scene.initial_rgbs,
        calibration=calibration,
        scene_geometry=scene.metadata.get("scene_geometry"),
        rig_ground_clearance_m=scene.metadata.get("rig_ground_clearance_m"),
        ego_dimensions_lwh=(
            None
            if vehicle_config is None
            else (
                vehicle_config.aabb_length_m,
                vehicle_config.aabb_width_m,
                vehicle_config.aabb_height_m,
            )
        ),
    )


def publish_initial_observation(
    endpoint: DriveEndpoint,
    episode: str,
    t: int,
    pose: np.ndarray,
    vehicle: VehicleState,
    names: Sequence[str] = (),
    images: Sequence[Any] = (),
    *,
    calibration: Sequence[dict] = (),
    scene_geometry: dict | None = None,
    ego_dimensions_lwh: tuple[float, float, float] | None = None,
    rig_ground_clearance_m: float | None = None,
) -> None:
    publish_frame(
        endpoint,
        episode,
        -1,
        t,
        pose,
        vehicle,
        names,
        images,
        calibration=list(calibration),
        scene_geometry=scene_geometry,
        scene_time_us=t if scene_geometry is not None else None,
        ego_dimensions_lwh=ego_dimensions_lwh,
        rig_ground_clearance_m=rig_ground_clearance_m,
    )
    endpoint.send(
        {
            "type": "done",
            "episode_id": episode,
            "step": -1,
            "next_step": 0,
            "applied_from_us": t,
            "last_observation_us": t,
            "next_apply_us": t,
            "frame_count": 1,
            "cameras": names,
        }
    )


def publish_step(
    endpoint: DriveEndpoint,
    request: dict,
    scene: SceneBundle,
    chunk: FrameChunk,
    trajectory: TrajectoryChunk,
    next_apply_us: int,
    command: DriverCommand,
) -> None:
    """Join emitted images to pre-integration states on the scene timeline."""
    episode, step = request["episode_id"], request["expected_step"]
    names = [camera.clipgt_name for camera in scene.cameras]
    timestamps = trajectory.timestamps_us
    if tuple(frame.timestamp_us for frame in chunk.frames) != tuple(timestamps):
        raise ValueError("Lockstep requires every trajectory frame in timestamp order")
    emitted = 0
    for frame, pose, vehicle in zip(
        chunk.frames, trajectory.rig_poses_world, trajectory.vehicle_states, strict=True
    ):
        if frame.timestamp_us == scene.initial_timestamp_us:
            continue  # The reset response already published this observation.
        publish_frame(
            endpoint,
            episode,
            step,
            int(frame.timestamp_us),
            pose,
            vehicle,
            names,
            frame.view_model_rgb_host_uint8,
            scene_time_us=frame.scene_time_us,
            condition=(
                frame.view_rgb_host_uint8[0]
                if request.get("front_condition")
                and names
                and frame.scene_time_us is not None
                and frame.view_rgb_host_uint8
                else None
            ),
        )
        emitted += 1
    finish_step(
        endpoint,
        request,
        command,
        int(timestamps[0]),
        int(timestamps[-1]),
        next_apply_us,
        emitted,
        names,
    )


def finish_step(
    endpoint: DriveEndpoint,
    request: dict,
    command: DriverCommand,
    applied_from_us: int,
    last_observation_us: int,
    next_apply_us: int,
    frame_count: int,
    names: Sequence[str] = (),
) -> None:
    step = request["expected_step"]
    endpoint.send(
        {
            "type": "done",
            "episode_id": request["episode_id"],
            "step": step,
            "next_step": step + 1,
            "applied_from_us": applied_from_us,
            "last_observation_us": last_observation_us,
            "next_apply_us": next_apply_us,
            "frame_count": frame_count,
            "cameras": names,
            **({"manual_gear": command.gear} if command.manual_control else {}),
            "control": {
                "steering_angle_rad": (
                    command.steer * MANUAL_STEER_RAD
                    if command.manual_control
                    else request["steering_angle_rad"]
                ),
                "throttle": command.throttle,
                "brake": command.brake,
                "reverse": command.reverse,
            },
        }
    )
