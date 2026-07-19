# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gRPC serialization/deserialization utilities.

This module contains functions to convert between gRPC proto messages and
Python types used by the video generation API and renderer.

Key conversions:
- Image proto → Tensor
- StaticWorldMap (zip) → SceneData
- Trajectory proto → camera poses (numpy)
- DynamicWorldState proto → Ludus dynamic obstacle pool
- CameraSpec proto → FThetaCamera
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from google.protobuf.json_format import MessageToDict
from loguru import logger
from ludus_renderer import CUBE_FLAG_WIREFRAME, PRIM_OBSTACLE, CubePool
from ludus_renderer.clipgt import OBSTACLE_COLORS_V3
from omnidreams.conditioning.renderer import load_and_attach_ludus_scene
from omnidreams.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.conditioning.world_scenario.data_types import SceneData
from omnidreams.conditioning.world_scenario.ftheta import FThetaCamera
from omnidreams.conditioning.world_scenario.settings import SETTINGS
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from torch import Tensor

from flashdreams.serving.realtime.media import encode_rgb_frame_to_jpeg


def decode_image(
    image_bytes: bytes,
    image_format: str,
    target_resolution_hw: tuple[int, int] | None = None,
) -> Tensor:
    """
    Decode an image from bytes to a tensor.

    Args:
        image_bytes: Raw image bytes.
        image_format: Format string ("PNG", "JPEG", "RGB_UINT8_PLANAR").
        target_resolution_hw: Optional (height, width) to resize to.

    Returns:
        Tensor of shape [3, H, W], dtype uint8.
    """
    if image_format in ("PNG", "JPEG", "JPEG2000"):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if target_resolution_hw is not None:
            target_h, target_w = target_resolution_hw
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)
    elif image_format == "RGB_UINT8_PLANAR":
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if target_resolution_hw is not None:
            target_h, target_w = target_resolution_hw
            arr = arr.reshape(3, target_h, target_w)
        return torch.from_numpy(arr.copy())
    else:
        raise ValueError(f"Unsupported image format: {image_format}")


def encode_image(
    image_np: np.ndarray,
    format: str = "PNG",
    quality: int = 90,
) -> bytes:
    """
    Encode a numpy image as PNG or JPEG bytes.

    Args:
        image_np: Numpy array [H, W, 3] uint8.
        format: Image format ("PNG" or "JPEG").
        quality: JPEG quality (1-100). Higher is better quality but larger file.
            Only used when format is "JPEG".

    Returns:
        Encoded image bytes.
    """
    if format.upper() == "JPEG":
        return encode_rgb_frame_to_jpeg(image_np, quality=quality, value_range="uint8")

    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def load_static_world_from_zip_bytes(
    hdmap_zip_bytes: bytes,
    camera_names: list[str],
    target_resolution_hw: tuple[int, int],
    perform_mirror_augment: bool = False,
    include_dynamic_obstacles: bool = True,
) -> SceneData:
    """
    Load static world (HD map) from zip-compressed parquet bytes.

    The zip should contain parquet files as expected by load_scene.

    Args:
        hdmap_zip_bytes: Zip-compressed parquet files.
        camera_names: Camera names to load.
        target_resolution_hw: Target resolution (height, width).
        include_dynamic_obstacles: Whether to include clipgt dynamic obstacle tracks.

    Returns:
        SceneData object with HD map loaded (dynamic objects may be empty).
    """
    res_H, res_W = target_resolution_hw

    # Extract zip to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        with zipfile.ZipFile(io.BytesIO(hdmap_zip_bytes), "r") as zf:
            zf.extractall(tmppath)

        # Load scene from extracted files
        scene_data = load_scene(
            tmppath,
            camera_names=camera_names,
            max_frames=-1,
            input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
            resize_resolution_hw=[res_H, res_W],
        )

        scene_data = load_and_attach_ludus_scene(
            tmppath,
            scene_data,
            device=torch.device("cuda"),
            perform_mirror_augment=perform_mirror_augment,
            include_dynamic_obstacles=include_dynamic_obstacles,
        )

    return scene_data


def pose_to_matrix(
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Convert translation + quaternion to 4x4 transformation matrix.

    Args:
        translation: (x, y, z) translation.
        quat_wxyz: (w, x, y, z) quaternion (scalar-first convention).

    Returns:
        4x4 transformation matrix.
    """
    # scipy uses (x, y, z, w) convention, proto uses (w, x, y, z)
    w, x, y, z = quat_wxyz
    rot = Rotation.from_quat([x, y, z, w])
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = translation
    return mat


def trajectory_to_camera_poses(
    poses: list[dict],
) -> tuple[np.ndarray, list[int]]:
    """
    Convert a list of PoseAtTime dicts to camera poses array.

    Args:
        poses: List of dicts with 'pose' (translation + quat) and 'timestamp_us'.

    Returns:
        Tuple of:
        - camera_poses: Array of shape [num_frames, 4, 4]
        - timestamps_us: List of timestamps in microseconds
    """
    matrices = []
    timestamps = []
    for p in poses:
        pose = p.get("pose", {})
        vec = pose.get("vec", {})
        quat = pose.get("quat", {})
        translation = (
            float(vec.get("x", 0.0)),
            float(vec.get("y", 0.0)),
            float(vec.get("z", 0.0)),
        )
        quat_wxyz = (
            float(quat.get("w", 1.0)),  # Default to identity quaternion
            float(quat.get("x", 0.0)),
            float(quat.get("y", 0.0)),
            float(quat.get("z", 0.0)),
        )
        matrices.append(pose_to_matrix(translation, quat_wxyz))
        timestamps.append(int(p.get("timestamp_us", 0)))
    return np.stack(matrices), timestamps


ACTOR_CLASS_TO_OBSTACLE_CATEGORY = {
    0: "Other",
    1: "Car",
    2: "Truck",
    3: "Pedestrian",
    4: "Cyclist",
    5: "Other",
    "INVALID": "Other",
    "CAR": "Car",
    "TRUCK": "Truck",
    "PEDESTRIAN": "Pedestrian",
    "CYCLIST": "Cyclist",
    "OTHER": "Other",
}


def _actor_class_to_obstacle_category(class_id: object) -> str:
    if isinstance(class_id, str):
        stripped = class_id.strip()
        if stripped.isdigit():
            return ACTOR_CLASS_TO_OBSTACLE_CATEGORY.get(int(stripped), "Other")
        return ACTOR_CLASS_TO_OBSTACLE_CATEGORY.get(stripped.upper(), "Other")
    if isinstance(class_id, (int, np.integer)):
        return ACTOR_CLASS_TO_OBSTACLE_CATEGORY.get(int(class_id), "Other")
    return "Other"


def _pose_dict_to_translation_and_quat_xyzw(
    pose_at_time: dict,
    *,
    actor_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    pose = pose_at_time.get("pose", {})
    vec = pose.get("vec", {})
    quat = pose.get("quat", {})

    translation = np.array(
        [
            float(vec.get("x", 0.0)),
            float(vec.get("y", 0.0)),
            float(vec.get("z", 0.0)),
        ],
        dtype=np.float32,
    )
    quat_xyzw = np.array(
        [
            float(quat.get("x", 0.0)),
            float(quat.get("y", 0.0)),
            float(quat.get("z", 0.0)),
            float(quat.get("w", 1.0)),
        ],
        dtype=np.float32,
    )
    quat_norm = float(np.linalg.norm(quat_xyzw))
    if quat_norm <= 1e-6:
        raise ValueError(f"DynamicActor[{actor_index}] has an invalid zero quaternion")
    return translation, quat_xyzw / quat_norm


def _sample_actor_pose_at_timestamp(
    timestamps_us: np.ndarray,
    translations: np.ndarray,
    quats_xyzw: np.ndarray,
    frame_timestamp_us: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(timestamps_us) == 0:
        return None

    ts = int(frame_timestamp_us)
    first_ts = int(timestamps_us[0])
    last_ts = int(timestamps_us[-1])
    if ts < first_ts or ts > last_ts:
        return None

    exact_idx = np.where(timestamps_us == ts)[0]
    if len(exact_idx) > 0:
        idx = int(exact_idx[0])
        return translations[idx], quats_xyzw[idx]

    if len(timestamps_us) == 1:
        return None

    right_idx = int(np.searchsorted(timestamps_us, ts, side="left"))
    left_idx = right_idx - 1
    if left_idx < 0 or right_idx >= len(timestamps_us):
        return None

    t0 = int(timestamps_us[left_idx])
    t1 = int(timestamps_us[right_idx])
    if t1 <= t0:
        return translations[left_idx], quats_xyzw[left_idx]

    alpha = (ts - t0) / (t1 - t0)
    translation = (
        (1.0 - alpha) * translations[left_idx] + alpha * translations[right_idx]
    ).astype(np.float32)
    rotations = Rotation.from_quat([quats_xyzw[left_idx], quats_xyzw[right_idx]])
    slerp = Slerp([float(t0), float(t1)], rotations)
    quat_xyzw = slerp([float(ts)]).as_quat()[0].astype(np.float32)
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    return translation, quat_xyzw


def dynamic_state_to_ludus_cube_pool(
    dynamic_state: dict,
    frame_timestamps_us: list[int],
    device: torch.device | str,
) -> CubePool | None:
    """Convert gRPC DynamicWorldState data into a Ludus dynamic obstacle pool.

    The gRPC request is authoritative: no actors means no dynamic obstacle pool.
    Actors are sampled only within their provided trajectory time range.
    """
    actors = dynamic_state.get("actors", [])
    if not actors:
        return None

    device = torch.device(device)
    included_track_timestamps: list[torch.Tensor] = []
    included_translations: list[torch.Tensor] = []
    included_quaternions: list[torch.Tensor] = []
    included_scales: list[torch.Tensor] = []
    included_colors: list[torch.Tensor] = []

    for actor_index, actor in enumerate(actors):
        trajectory = actor.get("trajectory", {}).get("poses", [])
        if not trajectory:
            raise ValueError(f"DynamicActor[{actor_index}] has no trajectory poses")

        bbox = actor.get("bbox_dims", {})
        scale = np.array(
            [
                float(bbox.get("size_x", 0.0)),
                float(bbox.get("size_y", 0.0)),
                float(bbox.get("size_z", 0.0)),
            ],
            dtype=np.float32,
        )
        if np.any(scale <= 0.0):
            raise ValueError(
                f"DynamicActor[{actor_index}] has nonpositive bbox dimensions: {scale.tolist()}"
            )

        pose_by_timestamp: dict[int, dict] = {}
        for pose_at_time in trajectory:
            ts = int(pose_at_time.get("timestamp_us", 0))
            pose_by_timestamp[ts] = pose_at_time
        sorted_items = sorted(pose_by_timestamp.items(), key=lambda item: item[0])

        actor_timestamps = np.array([ts for ts, _ in sorted_items], dtype=np.int64)
        actor_translations = []
        actor_quats_xyzw = []
        for _, pose_at_time in sorted_items:
            translation, quat_xyzw = _pose_dict_to_translation_and_quat_xyzw(
                pose_at_time, actor_index=actor_index
            )
            actor_translations.append(translation)
            actor_quats_xyzw.append(quat_xyzw)
        translations_np = np.stack(actor_translations).astype(np.float32)
        quats_xyzw_np = np.stack(actor_quats_xyzw).astype(np.float32)

        sampled_timestamps: list[int] = []
        sampled_translations: list[np.ndarray] = []
        sampled_quats: list[np.ndarray] = []
        for frame_ts in frame_timestamps_us:
            sampled = _sample_actor_pose_at_timestamp(
                actor_timestamps,
                translations_np,
                quats_xyzw_np,
                int(frame_ts),
            )
            if sampled is None:
                continue
            translation, quat_xyzw = sampled
            sampled_timestamps.append(int(frame_ts))
            sampled_translations.append(translation)
            sampled_quats.append(quat_xyzw)

        if not sampled_timestamps:
            continue

        sampled_items = sorted(
            zip(sampled_timestamps, sampled_translations, sampled_quats),
            key=lambda item: item[0],
        )
        sampled_timestamps = [item[0] for item in sampled_items]
        sampled_translations = [item[1] for item in sampled_items]
        sampled_quats = [item[2] for item in sampled_items]

        category = _actor_class_to_obstacle_category(actor.get("class_id", 0))
        front_color, back_color = OBSTACLE_COLORS_V3[category]
        colors = np.array([*front_color, *back_color], dtype=np.float32)

        included_track_timestamps.append(
            torch.tensor(sampled_timestamps, dtype=torch.int64, device=device)
        )
        included_translations.append(
            torch.tensor(
                np.stack(sampled_translations), dtype=torch.float32, device=device
            )
        )
        included_quaternions.append(
            torch.tensor(np.stack(sampled_quats), dtype=torch.float32, device=device)
        )
        included_scales.append(torch.tensor(scale, dtype=torch.float32, device=device))
        included_colors.append(torch.tensor(colors, dtype=torch.float32, device=device))

    if not included_track_timestamps:
        return None

    track_lengths = torch.tensor(
        [len(timestamps) for timestamps in included_track_timestamps],
        dtype=torch.int32,
        device=device,
    )
    cube_ts_prefix_sum = torch.cumsum(track_lengths, dim=0, dtype=torch.int32)
    all_track_timestamps = torch.cat(included_track_timestamps)
    timestamps_us = torch.unique(all_track_timestamps).sort()[0]

    return CubePool(
        timestamps_us=timestamps_us,
        cube_ts_prefix_sum=cube_ts_prefix_sum,
        track_timestamps_us=all_track_timestamps,
        translations=torch.cat(included_translations),
        quaternions=torch.cat(included_quaternions),
        scales=torch.stack(included_scales),
        colors=torch.stack(included_colors),
        prim_type_id=PRIM_OBSTACLE,
        render_flags=CUBE_FLAG_WIREFRAME,
    )


def parse_rig_to_camera(rig_to_camera_dict: dict) -> np.ndarray:
    """
    Convert a rig-to-camera Pose dict to a 4x4 transformation matrix.

    An absent or empty pose is treated as identity, meaning the camera coincides
    with the rig origin.

    Args:
        rig_to_camera_dict: Dict representation of a ``common.Pose`` proto message.

    Returns:
        4x4 rig_to_camera transformation matrix (float32).
    """
    rig_to_cam = rig_to_camera_dict
    if not rig_to_cam:
        return np.eye(4, dtype=np.float32)

    vec = rig_to_cam.get("vec", {})
    quat = rig_to_cam.get("quat", {})

    translation = (
        float(vec.get("x", 0.0)),
        float(vec.get("y", 0.0)),
        float(vec.get("z", 0.0)),
    )
    quat_wxyz = (
        float(quat.get("w", 1.0)),  # Default to identity quaternion
        float(quat.get("x", 0.0)),
        float(quat.get("y", 0.0)),
        float(quat.get("z", 0.0)),
    )

    return pose_to_matrix(translation, quat_wxyz)


def parse_rig_to_camera_transforms(
    rig_to_camera_messages: list[object],
    camera_names: list[str],
) -> dict[str, np.ndarray]:
    """Parse SessionRequest.rig_to_camera into a per-camera transform map."""
    if len(rig_to_camera_messages) != len(camera_names):
        raise ValueError(
            "SessionRequest.rig_to_camera must contain exactly one Pose per "
            f"camera_spec; got {len(rig_to_camera_messages)} poses for "
            f"{len(camera_names)} camera_specs."
        )

    return {
        cam_name: parse_rig_to_camera(proto_to_dict(rig_to_camera))
        for cam_name, rig_to_camera in zip(camera_names, rig_to_camera_messages)
    }


def compute_camera_poses_from_rig(
    rig_poses: np.ndarray | torch.Tensor,
    rig_to_camera: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Compute per-frame camera-to-world poses from rig-to-world poses.

    For each frame *t*::

        camera_to_world[t] = rig_to_world[t]  @  rig_to_camera

    Args:
        rig_poses: Array of shape ``[N, 4, 4]`` — rig-to-world transforms per frame.
        rig_to_camera: A single ``[4, 4]`` rig-to-camera transform.

    Returns:
        Array of shape ``[N, 4, 4]`` — camera-to-world transforms per frame.
    """
    # Vectorised matmul over the batch dimension
    if isinstance(rig_poses, torch.Tensor):
        return torch.einsum("nij,jk->nik", rig_poses, rig_to_camera)
    else:
        return np.einsum("nij,jk->nik", rig_poses, rig_to_camera).astype(np.float32)


def camera_spec_to_ftheta(camera_spec: dict) -> FThetaCamera:
    """
    Convert gRPC CameraSpec to FThetaCamera.

    Currently only supports FTheta camera model. Other models
    (OpenCV pinhole, fisheye) would need additional conversion logic.

    Args:
        camera_spec: Dict with camera parameters from proto.

    Returns:
        FThetaCamera instance.

    Raises:
        ValueError: If camera model is not supported.
    """
    resolution_h = camera_spec.get("resolution_h", 480)
    resolution_w = camera_spec.get("resolution_w", 832)

    # Check which camera model is present
    if "ftheta_param" in camera_spec:
        ftheta = camera_spec["ftheta_param"]

        # Build intrinsics array for FThetaCamera.from_numpy
        # Expected format: [cx, cy, width, height, *poly(6), is_bw_poly, linear_c, linear_d, linear_e]
        cx = ftheta.get("principal_point_x", resolution_w / 2)
        cy = ftheta.get("principal_point_y", resolution_h / 2)

        # Get polynomial coefficients.
        # NOTE: proto_to_dict (MessageToDict) converts enum fields to their
        # *string* names (e.g. "PIXELDIST_TO_ANGLE"), not integer values.
        poly_type = ftheta.get("reference_poly", 1)
        is_backward = poly_type in (1, "PIXELDIST_TO_ANGLE")

        if is_backward:
            is_bw_poly = 1.0
            poly_coeffs = ftheta.get("pixeldist_to_angle_poly", [])
        else:
            is_bw_poly = 0.0
            poly_coeffs = ftheta.get("angle_to_pixeldist_poly", [])

        linear_cde = ftheta.get("linear_cde", {})
        linear_c = linear_cde.get("linear_c", 1.0)
        linear_d = linear_cde.get("linear_d", 0.0)
        linear_e = linear_cde.get("linear_e", 0.0)

        # Pad polynomial to 6 coefficients (from_numpy reads positions 4:10)
        poly_padded = list(poly_coeffs) + [0.0] * max(0, 6 - len(poly_coeffs))

        # FThetaCamera.from_numpy expects:
        #   [cx, cy, width, height, *poly(6), is_bw_poly, linear_c, linear_d, linear_e]
        intrinsics = np.array(
            [
                cx,
                cy,
                resolution_w,
                resolution_h,
                poly_padded[0],
                poly_padded[1],
                poly_padded[2],
                poly_padded[3],
                poly_padded[4],
                poly_padded[5],
                is_bw_poly,
                linear_c,
                linear_d,
                linear_e,
            ],
            dtype=np.float64,
        )
        logger.debug(
            "camera_spec_to_ftheta: res=%dx%d, cx=%.1f, cy=%.1f, is_bw=%s, poly_type=%r, poly=%s"
            % (
                resolution_w,
                resolution_h,
                cx,
                cy,
                is_bw_poly,
                poly_type,
                poly_coeffs,
            )
        )

        return FThetaCamera.from_numpy(intrinsics)

    elif "opencv_pinhole_param" in camera_spec:
        raise ValueError(
            "OpenCV pinhole camera not yet supported. Convert to FTheta or implement opencv_pinhole_to_ftheta()."
        )
    elif "opencv_fisheye_param" in camera_spec:
        raise ValueError(
            "OpenCV fisheye camera not yet supported. Convert to FTheta or implement opencv_fisheye_to_ftheta()."
        )
    else:
        raise ValueError("No supported camera model found in camera_spec")


def proto_to_dict(proto_msg) -> dict:
    """Convert a protobuf message to a dict, preserving proto field names."""
    # always_print_fields_with_no_presence is protobuf 4.x+; older versions
    # spell the same option including_default_value_fields.
    try:
        return MessageToDict(
            proto_msg,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:
        return MessageToDict(
            proto_msg,
            preserving_proto_field_name=True,
            including_default_value_fields=True,  # ty:ignore[unknown-argument]
        )
