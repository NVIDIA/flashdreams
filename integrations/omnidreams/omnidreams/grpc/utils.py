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
- DynamicWorldState proto → renderer object_info
- CameraSpec proto → FThetaCamera
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from google.protobuf.json_format import MessageToDict
from loguru import logger
from omnidreams.conditioning.renderer import load_and_attach_ludus_scene
from omnidreams.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.conditioning.world_scenario.data_types import SceneData
from omnidreams.conditioning.world_scenario.data_utils import convert_pose_flu_to_rdf
from omnidreams.conditioning.world_scenario.ftheta import FThetaCamera
from omnidreams.conditioning.world_scenario.settings import SETTINGS
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor

# =============================================================================
# Image Encoding/Decoding
# =============================================================================


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
        # Convert to tensor [C, H, W]
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)
    elif image_format == "RGB_UINT8_PLANAR":
        # Raw planar RGB data
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
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    if format.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=format)
    return buf.getvalue()


# =============================================================================
# Static World Map Loading
# =============================================================================


def load_static_world_from_zip_bytes(
    hdmap_zip_bytes: bytes,
    camera_names: list[str],
    target_resolution_hw: tuple[int, int],
    perform_mirror_augment: bool = False,
) -> SceneData:
    """
    Load static world (HD map) from zip-compressed parquet bytes.

    The zip should contain parquet files as expected by load_scene.

    Args:
        hdmap_zip_bytes: Zip-compressed parquet files.
        camera_names: Camera names to load.
        target_resolution_hw: Target resolution (height, width).

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
        )

    return scene_data


# =============================================================================
# Pose and Trajectory Conversion
# =============================================================================


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


# =============================================================================
# Dynamic State Conversion (gRPC DynamicWorldState → renderer object_info)
# =============================================================================

# Actor class mapping from proto enum to renderer's expected types
ACTOR_CLASS_MAP = {
    0: "Others",  # INVALID
    1: "Car",  # CAR
    2: "Truck",  # TRUCK
    3: "Pedestrian",  # PEDESTRIAN
    4: "Cyclist",  # CYCLIST
    5: "Others",  # OTHER
}


def dynamic_actor_to_object_info(
    actor: dict,
    frame_timestamp_us: int,
    coordinate_system: Literal["FLU", "RDF"] = "FLU",
) -> dict | None:
    """
    Convert a single DynamicActor to renderer object_info format at a specific time.

    Interpolates the actor's trajectory to get pose at the given timestamp.

    Args:
        actor: Dict with 'class_id', 'bbox_dims', 'trajectory'.
        frame_timestamp_us: Timestamp in microseconds for which to get pose.

    Returns:
        Dict with 'object_type', 'object_to_world', 'object_lwh', or None if
        actor is not visible at this time.
    """
    trajectory = actor.get("trajectory", {}).get("poses", [])
    if not trajectory:
        return None

    # Find poses bracketing the requested timestamp for interpolation
    # For now, use nearest neighbor (TODO: linear interpolation)
    best_pose = None
    best_diff = float("inf")
    for pose_at_time in trajectory:
        ts = pose_at_time.get("timestamp_us", 0)
        diff = abs(ts - frame_timestamp_us)
        if diff < best_diff:
            best_diff = diff
            best_pose = pose_at_time

    if best_pose is None:
        return None

    # Convert pose to 4x4 matrix
    pose = best_pose.get("pose", {})
    vec = pose.get("vec", {})
    quat = pose.get("quat", {})
    translation = (
        float(vec.get("x", 0.0)),
        float(vec.get("y", 0.0)),
        float(vec.get("z", 0.0)),
    )
    quat_wxyz = (
        float(quat.get("w", 1.0)),
        float(quat.get("x", 0.0)),
        float(quat.get("y", 0.0)),
        float(quat.get("z", 0.0)),
    )
    object_to_world_flu = pose_to_matrix(translation, quat_wxyz)

    # Convert from FLU (client coordinate frame) to RDF (internal coordinate frame)
    if coordinate_system == "RDF":
        object_to_world = convert_pose_flu_to_rdf(object_to_world_flu)
    else:
        object_to_world = object_to_world_flu

    # Get dimensions (AABB: size_x, size_y, size_z → LWH convention)
    bbox = actor.get("bbox_dims", {})
    # Client sends dimensions in FLU order: size_x=forward/length, size_y=left/width, size_z=up/height
    # Renderer expects [length, width, height] which is the same semantic order
    object_lwh = np.array(
        [
            bbox.get("size_x", 1.0),  # length (forward direction)
            bbox.get("size_y", 1.0),  # width (left direction)
            bbox.get("size_z", 1.0),  # height (up direction)
        ],
        dtype=np.float32,
    )

    # Map class ID to type string
    class_id = actor.get("class_id", 0)
    object_type = ACTOR_CLASS_MAP.get(class_id, "Others")

    return {
        "object_type": object_type,
        "object_to_world": object_to_world,
        "object_lwh": object_lwh,
    }


def dynamic_state_to_object_info(
    dynamic_state: dict,
    frame_timestamp_us: int,
    coordinate_system: Literal["FLU", "RDF"] = "FLU",
) -> dict[str, dict]:
    """
    Convert gRPC DynamicWorldState to renderer's object_info format.

    Args:
        dynamic_state: Dict with 'actors' (list of DynamicActor dicts).
        frame_timestamp_us: Timestamp for which to get actor poses.

    Returns:
        Dict mapping tracking_id → object info suitable for renderer.
        Format: {tracking_id: {"object_type", "object_to_world", "object_lwh"}}
    """
    object_info = {}
    actors = dynamic_state.get("actors", [])

    for idx, actor in enumerate(actors):
        info = dynamic_actor_to_object_info(
            actor, frame_timestamp_us, coordinate_system=coordinate_system
        )
        if info is not None:
            # Use index as tracking ID (proto doesn't have explicit IDs)
            tracking_id = str(idx)
            object_info[tracking_id] = info

    return object_info


# =============================================================================
# Rig-to-Camera Transforms
# =============================================================================


def parse_rig_to_camera(camera_spec_dict: dict) -> np.ndarray:
    """
    Extract the rig_to_camera 4x4 transformation matrix from a CameraSpec dict.

    If the ``rig_to_camera`` field is absent or contains the default (zero)
    values, an identity matrix is returned — meaning the camera coincides
    with the rig origin.

    Args:
        camera_spec_dict: Dict representation of a ``CameraSpec`` proto message.

    Returns:
        4x4 rig_to_camera transformation matrix (float32).
    """
    rig_to_cam = camera_spec_dict.get("rig_to_camera", {})
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


# =============================================================================
# Camera Intrinsics Conversion (gRPC CameraSpec → FThetaCamera)
# =============================================================================


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


# =============================================================================
# Protobuf Helper
# =============================================================================


def proto_to_dict(proto_msg) -> dict:
    """
    Convert protobuf message to dictionary.

    Args:
        proto_msg: Protobuf message.

    Returns:
        Dictionary representation.
    """
    # Handle both old and new protobuf API
    # Old: including_default_value_fields (deprecated in protobuf 4.x)
    # New: always_print_fields_with_no_presence (protobuf 4.x+)
    try:
        return MessageToDict(
            proto_msg,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:
        # Fall back to old API for older protobuf versions
        return MessageToDict(
            proto_msg,
            preserving_proto_field_name=True,
            including_default_value_fields=True,  # ty:ignore[unknown-argument]
        )
