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

"""SANA-WM camera-pose DSL, intrinsics, and Plucker conditioning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sana_wm.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    SANA_WM_VAE_SPATIAL_COMPRESSION,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)

FPS = 16
"""Camera-control integration frame rate."""

DEFAULT_TRANSLATION_SPEED = 0.025
"""Default per-frame translation magnitude used by SANA-WM."""

DEFAULT_ROTATION_SPEED_DEG = 0.6
"""Default per-frame rotation magnitude in degrees."""

DEFAULT_PITCH_LIMIT_DEG = 60.0
"""Maximum absolute camera pitch in degrees."""

TAU_PRESS = 0.45
"""Velocity smoothing time constant when a key is pressed."""

TAU_COAST = 1.0
"""Velocity smoothing time constant when controls are released."""

DSL_KEY_TO_CONTROL: dict[str, str] = {
    "w": "forward",
    "s": "back",
    "a": "yaw_left",
    "d": "yaw_right",
    "i": "pitch_up",
    "k": "pitch_down",
    "j": "strafe_left",
    "l": "strafe_right",
}
"""Mapping from SANA-WM action DSL letters to canonical controls."""

ALLOWED_ACTION_KEYS = frozenset(DSL_KEY_TO_CONTROL)
"""Action DSL keys accepted in ``<keys>-<frames>`` segments."""


@dataclass
class VelocityState:
    """Per-frame camera velocity in OpenCV camera coordinates."""

    tx: float = 0.0
    """Forward translation velocity."""

    sx: float = 0.0
    """Rightward strafe velocity."""

    yaw: float = 0.0
    """Positive yaw-right angular velocity in radians."""

    pitch: float = 0.0
    """Positive pitch-up angular velocity in radians."""

    def snap_to(self, target: "VelocityState") -> None:
        """Copy all velocity components from ``target``."""
        self.tx, self.sx = target.tx, target.sx
        self.yaw, self.pitch = target.yaw, target.pitch

    def step_toward(self, target: "VelocityState", dt: float) -> None:
        """Ease velocity components toward ``target`` over one timestep."""
        for attr in ("tx", "sx", "yaw", "pitch"):
            cur = getattr(self, attr)
            tgt = getattr(target, attr)
            tau = TAU_PRESS if abs(tgt) > 1e-12 else TAU_COAST
            alpha = 1.0 - math.exp(-dt / tau)
            setattr(self, attr, cur + alpha * (tgt - cur))


class CameraPoseIntegrator:
    """Integrate camera velocity into camera-to-world poses."""

    def __init__(
        self, pitch_limit_rad: float = math.radians(DEFAULT_PITCH_LIMIT_DEG)
    ) -> None:
        self.pose = np.eye(4, dtype=np.float64)
        self.pitch = 0.0
        self.pitch_limit = float(pitch_limit_rad)

    def step(self, velocity: VelocityState) -> np.ndarray:
        """Advance one frame and return the current camera-to-world pose."""
        new_pitch = max(
            -self.pitch_limit, min(self.pitch_limit, self.pitch + velocity.pitch)
        )
        pitch_step = new_pitch - self.pitch
        self.pitch = new_pitch

        rotation = self.pose[:3, :3]
        rotation_new = _rot_y(velocity.yaw) @ rotation @ _rot_x(pitch_step)

        forward = rotation_new[:, 2].copy()
        forward[1] = 0.0
        right = rotation_new[:, 0].copy()
        right[1] = 0.0
        forward_norm = float(np.linalg.norm(forward))
        right_norm = float(np.linalg.norm(right))
        if forward_norm > 0:
            forward /= forward_norm + 1e-6
        if right_norm > 0:
            right /= right_norm + 1e-6

        translation = self.pose[:3, 3] + forward * velocity.tx + right * velocity.sx
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, :3] = rotation_new
        self.pose[:3, 3] = translation
        return self.pose.copy()


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def controls_to_target_velocity(
    controls: set[str],
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_rad: float | None = None,
) -> VelocityState:
    """Map canonical control tokens to a target velocity."""
    if rotation_speed_rad is None:
        rotation_speed_rad = math.radians(DEFAULT_ROTATION_SPEED_DEG)
    forward = (1.0 if "forward" in controls else 0.0) - (
        1.0 if "back" in controls else 0.0
    )
    strafe = (1.0 if "strafe_right" in controls else 0.0) - (
        1.0 if "strafe_left" in controls else 0.0
    )
    yaw = (1.0 if "yaw_right" in controls else 0.0) - (
        1.0 if "yaw_left" in controls else 0.0
    )
    pitch = (1.0 if "pitch_up" in controls else 0.0) - (
        1.0 if "pitch_down" in controls else 0.0
    )
    return VelocityState(
        tx=forward * translation_speed,
        sx=strafe * translation_speed,
        yaw=yaw * rotation_speed_rad,
        pitch=pitch * rotation_speed_rad,
    )


def parse_action_string(
    action: str, *, max_frames: int | None = None
) -> list[list[str]]:
    """Expand a SANA-WM action string into per-frame held keys.

    Args:
        action: Comma-separated ``<keys>-<frames>`` segments. ``none-N``
            holds the pose for ``N`` frames.
        max_frames: Optional maximum number of per-frame entries to
            materialize. All segments are still validated.

    Returns:
        Per-frame lists of DSL keys.

    Raises:
        ValueError: The action string is empty or contains an invalid segment.
    """
    if max_frames is not None and max_frames < 0:
        raise ValueError(f"max_frames must be non-negative, got {max_frames}.")
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(
                f"Invalid action segment {segment!r}: expected '<keys>-<frames>'."
            )
        keys_part, duration_part = segment.rsplit("-", 1)
        if not duration_part.isdigit():
            raise ValueError(f"Action segment {segment!r} has a non-positive duration.")
        duration = int(duration_part)
        if duration <= 0:
            raise ValueError(f"Action segment {segment!r} has a non-positive duration.")
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted({key for key in keys_lower if key not in ALLOWED_ACTION_KEYS})
            if bad:
                raise ValueError(
                    f"Action segment {segment!r} contains unknown keys {bad}; "
                    f"allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        if max_frames is None:
            per_frame.extend([keys] * duration)
        elif len(per_frame) < max_frames:
            per_frame.extend([keys] * min(duration, max_frames - len(per_frame)))
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
    smooth: bool = True,
    num_frames: int | None = None,
) -> np.ndarray:
    """Roll out a camera-to-world trajectory from an action string.

    Args:
        action: Comma-separated ``<keys>-<frames>`` segments.
        translation_speed: Per-frame translation magnitude.
        rotation_speed_deg: Per-frame angular magnitude in degrees.
        pitch_limit_deg: Maximum absolute pitch in degrees.
        smooth: Whether to use the SANA-WM velocity smoothing model.
        num_frames: Optional output frame count. When provided, the action
            prompt is repeated or truncated to produce exactly this many
            camera poses.

    Returns:
        ``[F + 1, 4, 4]`` camera-to-world matrices in OpenCV convention.
    """
    if num_frames is None:
        per_frame = parse_action_string(action)
    else:
        if num_frames < 1:
            raise ValueError(f"num_frames must be positive, got {num_frames}.")
        target_steps = num_frames - 1
        per_frame = parse_action_string(action, max_frames=target_steps)
        if target_steps == 0:
            per_frame = []
        elif len(per_frame) < target_steps:
            repeats = math.ceil(target_steps / len(per_frame))
            per_frame = (per_frame * repeats)[:target_steps]
        else:
            per_frame = per_frame[:target_steps]
    integrator = CameraPoseIntegrator(math.radians(pitch_limit_deg))
    velocity = VelocityState()
    poses = [integrator.pose.copy()]
    last_controls: set[str] = set()
    dt = 1.0 / FPS
    rotation_speed_rad = math.radians(rotation_speed_deg)

    for keys in per_frame:
        controls = {DSL_KEY_TO_CONTROL[key] for key in keys}
        target = controls_to_target_velocity(
            controls,
            translation_speed=translation_speed,
            rotation_speed_rad=rotation_speed_rad,
        )
        if smooth:
            if controls - last_controls:
                velocity.snap_to(target)
            else:
                velocity.step_toward(target, dt)
            last_controls = controls
        else:
            velocity = target
        poses.append(integrator.step(velocity))

    return np.stack(poses, axis=0).astype(np.float32)


def fit_camera_trajectory(c2w: np.ndarray, num_frames: int) -> np.ndarray:
    """Return a camera-to-world trajectory fitted to ``num_frames`` poses.

    Translations are interpolated linearly. Rotations are interpolated in
    matrix space and projected back to the nearest proper rotation, avoiding a
    new SciPy dependency for runner-time camera fitting.
    """
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"camera trajectory must be [F, 4, 4], got {c2w.shape}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if c2w.shape[0] == num_frames:
        return c2w.copy()
    if c2w.shape[0] == 1:
        return np.broadcast_to(c2w[:1], (num_frames, 4, 4)).copy()

    old_t = np.linspace(0.0, 1.0, c2w.shape[0], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    fitted = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, 4, 4)).copy()

    for axis in range(3):
        fitted[:, axis, 3] = np.interp(
            new_t,
            old_t,
            c2w[:, axis, 3],
        ).astype(np.float32)

    rotations = c2w[:, :3, :3].reshape(c2w.shape[0], 9)
    interp_rotations = np.empty((num_frames, 9), dtype=np.float32)
    for idx in range(9):
        interp_rotations[:, idx] = np.interp(
            new_t,
            old_t,
            rotations[:, idx],
        ).astype(np.float32)
    for frame_idx, rotation in enumerate(interp_rotations.reshape(num_frames, 3, 3)):
        u, _, vh = np.linalg.svd(rotation.astype(np.float64), full_matrices=False)
        projected = u @ vh
        if np.linalg.det(projected) < 0.0:
            u[:, -1] *= -1.0
            projected = u @ vh
        fitted[frame_idx, :3, :3] = projected.astype(np.float32)
    return fitted


def fit_intrinsics_sequence(arr: np.ndarray, num_frames: int) -> np.ndarray:
    """Return ``arr`` fitted to ``num_frames`` along axis 0."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == num_frames:
        return arr.copy()
    if arr.shape[0] > num_frames:
        return arr[:num_frames].copy()
    if arr.shape[0] == 1:
        return np.broadcast_to(arr[:1], (num_frames, *arr.shape[1:])).copy()

    old_t = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    fitted = np.empty((num_frames, flat.shape[1]), dtype=np.float32)
    for idx in range(flat.shape[1]):
        fitted[:, idx] = np.interp(new_t, old_t, flat[:, idx]).astype(np.float32)
    return fitted.reshape((num_frames, *arr.shape[1:]))


def load_intrinsics(path: Path, num_frames: int) -> np.ndarray:
    """Return ``[num_frames, 4]`` intrinsics as ``[fx, fy, cx, cy]``.

    Args:
        path: ``.npy`` file shaped ``[3, 3]``, ``[F, 3, 3]``, ``[4]``, or
            ``[F, 4]``.
        num_frames: Number of frames required by the rollout.

    Returns:
        Per-frame vector intrinsics.

    Raises:
        ValueError: The file shape is unsupported.
    """
    arr = np.load(path).astype(np.float32)
    if arr.shape == (4,):
        return np.broadcast_to(arr, (num_frames, 4)).copy()
    if arr.shape == (3, 3):
        vector = np.array(
            [arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=np.float32
        )
        return np.broadcast_to(vector, (num_frames, 4)).copy()
    if arr.ndim == 2 and arr.shape[1] == 4:
        return fit_intrinsics_sequence(arr, num_frames)
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        matrix = fit_intrinsics_sequence(arr, num_frames)
        return np.stack(
            [matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 0, 2], matrix[:, 1, 2]],
            axis=1,
        )
    raise ValueError(
        f"Unsupported intrinsics shape {arr.shape} for num_frames={num_frames}; "
        "expected (3,3), (F,3,3), (4,), or (F,4)."
    )


def default_intrinsics_vec4(
    src_size: tuple[int, int],
    num_frames: int,
    *,
    hfov_deg: float = 90.0,
) -> np.ndarray:
    """Derive ``[fx, fy, cx, cy]`` intrinsics from an image size.

    Used when the caller does not supply an intrinsics file. Assumes square
    pixels and a principal point at the image center, with focal length set by
    a horizontal field of view. The public SANA-WM demo intrinsics correspond
    to roughly ``90`` degrees; lower values are narrower/more zoomed-in. The
    returned intrinsics are in the source image's pixel coordinates, matching
    :func:`load_intrinsics`, so they flow through
    :func:`transform_intrinsics_for_crop` identically.

    Args:
        src_size: Source image size as ``(width, height)``.
        num_frames: Number of frames required by the rollout.
        hfov_deg: Horizontal field of view in degrees.

    Returns:
        ``[num_frames, 4]`` intrinsics as ``[fx, fy, cx, cy]``.
    """
    src_w, src_h = src_size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"src_size must be positive, got {src_size}.")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}.")
    focal = 0.5 * src_w / math.tan(math.radians(hfov_deg) / 2.0)
    vector = np.array([focal, focal, src_w / 2.0, src_h / 2.0], dtype=np.float32)
    return np.broadcast_to(vector, (num_frames, 4)).copy()


def snap_num_frames(
    num_frames: int,
    *,
    stride: int = SANA_WM_VAE_TEMPORAL_COMPRESSION,
    upper_bound: int | None = None,
) -> int:
    """Snap frame count to the nearest ``stride * k + 1`` value."""
    if num_frames < 1:
        return 1
    if (num_frames - 1) % stride == 0:
        return num_frames
    floor_cand = num_frames - ((num_frames - 1) % stride)
    ceil_cand = floor_cand + stride
    snapped = (
        floor_cand
        if (num_frames - floor_cand) < (ceil_cand - num_frames)
        else ceil_cand
    )
    if upper_bound is not None and snapped > upper_bound:
        snapped = floor_cand
    return max(snapped, 1)


def transform_intrinsics_for_crop(
    intrinsics_vec4: np.ndarray,
    src_size: tuple[int, int],
    resized_size: tuple[int, int],
    crop_offset: tuple[int, int],
) -> np.ndarray:
    """Adjust ``[fx, fy, cx, cy]`` to match resize-then-center-crop pixels."""
    src_w, src_h = src_size
    resized_w, resized_h = resized_size
    crop_left, crop_top = crop_offset
    sx, sy = resized_w / src_w, resized_h / src_h
    out = intrinsics_vec4.copy()
    out[..., 0] *= sx
    out[..., 2] = out[..., 2] * sx - crop_left
    out[..., 1] *= sy
    out[..., 3] = out[..., 3] * sy - crop_top
    return out


def resize_center_crop_geometry(
    src_size: tuple[int, int],
    *,
    target_h: int = DEFAULT_VIDEO_HEIGHT,
    target_w: int = DEFAULT_VIDEO_WIDTH,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return resize dimensions and center-crop offset for an input image size.

    Args:
        src_size: Source image size as ``(width, height)``.
        target_h: Target crop height.
        target_w: Target crop width.

    Returns:
        ``(resized_size, crop_offset)`` where sizes are ``(width, height)``
        and offsets are ``(left, top)``.
    """
    src_w, src_h = src_size
    scale = max(target_h / src_h, target_w / src_w)
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    return (resized_w, resized_h), (left, top)


def get_pose_inverse(transform: torch.Tensor) -> torch.Tensor:
    """Return inverse homogeneous transforms using rotation transpose."""
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -torch.matmul(rotation_inv, translation.unsqueeze(-1)).squeeze(-1)
    out = torch.eye(4, dtype=transform.dtype, device=transform.device).repeat(
        transform.shape[:-2] + (1, 1)
    )
    out[..., :3, :3] = rotation_inv
    out[..., :3, 3] = translation_inv
    return out


def compute_raymap(
    intrinsics: torch.Tensor,
    poses: torch.Tensor,
    height: int,
    width: int,
    *,
    use_plucker: bool = True,
) -> torch.Tensor:
    """Compute world-space ray directions and moments for camera poses.

    Args:
        intrinsics: ``[T, 4]`` tensor of ``[fx, fy, cx, cy]``.
        poses: ``[T, 4, 4]`` camera-to-world matrices.
        height: Raymap height.
        width: Raymap width.
        use_plucker: ``True`` returns ``[direction, moment]``; otherwise
            returns ``[origin, direction]``.

    Returns:
        ``[T, height, width, 6]`` raymap tensor.
    """
    num_frames = intrinsics.shape[0]
    device, dtype = intrinsics.device, intrinsics.dtype
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_grid = x_grid[None].expand(num_frames, -1, -1)
    y_grid = y_grid[None].expand(num_frames, -1, -1)

    fx = intrinsics[:, 0].view(num_frames, 1, 1)
    fy = intrinsics[:, 1].view(num_frames, 1, 1)
    cx = intrinsics[:, 2].view(num_frames, 1, 1)
    cy = intrinsics[:, 3].view(num_frames, 1, 1)

    dirs_cam = torch.stack(
        [(x_grid - cx) / fx, (y_grid - cy) / fy, torch.ones_like(x_grid)],
        dim=-1,
    )
    rotation = poses[:, :3, :3]
    translation = poses[:, :3, 3]
    dirs_world = torch.einsum("tij,thwj->thwi", rotation, dirs_cam)
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
    origins = translation.view(num_frames, 1, 1, 3).expand_as(dirs_world)
    if use_plucker:
        return torch.cat([dirs_world, torch.cross(origins, dirs_world, dim=-1)], dim=-1)
    return torch.cat([origins, dirs_world], dim=-1)


def prepare_camera(
    poses_c2w: np.ndarray,
    intrinsics_vec4: np.ndarray,
    *,
    target_size: tuple[int, int] = (DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH),
    vae_stride: tuple[int, int, int] = (
        SANA_WM_VAE_TEMPORAL_COMPRESSION,
        SANA_WM_VAE_SPATIAL_COMPRESSION,
        SANA_WM_VAE_SPATIAL_COMPRESSION,
    ),
) -> dict[str, torch.Tensor]:
    """Build SANA-WM raymap and chunk-Plucker conditioning tensors."""
    num_frames = poses_c2w.shape[0]
    vae_time_stride, vae_spatial_stride = vae_stride[0], vae_stride[-1]
    pixel_h, pixel_w = target_size
    latent_h = pixel_h // vae_spatial_stride
    latent_w = pixel_w // vae_spatial_stride
    latent_frames = (num_frames - 1) // vae_time_stride + 1

    poses = torch.from_numpy(poses_c2w).float()
    first_inv = get_pose_inverse(poses[0:1]).squeeze(0)
    poses_rel = torch.matmul(first_inv, poses[1:])
    poses = torch.cat([torch.eye(4).unsqueeze(0), poses_rel], dim=0)

    intrinsics = torch.from_numpy(intrinsics_vec4).float()
    intrinsics_latent = intrinsics.clone()
    intrinsics_latent[:, [0, 2]] *= latent_w / float(pixel_w)
    intrinsics_latent[:, [1, 3]] *= latent_h / float(pixel_h)

    time_indices = torch.arange(0, num_frames, vae_time_stride)
    if len(time_indices) > latent_frames:
        time_indices = time_indices[:latent_frames]
    raymap = torch.cat(
        [
            poses[time_indices].reshape(len(time_indices), -1),
            intrinsics_latent[time_indices],
        ],
        dim=-1,
    )

    chunks = []
    for start in time_indices - (vae_time_stride - 1):
        chunk_start = max(0, int(start))
        chunk_end = chunk_start + vae_time_stride
        chunk_poses = poses[chunk_start:chunk_end]
        chunk_intrinsics = intrinsics_latent[chunk_start:chunk_end]
        if chunk_poses.shape[0] < vae_time_stride:
            pad = vae_time_stride - chunk_poses.shape[0]
            chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].repeat(pad, 1, 1)])
            chunk_intrinsics = torch.cat(
                [chunk_intrinsics, chunk_intrinsics[-1:].repeat(pad, 1)]
            )
        plucker = compute_raymap(
            chunk_intrinsics,
            chunk_poses,
            latent_h,
            latent_w,
            use_plucker=True,
        )
        chunks.append(plucker.permute(0, 3, 1, 2).reshape(-1, latent_h, latent_w))
    chunk_plucker = torch.stack(chunks).permute(1, 0, 2, 3)
    return {"raymap": raymap, "chunk_plucker": chunk_plucker}
