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

"""Render a simple top-down 45-degree camera-trajectory visualization video.

Usage:
    From repository root, render from bundled example ``00``:

    .. code-block:: bash

        uv run --package flashdreams-lingbot python integrations/lingbot/lingbot/trajectory_viz.py \
            --example_idx 0 \
            --output_path outputs/lingbot-world-traj-00.mp4

    Render a different built-in example (allowed: ``0, 1, 2, 5``):

    .. code-block:: bash

        uv run --package flashdreams-lingbot python integrations/lingbot/lingbot/trajectory_viz.py \
            --example_idx 5 \
            --output_path outputs/lingbot-world-traj-05.mp4

    Render from explicit ``poses.npy`` / ``intrinsics.npy`` paths:

    .. code-block:: bash

        uv run --package flashdreams-lingbot python integrations/lingbot/lingbot/trajectory_viz.py \
            --poses_path /path/to/poses.npy \
            --intrinsics_path /path/to/intrinsics.npy \
            --output_path outputs/custom-traj.mp4

    Speed up visualization by skipping every other pose:

    .. code-block:: bash

        uv run --package flashdreams-lingbot python integrations/lingbot/lingbot/trajectory_viz.py \
            --example_idx 0 \
            --stride 2 \
            --output_path outputs/lingbot-world-traj-00-fast.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXAMPLE_DATA_ROOT = REPO_ROOT / "assets/example_data/lingbot_world"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs/lingbot_camera_trajectory.mp4"
AVAILABLE_EXAMPLE_IDXS = (0, 1, 2, 5)
DEFAULT_OUTPUT_FPS = 16
"""Default trajectory video FPS, aligned with ``LingbotWorldRunnerConfig.fps``."""
LINE_THICKNESS = 6
"""Line thickness for trajectory and camera-structure rendering."""


def _parse_args() -> argparse.Namespace:
    """Build CLI args for trajectory visualization rendering."""
    parser = argparse.ArgumentParser(
        description=(
            "Render a simple 3D camera trajectory + camera-frustum animation "
            "from Lingbot example poses.npy."
        )
    )
    parser.add_argument(
        "--example_idx",
        type=int,
        default=0,
        choices=AVAILABLE_EXAMPLE_IDXS,
        help="Example folder index under assets/example_data/lingbot_world.",
    )
    parser.add_argument(
        "--example_data_root",
        type=Path,
        default=DEFAULT_EXAMPLE_DATA_ROOT,
        help="Base directory containing example folders (00, 01, 02, 05).",
    )
    parser.add_argument(
        "--poses_path",
        type=Path,
        default=None,
        help="Optional explicit path to poses.npy (overrides example_idx path).",
    )
    parser.add_argument(
        "--intrinsics_path",
        type=Path,
        default=None,
        help="Optional explicit path to intrinsics.npy for frustum aspect ratio.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_OUTPUT_FPS, help="Output video FPS."
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Output video width in pixels."
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Output video height in pixels."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride over trajectory points to shorten output.",
    )
    return parser.parse_args()


def _look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Return a world-to-view rotation matrix from look-at parameters."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)
    return np.stack((right, up, forward), axis=0).astype(np.float32)


def _project_points(
    points_world: np.ndarray,
    *,
    eye: np.ndarray,
    rot_w2v: np.ndarray,
    width: int,
    height: int,
    focal_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into pixel coordinates under a pinhole camera."""
    rel = points_world - eye[None, :]
    points_view = rel @ rot_w2v.T
    z = points_view[:, 2]
    valid = z > 1e-4
    uv = np.full((points_world.shape[0], 2), -1.0, dtype=np.float32)
    uv[valid, 0] = focal_px * (points_view[valid, 0] / z[valid]) + width * 0.5
    uv[valid, 1] = focal_px * (-points_view[valid, 1] / z[valid]) + height * 0.5
    return uv, valid


def _auto_focal_for_target_coverage(
    points_world: np.ndarray,
    *,
    eye: np.ndarray,
    rot_w2v: np.ndarray,
    width: int,
    height: int,
    coverage: float,
) -> float:
    """Compute focal length so projected point range fills target screen coverage."""
    rel = points_world - eye[None, :]
    points_view = rel @ rot_w2v.T
    z = points_view[:, 2]
    valid = z > 1e-4
    if not valid.any():
        return 0.9 * min(width, height)
    nx = points_view[valid, 0] / z[valid]
    ny = points_view[valid, 1] / z[valid]
    # Fit by max absolute offset around the optical center (not by range),
    # so asymmetric perspective does not push one side out of frame.
    max_abs_x = float(max(1e-6, float(np.max(np.abs(nx)))))
    max_abs_y = float(max(1e-6, float(np.max(np.abs(ny)))))
    safety = 0.95
    target_half_w = 0.5 * coverage * float(width) * safety
    target_half_h = 0.5 * coverage * float(height) * safety
    return min(target_half_w / max_abs_x, target_half_h / max_abs_y)


def _build_frustum_points(
    c2w: np.ndarray, *, depth: float, half_width: float, half_height: float
) -> np.ndarray:
    """Build 5 world points (origin + 4 image-plane corners) for camera icon."""
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
        ],
        dtype=np.float32,
    )
    rot = c2w[:3, :3].astype(np.float32)
    trans = c2w[:3, 3].astype(np.float32)
    return local @ rot.T + trans[None, :]


def _normalize_poses_to_first_camera_frame(poses: np.ndarray) -> np.ndarray:
    """Rebase poses so frame 0 camera is at origin and facing canonical forward."""
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), (
        f"Expected poses with shape [T, 4, 4], got {tuple(poses.shape)}."
    )
    world_to_cam0 = np.linalg.inv(poses[0]).astype(np.float32)
    normalized = world_to_cam0[None, :, :] @ poses
    # Visualization convention: keep "start facing forward" while making
    # the rendered world Z-up.
    # cam frame: x=right, y=down, z=forward
    # viz frame: x=forward, y=right, z=up
    cam_to_viz = np.eye(4, dtype=np.float32)
    cam_to_viz[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],  # forward <- +z_cam
            [1.0, 0.0, 0.0],  # right   <- +x_cam
            [0.0, -1.0, 0.0],  # up      <- -y_cam
        ],
        dtype=np.float32,
    )
    normalized = cam_to_viz[None, :, :] @ normalized
    normalized[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return normalized


def _load_intrinsics_aspect(intrinsics_path: Path | None) -> float:
    """Return ``fx / fy`` aspect estimate from intrinsics, or ``1.0`` fallback."""
    if intrinsics_path is None or not intrinsics_path.exists():
        return 1.0
    intr = np.load(intrinsics_path)
    if intr.ndim == 1:
        fx, fy = float(intr[0]), float(intr[1])
    elif intr.ndim == 2:
        fx, fy = float(intr[0, 0]), float(intr[0, 1])
    else:
        return 1.0
    if abs(fy) < 1e-8:
        return 1.0
    return max(0.25, min(4.0, abs(fx / fy)))


def _resolve_example_file(
    *, root: Path, example_idx: int, filename: str, explicit_path: Path | None
) -> Path:
    """Resolve file path from explicit CLI path or example-index folder."""
    if explicit_path is not None:
        return explicit_path
    return root / f"{example_idx:02d}" / filename


def _draw_polyline(
    image: np.ndarray, uv: np.ndarray, valid: np.ndarray, color: tuple[int, int, int]
) -> None:
    """Draw a connected 2D polyline while skipping invalid projected points."""
    for i in range(1, len(uv)):
        if not (valid[i - 1] and valid[i]):
            continue
        p0 = tuple(np.round(uv[i - 1]).astype(np.int32))
        p1 = tuple(np.round(uv[i]).astype(np.int32))
        cv2.line(image, p0, p1, color, LINE_THICKNESS, cv2.LINE_AA)


def _open_video_writer(
    *, output_path: Path, fps: int, width: int, height: int
) -> tuple[cv2.VideoWriter, Path | None]:
    """Open an OpenCV ``mp4v`` writer into a temp file for later H.264 transcode."""
    fd, tmp_name = tempfile.mkstemp(
        prefix=".trajectory_viz_", suffix=".mp4", dir=output_path.parent
    )
    # Close immediately; OpenCV re-opens the path for streaming writes.
    # ``mkstemp`` is only used for a collision-resistant filename.
    os.close(fd)
    Path(tmp_name).unlink(missing_ok=True)
    temp_mp4v_path = Path(tmp_name)
    writer = cv2.VideoWriter(
        str(temp_mp4v_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    assert writer.isOpened(), f"Failed to open mp4v writer for {output_path}."
    return writer, temp_mp4v_path


def _transcode_mp4v_to_h264(*, src_path: Path, dst_path: Path, fps: int) -> None:
    """Transcode ``src_path`` to H.264 ``dst_path`` using ``imageio-ffmpeg``."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "OpenCV H.264 writer is unavailable and fallback transcoding needs "
            "imageio-ffmpeg. Install it with uv/pip and rerun."
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src_path),
        "-an",
        "-c:v",
        "libx264",
        "-r",
        str(float(fps)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to transcode {src_path} to H.264 {dst_path}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def main() -> None:
    """Render and save a trajectory MP4 from Lingbot ``poses.npy``."""
    args = _parse_args()
    assert args.stride > 0, "--stride must be >= 1."
    poses_path = _resolve_example_file(
        root=args.example_data_root,
        example_idx=args.example_idx,
        filename="poses.npy",
        explicit_path=args.poses_path,
    )
    intrinsics_path = _resolve_example_file(
        root=args.example_data_root,
        example_idx=args.example_idx,
        filename="intrinsics.npy",
        explicit_path=args.intrinsics_path,
    )

    poses = np.load(poses_path).astype(np.float32)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), (
        f"Expected poses.npy with shape [T, 4, 4], got {tuple(poses.shape)}."
    )
    poses = poses[:: args.stride]
    poses = _normalize_poses_to_first_camera_frame(poses)
    camera_positions = poses[:, :3, 3]

    mins = camera_positions.min(axis=0)
    maxs = camera_positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    diag = np.linalg.norm(maxs - mins)
    scene_scale = float(max(1e-4, float(diag)))

    # Fixed observer camera: roughly top-down, 45-degree perspective.
    eye = center + scene_scale * np.array([1.4, -1.4, 1.4], dtype=np.float32)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rot_w2v = _look_at(eye, center, up_hint)

    focal_px = _auto_focal_for_target_coverage(
        camera_positions,
        eye=eye,
        rot_w2v=rot_w2v,
        width=args.width,
        height=args.height,
        coverage=0.90,
    )
    frustum_depth = 0.12 * scene_scale
    frustum_half_h = 0.06 * scene_scale
    intrinsics_aspect = _load_intrinsics_aspect(intrinsics_path)
    frustum_half_w = frustum_half_h * intrinsics_aspect

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer, temp_mp4v_path = _open_video_writer(
        output_path=args.output_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    try:
        traj_uv, traj_valid = _project_points(
            camera_positions,
            eye=eye,
            rot_w2v=rot_w2v,
            width=args.width,
            height=args.height,
            focal_px=focal_px,
        )
        for t in range(camera_positions.shape[0]):
            frame = np.full((args.height, args.width, 3), 18, dtype=np.uint8)

            # Full trajectory (gray) + traversed prefix (green).
            _draw_polyline(frame, traj_uv, traj_valid, (90, 90, 90))
            _draw_polyline(frame, traj_uv[: t + 1], traj_valid[: t + 1], (50, 220, 100))

            # Current camera icon as a wireframe frustum.
            frustum_world = _build_frustum_points(
                poses[t],
                depth=frustum_depth,
                half_width=frustum_half_w,
                half_height=frustum_half_h,
            )
            frustum_uv, frustum_valid = _project_points(
                frustum_world,
                eye=eye,
                rot_w2v=rot_w2v,
                width=args.width,
                height=args.height,
                focal_px=focal_px,
            )
            if frustum_valid.all():
                edges = (
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 1),
                )
                for a, b in edges:
                    p0 = tuple(np.round(frustum_uv[a]).astype(np.int32))
                    p1 = tuple(np.round(frustum_uv[b]).astype(np.int32))
                    cv2.line(frame, p0, p1, (70, 170, 255), LINE_THICKNESS, cv2.LINE_AA)

            if traj_valid[t]:
                cur = tuple(np.round(traj_uv[t]).astype(np.int32))
                cv2.circle(frame, cur, 6, (0, 255, 255), -1, cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()

    if temp_mp4v_path is not None:
        try:
            _transcode_mp4v_to_h264(
                src_path=temp_mp4v_path, dst_path=args.output_path, fps=args.fps
            )
        finally:
            temp_mp4v_path.unlink(missing_ok=True)

    print(f"Wrote trajectory visualization video: {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
