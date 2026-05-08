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

"""Render a short HD-map preview using the SlangPy rasterizer.

Standalone script for verifying that the new SlangPy-based
:class:`LudusRenderer` produces sensible HD-map conditioning frames. Runs
the rasterizer directly (no gRPC server, no video model), feeds it a
CLIPGT scene + the ego trajectory as camera poses, and writes the result
out as an MP4 + the first frame as a PNG so you can eyeball it.

Example::

    # Inside the docker container, after running `bash assets/download.sh`
    uv run --package flash-alpadreams \\
      python integrations/alpadreams/scripts/render_hdmap_preview.py

    # With explicit args
    uv run --package flash-alpadreams \\
      python integrations/alpadreams/scripts/render_hdmap_preview.py \\
        --scene assets/example_data/alpadreams/clipgt.zip \\
        --output outputs/hdmap_preview.mp4 \\
        --frames 90 --stride 5 --fps 15

The first run is slow because SlangPy compiles the kernels on demand;
subsequent runs hit the shader cache.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import mediapy
import numpy as np
import torch
from alpadreams.conditioning.renderer import LudusRenderer, load_and_attach_ludus_scene
from alpadreams.conditioning.world_scenario.data_loaders import load_scene
from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
from scipy.spatial.transform import Rotation

# flashdreams' ClipGTLoader stores ego poses with `double_sided=True`
# basis change, so both the local frame and the world frame are OpenCV
# RDF. The slang shader also expects RDF (camera +x=right, +y=down,
# +z=forward). So we can pass ego_to_world straight through as
# camera_to_world -- no rig_to_camera basis change is needed for this
# script's synthetic pinhole camera.

DEFAULT_SCENE = Path("assets/example_data/alpadreams/clipgt.zip")
DEFAULT_OUTPUT = Path("outputs/hdmap_preview.mp4")
DEFAULT_CAMERA = "camera_front_wide_120fov"

# Mount the synthetic camera 1.5 m above the rig origin (typical dashcam
# height). Without this offset the camera ends up at the rig's local
# y=0 -- which in CLIPGT's RDF data is essentially "vehicle-origin
# altitude", *not* "1.5 m above the road surface". Real road geometry
# (lane lines, crosswalks, road islands, ...) sits at y ≈ 0 too, so
# without the offset the camera is at the same vertical level as the
# road and ground polygons collapse into thin horizontal slices that
# fan out across the screen.
_CAMERA_HEIGHT_M = 1.5


def _make_default_camera(width: int, height: int) -> FThetaCamera:
    """Build a 120-FOV F-theta camera at the requested framebuffer size."""
    f = width / (2.0 * np.radians(60.0))
    intrinsics = np.array(
        [width / 2.0, height / 2.0, width, height, f, 0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return FThetaCamera.from_numpy(intrinsics)


def _ego_poses_to_camera_matrices(
    ego_poses: list, count: int, stride: int, camera_height_m: float
) -> tuple[np.ndarray, list[int]]:
    """Sample every Nth ego pose and convert to camera-to-world matrices.

    flashdreams' loader stores ego poses in RDF (with double-sided basis
    change), so we use them directly as ``camera_to_world``. We bake a
    ``rig_to_camera`` translation of (0, -camera_height_m, 0) -- minus
    Y in RDF means UP -- to lift the camera off the rig origin.
    """
    selected = ego_poses[: count * stride : stride]
    if len(selected) == 0:
        raise RuntimeError(
            f"Scene has no ego poses (or stride {stride} skipped them all). "
            f"Got {len(ego_poses)} total poses."
        )
    rig_to_camera = np.eye(4, dtype=np.float32)
    rig_to_camera[:3, 3] = np.array([0.0, -camera_height_m, 0.0], dtype=np.float32)

    matrices = []
    timestamps: list[int] = []
    for ego in selected:
        rot = Rotation.from_quat(ego.orientation).as_matrix().astype(np.float32)
        rig_to_world = np.eye(4, dtype=np.float32)
        rig_to_world[:3, :3] = rot
        rig_to_world[:3, 3] = np.asarray(ego.position, dtype=np.float32)
        camera_to_world = rig_to_world @ rig_to_camera
        matrices.append(camera_to_world)
        timestamps.append(int(ego.timestamp))
    return np.stack(matrices, axis=0).astype(np.float32), timestamps


def _open_scene(scene_path: Path) -> tuple[Path, "tempfile.TemporaryDirectory[str] | None"]:
    """If ``scene_path`` is a zip, extract to a temp dir; otherwise return it as-is."""
    if scene_path.is_dir():
        return scene_path, None
    if scene_path.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        extract_dir = Path(tmp.name) / "scene"
        extract_dir.mkdir()
        with zipfile.ZipFile(scene_path) as zf:
            zf.extractall(extract_dir)
        children = list(extract_dir.iterdir())
        # Some clipgt zips wrap their contents in a single subdirectory.
        data_path = children[0] if len(children) == 1 and children[0].is_dir() else extract_dir
        return data_path, tmp
    raise FileNotFoundError(f"Scene path is neither a directory nor a zip: {scene_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                        help=f"CLIPGT zip or directory (default: {DEFAULT_SCENE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"MP4 output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA,
                        help="Camera name passed to load_scene")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--frames", type=int, default=60,
                        help="How many frames to render")
    parser.add_argument("--stride", type=int, default=10,
                        help="Subsample the ego trajectory by this stride")
    parser.add_argument("--fps", type=int, default=10,
                        help="Output video FPS")
    parser.add_argument("--camera-height-m", type=float, default=_CAMERA_HEIGHT_M,
                        help=f"Camera mount height above the rig origin (default: {_CAMERA_HEIGHT_M})")
    args = parser.parse_args()

    if not args.scene.exists():
        raise SystemExit(
            f"Scene not found at {args.scene}. Run `bash assets/download.sh` "
            "to fetch the bundled example, or pass --scene explicitly."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the SlangPy rasterizer renders on GPU).")

    device = torch.device("cuda")
    data_path, tmp_holder = _open_scene(args.scene)
    try:
        scene_data = load_scene(
            data_path,
            camera_names=[args.camera_name],
            max_frames=-1,
            input_pose_fps=30,
            resize_resolution_hw=[args.height, args.width],
        )
        scene_data = load_and_attach_ludus_scene(data_path, scene_data, device=device)

        print(f"Scene id            : {scene_data.scene_id}")
        print(f"Ego poses           : {len(scene_data.ego_poses)}")
        print(f"Lane lines          : {len(scene_data.lane_lines)}")
        print(f"Road boundaries    : {len(scene_data.road_boundaries)}")
        print(f"Crosswalks          : {len(scene_data.crosswalks)}")
        print(f"Road markings       : {len(scene_data.road_markings)}")
        print(f"Wait lines          : {len(scene_data.wait_lines)}")
        print(f"Poles               : {len(scene_data.poles)}")
        print(f"Traffic signs       : {len(scene_data.traffic_signs)}")
        print(f"Traffic lights      : {len(scene_data.traffic_lights)}")
        print(f"Intersection areas : {len(scene_data.intersection_areas)}")
        print(f"Road islands        : {len(scene_data.road_islands)}")

        # Prefer the real camera calibration from the scene if present;
        # the loader populates ``scene_data.camera_models`` with intrinsics
        # parsed from the CLIPGT calibration parquet (right cx/cy, full
        # F-theta polynomial, real lens distortion, real sensor_to_rig
        # transform). Falling back to a synthetic 120 FOV pinhole
        # would mismatch every CLIPGT-recorded frame and especially
        # break for fisheye / wide-angle automotive cameras.
        loaded_camera = scene_data.camera_models.get(args.camera_name)
        if isinstance(loaded_camera, FThetaCamera):
            print(
                f"Using calibrated camera '{args.camera_name}' from scene"
                f" (width={loaded_camera.width}, height={loaded_camera.height})"
            )
            if loaded_camera.height != args.height or loaded_camera.width != args.width:
                scale_h = args.height / loaded_camera.height
                scale_w = args.width / loaded_camera.width
                loaded_camera = FThetaCamera.from_numpy(loaded_camera.intrinsics.copy())
                loaded_camera.rescale(ratio_h=scale_h, ratio_w=scale_w)
            camera = loaded_camera
        else:
            print("Scene has no calibrated camera; using synthetic 120 FOV pinhole.")
            camera = _make_default_camera(args.width, args.height)
        renderer = LudusRenderer(
            scene_data=scene_data,
            camera_models={args.camera_name: camera},
            device=device,
        )

        coordinate_frame = str(scene_data.metadata.get("coordinate_frame", "flu"))
        print(f"Coordinate frame    : {coordinate_frame}")
        print(f"Camera mount height : {args.camera_height_m} m above rig origin")
        camera_poses_np, timestamps = _ego_poses_to_camera_matrices(
            scene_data.ego_poses,
            count=args.frames,
            stride=args.stride,
            camera_height_m=args.camera_height_m,
        )
        # Print the bounding range of every layer's vertices and the
        # first ego pose. With the loader storing everything in OpenCV
        # RDF, expect: x range = horizontal extent (right is +), y range
        # ~= 0 (ground) for road geometry / negative for ego (above
        # ground), z range = forward extent.
        ego0 = scene_data.ego_poses[0]
        print(f"Ego[0] position rdf: {tuple(ego0.position.tolist())}")
        print(f"Ego[0] orientation : {tuple(ego0.orientation.tolist())} (xyzw)")
        forward_axis_world = (
            Rotation.from_quat(ego0.orientation).as_matrix()[:, 2].tolist()
        )
        print(f"Ego[0] +Z (forward): {forward_axis_world}")
        print(f"Camera[0] world pos: ({camera_poses_np[0, 0, 3]:.3f}, "
              f"{camera_poses_np[0, 1, 3]:.3f}, {camera_poses_np[0, 2, 3]:.3f})")
        all_y = []
        if scene_data.lane_lines:
            for ll in scene_data.lane_lines[:5]:
                if ll.points.size:
                    all_y.append(("laneline", ll.points[:, 1].min(), ll.points[:, 1].max()))
        if scene_data.crosswalks:
            for cw in scene_data.crosswalks[:5]:
                if cw.vertices.size:
                    all_y.append(("crosswalk", cw.vertices[:, 1].min(), cw.vertices[:, 1].max()))
        if scene_data.road_boundaries:
            for rb in scene_data.road_boundaries[:5]:
                if rb.points.size:
                    all_y.append(("road_bound", rb.points[:, 1].min(), rb.points[:, 1].max()))
        for name, ymin, ymax in all_y:
            print(f"  {name:12s} y range = [{ymin:.2f}, {ymax:.2f}]")
        print(f"Rendering {len(timestamps)} frames at {args.width}x{args.height}...")
        camera_poses = torch.from_numpy(camera_poses_np).to(device)

        rendered = renderer.render_all_frames_and_cameras(
            camera_names=[args.camera_name],
            camera_poses_per_camera={args.camera_name: camera_poses},
            frame_timestamps_us=timestamps,
        )
        # rendered shape: [V, T, 3, H, W] uint8 on CUDA.
        frames_hwc = rendered[0].permute(0, 2, 3, 1).cpu().numpy()
        print(f"Rendered tensor: shape={tuple(rendered.shape)} dtype={rendered.dtype}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(args.output, frames_hwc, fps=args.fps)
        print(f"Wrote video    : {args.output}")

        png_path = args.output.with_suffix(".png")
        mediapy.write_image(png_path, frames_hwc[0])
        print(f"Wrote first frame: {png_path}")
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()


if __name__ == "__main__":
    main()
