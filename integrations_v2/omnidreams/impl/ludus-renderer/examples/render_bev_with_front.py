#!/usr/bin/env python3
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

"""Minimal example: render front camera + BEV in a single batched call.

Outputs JPEG files using nvJPEG GPU encoder (no CPU round-trip for encoding).
"""

import argparse
import os

import torch
from ludus_renderer import load_scene, nvjpeg
from ludus_renderer.render_utils import (
    SceneAdapter,
    create_camera,
    get_bev_camera_pose,
    get_camera_pose,
)
from ludus_renderer.torch import LudusCudaTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_BEV, CAMERA_TYPE_REGULAR
from ludus_renderer.util import resample_timestamps

DEFAULT_SCENE = os.path.join(os.path.dirname(__file__), "../example_data/test_hdmap")
FRONT_CAM = "camera:front:wide:120fov"
WIDTH, HEIGHT = 1280, 720
BEV_HEIGHT = 80.0
BEV_FOV = 60.0
JPEG_QUALITY = 90


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        nargs="?",
        default=DEFAULT_SCENE,
        help="Scene directory, archive, or AV2 tar to render",
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load scene (include_ego_obstacle makes the ego cube visible in BEV)
    raw_scene = load_scene(args.scene, device=device, include_ego_obstacle=True)
    scene = SceneAdapter(raw_scene)
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100_000, 20_000_000)
    print(f"Scene loaded: {len(timestamps)} frames")

    # Setup renderer with both cameras: 0=front (scaled to render res), 1=bev
    ctx = LudusCudaTimestampedContext(device=device)
    ctx.set_depth_scaling(True)

    front_camera = create_camera(
        WIDTH, HEIGHT, device, scene=scene, camera_name=FRONT_CAM
    )
    bev_camera = create_camera(
        WIDTH,
        HEIGHT,
        device,
        bev=True,
        bev_height=BEV_HEIGHT,
        bev_fov=BEV_FOV,
    )
    ctx.upload_cameras([front_camera, bev_camera])

    scene_id = ctx.upload_scene(scene.timestamped_scene)

    # Pick a frame
    frame_idx = min(50, len(timestamps) - 1)
    ts = timestamps[frame_idx]
    print(f"Rendering frame {frame_idx} (timestamp {ts.item()})")

    # Compute poses for both views
    front_pose = get_camera_pose(scene, ts, FRONT_CAM, device)
    bev_pose = get_bev_camera_pose(scene, ts, BEV_HEIGHT, device)

    # Render both views in a single batched call
    scene_ids = torch.tensor([scene_id, scene_id], dtype=torch.int32, device=device)
    camera_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    timestamps_us = torch.tensor([ts, ts], dtype=torch.int64, device=device)
    camera_type_ids = torch.tensor(
        [CAMERA_TYPE_REGULAR, CAMERA_TYPE_BEV], dtype=torch.int32, device=device
    )
    poses = torch.stack([front_pose, bev_pose])

    images = ctx.render(
        scene_ids,
        camera_ids,
        timestamps_us,
        camera_type_ids,
        poses,
        resolution=(HEIGHT, WIDTH),
    )

    # Encode to JPEG on GPU via nvJPEG
    if not nvjpeg.is_available():
        print("WARNING: nvJPEG not available, falling back to PIL")
        from PIL import Image as PILImage

        output_dir = os.path.join(os.path.dirname(__file__), "../_images")
        os.makedirs(output_dir, exist_ok=True)
        names = ["example_front.png", "example_bev.png"]
        for i, name in enumerate(names):
            img_np = images[i, :, :, :3].cpu().numpy()
            PILImage.fromarray(img_np).save(os.path.join(output_dir, name))
        print(f"Saved: _images/{names[0]}, _images/{names[1]}")
    else:
        # Convert [N, H, W, 4] RGBA -> [N, 3, H, W] RGB for nvjpeg
        images_rgb = images[:, :, :, :3].permute(0, 3, 1, 2).contiguous()
        jpeg_list = nvjpeg.encode(images_rgb, quality=JPEG_QUALITY)
        print(f"Encoded {len(jpeg_list)} JPEG images on GPU (quality={JPEG_QUALITY})")

        # Write JPEG bytes to disk
        output_dir = os.path.join(os.path.dirname(__file__), "../_images")
        os.makedirs(output_dir, exist_ok=True)

        names = ["example_front.jpg", "example_bev.jpg"]
        for jpeg_bytes, name in zip(jpeg_list, names):
            path = os.path.join(output_dir, name)
            with open(path, "wb") as f:
                f.write(jpeg_bytes)

        print(f"Saved: _images/{names[0]}, _images/{names[1]}")


if __name__ == "__main__":
    main()
