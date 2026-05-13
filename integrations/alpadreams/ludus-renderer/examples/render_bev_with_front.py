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

Outputs JPEG files using the renderer's nvJPEG GPU encoder (no CPU round-trip).
"""

import os
import torch

from ludus_renderer import load_clipgt_scene
from ludus_renderer.torch import LudusTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR, CAMERA_TYPE_BEV
from ludus_renderer.render_utils import (
    SceneAdapter, create_camera,
    get_camera_pose, get_bev_camera_pose,
)
from ludus_renderer.util import resample_timestamps

SCENE_PATH = os.path.join(os.path.dirname(__file__), "../example_data/test_hdmap")
FRONT_CAM = "camera:front:wide:120fov"
WIDTH, HEIGHT = 1280, 720
BEV_HEIGHT = 80.0
BEV_FOV = 60.0
JPEG_QUALITY = 90

device = torch.device("cuda")

# Load scene (include_ego_obstacle makes the ego cube visible in BEV)
raw_scene = load_clipgt_scene(SCENE_PATH, device=device, include_ego_obstacle=True)
scene = SceneAdapter(raw_scene)
timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100_000, 20_000_000)
print(f"Scene loaded: {len(timestamps)} frames")

# Setup renderer with both cameras: 0=front (scaled to render res), 1=bev
ctx = LudusTimestampedContext(device=device)
ctx.set_depth_scaling(True)

front_camera = create_camera(WIDTH, HEIGHT, device, scene=scene, camera_name=FRONT_CAM)
bev_camera = create_camera(WIDTH, HEIGHT, device, bev=True, bev_height=BEV_HEIGHT, bev_fov=BEV_FOV)
ctx.upload_cameras([front_camera, bev_camera])

scene_id = ctx.upload_scene(scene.timestamped_scene)

# Pick a frame
frame_idx = min(50, len(timestamps) - 1)
ts = timestamps[frame_idx]
print(f"Rendering frame {frame_idx} (timestamp {ts.item()})")

# Compute poses for both views
front_pose = get_camera_pose(scene, ts, FRONT_CAM, device)
bev_pose = get_bev_camera_pose(scene, ts, BEV_HEIGHT, device)

# Render to staging buffer, then encode to JPEG on the GPU via nvJPEG
queries = ctx.pack_queries_fast(
    scene_ids=torch.tensor([scene_id, scene_id], dtype=torch.int32, device=device),
    camera_ids=torch.tensor([0, 1], dtype=torch.int32, device=device),
    timestamps_us=torch.tensor([ts, ts], dtype=torch.int64, device=device),
    camera_type_ids=torch.tensor([CAMERA_TYPE_REGULAR, CAMERA_TYPE_BEV], dtype=torch.int32, device=device),
)
staging_idx, _ = ctx.render_batch_to_staging(
    queries, torch.stack([front_pose, bev_pose]), (HEIGHT, WIDTH),
)
jpeg_list = ctx.encode_jpeg_batch_staging(staging_idx, JPEG_QUALITY)
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
