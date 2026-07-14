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

"""Example: render a mirror-augmented scene as a BEV MP4.

Mirror augmentation reflects the ego trajectory and all scene geometry
at the scene boundary, producing a longer looping sequence from a single
clip.  This example augments a scene with 2 mirrors (3x length), then
renders the full BEV sequence to an MP4.

Usage:
    python render_mirror_augmented.py
    python render_mirror_augmented.py --scene <path> --n-mirrors 4
"""

import argparse
import os
import time

import torch
from ludus_renderer import load_scene
from ludus_renderer.augmentation import mirror_augment_scene
from ludus_renderer.render_utils import (
    SceneAdapter,
    compute_camera_poses,
    create_camera,
)
from ludus_renderer.torch import LudusCudaTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_BEV
from ludus_renderer.util import resample_timestamps

DEFAULT_SCENE = os.path.join(os.path.dirname(__file__), "../example_data/test_hdmap")


def main():
    parser = argparse.ArgumentParser(description="Mirror augmentation example")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE)
    parser.add_argument(
        "--n-mirrors",
        type=int,
        default=2,
        help="Number of mirror reflections (default: 2)",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=50.0,
        help="Ego extrapolation before mirror plane (metres)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--bev-height", type=float, default=80.0)
    parser.add_argument("--bev-fov", type=float, default=60.0)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load and augment
    print(f"Loading scene: {args.scene}")
    raw_scene = load_scene(
        args.scene,
        device=device,
        include_ego_trajectory=True,
        include_ego_obstacle=True,
    )
    orig_poses = len(raw_scene.ego_track.timestamps)

    print(f"Augmenting with {args.n_mirrors} mirrors (lookahead={args.lookahead}m)...")
    aug_scene = mirror_augment_scene(
        raw_scene, n_mirrors=args.n_mirrors, lookahead_m=args.lookahead
    )
    scene = SceneAdapter(aug_scene)
    aug_poses = len(aug_scene.ego_track.timestamps)
    print(f"  Ego poses: {orig_poses} -> {aug_poses} ({aug_poses / orig_poses:.1f}x)")

    # Timestamps
    timestep_us = 1_000_000 // args.fps
    duration_us = (
        scene.ego_tracks.timestamps[-1] - scene.ego_tracks.timestamps[0]
    ).item()
    timestamps = resample_timestamps(
        scene.ego_tracks.timestamps, timestep_us, duration_us
    )
    n_frames = len(timestamps)
    print(f"  {n_frames} frames at {args.fps} Hz, duration {duration_us / 1e6:.1f}s")

    # Renderer setup
    ctx = LudusCudaTimestampedContext(device=device)
    ctx.set_depth_scaling(True)

    camera = create_camera(
        args.width,
        args.height,
        device,
        bev=True,
        bev_height=args.bev_height,
        bev_fov=args.bev_fov,
    )
    ctx.upload_cameras([camera])
    scene_id = ctx.upload_scene(scene.timestamped_scene)

    poses, ctype = compute_camera_poses(
        scene, timestamps, device, bev_height=args.bev_height
    )
    if poses.dim() == 4:
        poses = poses.squeeze(1)

    # Render + encode MP4
    import subprocess
    import tempfile

    import PyNvVideoCodec as nvc  # ty:ignore[unresolved-import]

    output_path = os.path.join(
        os.path.dirname(__file__), f"../_images/mirror_augmented_bev.mp4"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    raw_h264 = tempfile.NamedTemporaryFile(suffix=".h264", delete=False)
    encoder = nvc.CreateEncoder(
        width=args.width,
        height=args.height,
        fmt="ABGR",
        usecpuinputbuffer=False,
        codec="h264",
        bitrate=10_000_000,
        fps=args.fps,
        preset="P4",
    )

    print(f"Rendering {n_frames} frames...")
    t0 = time.time()
    batch_size = 64

    for i in range(0, n_frames, batch_size):
        end = min(i + batch_size, n_frames)
        n = end - i

        images = ctx.render(
            torch.full((n,), scene_id, dtype=torch.int32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device),
            timestamps[i:end].to(torch.int64),
            torch.full((n,), ctype, dtype=torch.int32, device=device),
            poses[i:end],
            resolution=(args.height, args.width),
        )

        for j in range(n):
            frame = images[j].flip(0).contiguous()
            bs = encoder.Encode(frame)
            if bs:
                raw_h264.write(bs)

    bs = encoder.EndEncode()
    if bs:
        raw_h264.write(bs)
    raw_h264.close()

    elapsed = time.time() - t0
    print(f"  Rendered in {elapsed:.1f}s ({n_frames / elapsed:.0f} FPS)")

    # Mux to MP4
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-framerate",
            str(args.fps),
            "-i",
            raw_h264.name,
            "-c",
            "copy",
            output_path,
        ],
        check=True,
    )
    os.remove(raw_h264.name)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
