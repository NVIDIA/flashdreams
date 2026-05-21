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

"""Test rasterizer determinism by rendering the same sequence N times and comparing GPU tensors."""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ludus_renderer.render_utils import (
    compute_camera_poses,
    create_camera,
)
from ludus_renderer.render_utils import (
    load_scene_adapted as load_scene,
)
from ludus_renderer.torch import LudusCudaTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR
from ludus_renderer.util import resample_timestamps

SCENE_PATH = os.path.join(os.path.dirname(__file__), "../example_data/test_hdmap")
CAMERA_NAME = "camera:front:wide:120fov"
WIDTH, HEIGHT = 1280, 720
FPS = 30


def render_sequence(ctx, scene_id, timestamps, poses, device):
    """Render full sequence, return list of GPU image tensors."""
    n_frames = len(timestamps)
    scene_ids = torch.full((n_frames,), scene_id, dtype=torch.int32, device=device)
    camera_ids = torch.zeros(n_frames, dtype=torch.int32, device=device)
    camera_type_ids = torch.full(
        (n_frames,), CAMERA_TYPE_REGULAR, dtype=torch.int32, device=device
    )
    ts = timestamps.to(torch.int64)

    images = ctx.render(
        scene_ids, camera_ids, ts, camera_type_ids, poses, resolution=(HEIGHT, WIDTH)
    )
    torch.cuda.synchronize()
    return images[:, :, :, :3].contiguous()


def main():
    parser = argparse.ArgumentParser(description="Test rasterizer determinism")
    parser.add_argument(
        "-n",
        "--n-runs",
        type=int,
        default=1,
        help="Number of comparison passes against the reference (default: 1)",
    )
    parser.add_argument(
        "--dump-ref",
        type=str,
        default=None,
        metavar="DIR",
        help="Dump reference frames as PNGs to this directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    print("Loading scene...")
    scene = load_scene(SCENE_PATH, device)

    timestep_us = 1_000_000 // FPS
    duration_us = (
        scene.ego_tracks.timestamps[-1] - scene.ego_tracks.timestamps[0]
    ).item()
    timestamps = resample_timestamps(
        scene.ego_tracks.timestamps, timestep_us, duration_us
    )
    n_frames = len(timestamps)
    print(f"  {n_frames} frames at {FPS}Hz")

    ctx = LudusCudaTimestampedContext(device=device)
    ctx.set_depth_scaling(True)
    camera = create_camera(WIDTH, HEIGHT, device, scene=scene, camera_name=CAMERA_NAME)
    ctx.upload_cameras([camera])
    scene_id = ctx.upload_scene(scene.timestamped_scene)

    poses, _ = compute_camera_poses(scene, timestamps, device, camera_name=CAMERA_NAME)

    # --- Reference render ---
    print("\nRender pass 1 (reference)...")
    t0 = time.time()
    ref = render_sequence(ctx, scene_id, timestamps, poses, device)
    t1 = time.time()
    print(f"  Done in {t1 - t0:.2f}s  ({n_frames / (t1 - t0):.0f} FPS)")

    if args.dump_ref:
        os.makedirs(args.dump_ref, exist_ok=True)
        ref_cpu = ref.cpu().numpy()
        for i in range(n_frames):
            Image.fromarray(ref_cpu[i]).save(
                os.path.join(args.dump_ref, f"frame_{i:04d}.png")
            )
        print(f"  Saved {n_frames} reference frames to {args.dump_ref}/")

    total_pixels = n_frames * HEIGHT * WIDTH
    all_pass = True

    diff_dir = os.path.join(os.path.dirname(__file__), "../_images/determinism_diffs")
    os.makedirs(diff_dir, exist_ok=True)

    for run in range(args.n_runs):
        label = f"pass {run + 2}" if args.n_runs > 1 else "pass 2"
        print(f"\nRender {label}...")
        t0 = time.time()
        imgs = render_sequence(ctx, scene_id, timestamps, poses, device)
        t1 = time.time()
        print(f"  Done in {t1 - t0:.2f}s  ({n_frames / (t1 - t0):.0f} FPS)")

        diff_mask = (ref != imgs).any(dim=-1)
        diff_per_frame = diff_mask.sum(dim=(1, 2))
        total_diff_pixels = diff_per_frame.sum().item()

        if total_diff_pixels == 0:
            print(
                f"  PIXEL-PERFECT: 0 diffs across {n_frames} frames ({total_pixels:,} pixels)"
            )
        else:
            all_pass = False
            bad_indices = (diff_per_frame > 0).nonzero(as_tuple=True)[0]
            print(
                f"  FAILED: {len(bad_indices)}/{n_frames} frames have diffs, {total_diff_pixels} total diff pixels"
            )

            if run == 0:
                print(f"  Saving diff visualizations to {diff_dir}/")
                ref_cpu = ref.cpu().numpy()
                imgs_cpu = imgs.cpu().numpy()
                mask_cpu = diff_mask.cpu().numpy()

                for idx in bad_indices:
                    i = idx.item()
                    n_diff = diff_per_frame[i].item()
                    Image.fromarray(ref_cpu[i]).save(
                        os.path.join(diff_dir, f"frame_{i:04d}_ref.png")
                    )

                    pass2_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    pass2_vis[mask_cpu[i]] = imgs_cpu[i][mask_cpu[i]]
                    Image.fromarray(pass2_vis).save(
                        os.path.join(diff_dir, f"frame_{i:04d}_pass2.png")
                    )

                    diff_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    diff_vis[mask_cpu[i]] = 255
                    Image.fromarray(diff_vis).save(
                        os.path.join(diff_dir, f"frame_{i:04d}_diff.png")
                    )
                    print(f"    frame_{i:04d}: {n_diff} diff pixels")

    if all_pass:
        print(f"\nALL {args.n_runs} RUNS PIXEL-PERFECT")
    else:
        print(f"\nSOME RUNS HAD DIFFS")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
