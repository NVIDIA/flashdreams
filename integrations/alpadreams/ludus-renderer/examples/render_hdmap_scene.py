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

"""
HDMap Scene Renderer Example

Renders clipgt HDMap scenes using the Ludus GPU renderer.

Scene elements:
- Ego trajectory (green polyline)
- Road boundaries (magenta polylines)  
- Lane lines (cyan/yellow, solid/dashed/dotted)
- Crosswalks (purple polygons)
- Road markings, wait lines
- Traffic lights, traffic signs, poles
- Dynamic obstacles (blue cubes, wireframe)
- BEV ego obstacle (red/green cube, BEV only)

Usage:
    python render_hdmap_scene.py --scene <path>              # Render frame 0
    python render_hdmap_scene.py --scene <path> --frame 100  # Render frame 100
    python render_hdmap_scene.py --scene <path> --sequence   # Render full sequence
    python render_hdmap_scene.py --scene <path> --bev        # Bird's eye view

For benchmarking, use benchmark_renderer.py instead.
"""

import os
import sys
import argparse
import time

import torch
from PIL import Image

# Default scene path
DEFAULT_SCENE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../example_data/test_hdmap'
)


from ludus_renderer.render_utils import (
    load_scene_adapted as load_scene,
    create_camera, get_available_cameras,
)


DEFAULT_CAMERA = 'camera:front:wide:120fov'


def _get_gpu_decoder_flag(args):
    """Derive use_gpu_decoder from --loader flag."""
    loader = getattr(args, 'loader', 'gpu')
    return loader == 'gpu'


def _create_context(args, device):
    """Create the rendering context (CUDA software rasterizer)."""
    from ludus_renderer.torch import LudusCudaTimestampedContext
    print("Render backend: CUDA software rasterizer (CudaRaster)")
    ctx = LudusCudaTimestampedContext(device=device)
    msaa = getattr(args, 'msaa', 0)
    if msaa > 0:
        ctx.set_msaa_samples(msaa)
        print(f"  MSAA: {msaa}x")
    return ctx


def render_single_frame(args):
    """Render a single frame."""
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.render_utils import render_frame
    
    device = torch.device('cuda')
    
    print(f"Loading scene: {args.scene}")
    include_ego_traj = getattr(args, 'ego_trajectory', False)
    scene = load_scene(args.scene, device, include_ego_obstacle=args.bev,
                       include_ego_trajectory=include_ego_traj,
                       use_gpu_decoder=_get_gpu_decoder_flag(args))
    
    # Resample timestamps to 10Hz
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
    print(f"Timestamps: {len(timestamps)} frames at 10Hz")
    
    if args.frame >= len(timestamps):
        print(f"Error: Frame {args.frame} out of range (max {len(timestamps) - 1})")
        return
    
    camera_name = args.camera or DEFAULT_CAMERA
    camera_mode = "BEV" if args.bev else camera_name
    print(f"Rendering frame {args.frame} (timestamp {timestamps[args.frame].item()}) with {camera_mode}")
    
    # Create context
    ctx = _create_context(args, device)
    ctx.set_depth_scaling(True)
    
    # Create and upload camera (use scene camera scaled to target resolution)
    width, height = args.width, args.height
    camera = create_camera(width, height, device, bev=args.bev, 
                          bev_height=args.bev_height, bev_fov=args.bev_fov,
                          scene=scene, camera_name=camera_name)
    ctx.upload_cameras([camera])
    
    # Upload scene
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    
    # Render
    bev_height = args.bev_height if args.bev else None
    img = render_frame(ctx, scene, scene_id, timestamps, args.frame, 
                       width, height, device, bev_height=bev_height,
                       camera_name=camera_name)
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "../_images")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_bev" if args.bev else ""
    # Include camera name in filename for multi-camera renders
    cam_suffix = f"_{camera_name.replace(':', '_')}" if getattr(args, 'all_cameras', False) else ""
    output_path = os.path.join(output_dir, f"hdmap_scene_{args.frame:04d}{suffix}{cam_suffix}.png")
    img.save(output_path)
    print(f"Saved: {output_path}")


def render_sequence(args, all_cameras: bool = False):
    """Render a full sequence for one or more cameras.
    
    Args:
        args: Command line arguments
        all_cameras: If True, render all available cameras. Otherwise uses args.camera.
    
    Supports output formats:
    - png: GPU -> CPU transfer -> PNG files (default)
    - jpg: GPU -> nvJPEG hardware encode -> JPG files
    - mp4: GPU -> ffmpeg libx264 software encode -> MP4 per camera
    """
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.render_utils import compute_camera_poses
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR, CAMERA_TYPE_BEV
    
    device = torch.device('cuda')
    output_format = args.output_format
    
    print(f"Loading scene: {args.scene}")
    include_ego_traj = getattr(args, 'ego_trajectory', False)
    scene = load_scene(args.scene, device, include_ego_obstacle=args.bev,
                       include_ego_trajectory=include_ego_traj,
                       use_gpu_decoder=_get_gpu_decoder_flag(args))
    
    # Determine cameras to render
    if all_cameras:
        camera_names = get_available_cameras(scene)
        print(f"Using all {len(camera_names)} cameras")
    elif args.bev:
        camera_names = ['bev']
    else:
        camera_names = [args.camera or DEFAULT_CAMERA]
    n_cameras = len(camera_names)
    
    # Resample timestamps
    fps = args.fps or 10
    timestep_us = 1000000 // fps  # microseconds per frame
    duration_us = (scene.ego_tracks.timestamps[-1] - scene.ego_tracks.timestamps[0]).item()
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, timestep_us, duration_us)
    n_frames = len(timestamps)
    total_images = n_frames * n_cameras
    
    camera_desc = camera_names[0] if n_cameras == 1 else f"{n_cameras} cameras"
    print(f"Rendering {n_frames} frames x {camera_desc} = {total_images} images")
    print(f"  Resolution: {args.width}x{args.height}, fps={fps}, format={output_format.upper()}")
    
    # Create context
    ctx = _create_context(args, device)
    ctx.set_depth_scaling(True)

    max_batch = ctx.max_batch_size
    batch_size = args.batch_size if args.batch_size else n_frames
    if batch_size > max_batch:
        print(f"  Clamping batch_size {batch_size} -> {max_batch}")
        batch_size = max_batch
    
    # Create and upload cameras
    width, height = args.width, args.height
    
    all_cameras: list = []
    for cam_name in camera_names:
        is_bev = (cam_name == 'bev')
        cam = create_camera(width, height, device, bev=is_bev,
                           bev_height=args.bev_height, bev_fov=args.bev_fov,
                           scene=scene, camera_name=cam_name)
        all_cameras.append(cam)
    ctx.upload_cameras(all_cameras)
    
    # Upload scene
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    
    # Compute poses for all cameras
    print("  Computing camera poses...")
    t0 = time.time()
    all_poses = []
    for cam_name in camera_names:
        bev_h = args.bev_height if cam_name == 'bev' else None
        poses, _ = compute_camera_poses(scene, timestamps, device, 
                                        bev_height=bev_h, camera_name=cam_name)
        all_poses.append(poses)
    pose_time = time.time() - t0
    print(f"    Time: {pose_time*1000:.1f}ms")
    
    # Determine camera type based on actual cameras, not args.bev
    camera_type_id = CAMERA_TYPE_BEV if 'bev' in camera_names else CAMERA_TYPE_REGULAR
    
    # Setup output directories
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = "hdmap_sequence_bev" if args.bev else "hdmap_sequence"
    output_base = os.path.join(os.path.dirname(__file__), f"../_images/{base_dir}")
    os.makedirs(output_base, exist_ok=True)
    
    output_dirs = []
    for cam_name in camera_names:
        cam_subdir = cam_name.replace(':', '_')
        output_dir = os.path.join(output_base, cam_subdir)
        os.makedirs(output_dir, exist_ok=True)
        output_dirs.append(output_dir)
    
    # Format-specific rendering pipeline
    if output_format == 'mp4':
        gpu_render_time, encode_time = _render_mp4_all_cameras(
            ctx, scene_id, timestamps, all_poses, camera_type_id,
            camera_names, output_base, width, height, n_frames, n_cameras,
            batch_size, device, fps=fps, bitrate=args.bitrate
        )
        transfer_time = 0.0
        save_time = encode_time
        
    elif output_format == 'jpg':
        gpu_render_time, transfer_time, save_time = _render_jpg_all_cameras(
            ctx, scene_id, timestamps, all_poses, camera_type_id,
            output_dirs, camera_names, width, height, n_frames, n_cameras,
            batch_size, device, quality=args.quality
        )
        
    else:
        # PNG: Default path with CPU transfer
        gpu_render_time, transfer_time, save_time = _render_png_all_cameras(
            ctx, scene_id, timestamps, all_poses, camera_type_id,
            output_dirs, camera_names, width, height, n_frames, n_cameras,
            batch_size, device
        )
    
    # Summary
    total_time = pose_time + gpu_render_time + transfer_time + save_time
    print(f"\nDone! Rendered {total_images} images:")
    print(f"  Pose compute: {pose_time*1000:.1f}ms")
    print(f"  GPU render:   {gpu_render_time*1000:.1f}ms ({total_images/gpu_render_time:.0f} FPS)")
    if output_format == 'mp4':
        print(f"  H264 encode:  {save_time:.2f}s")
    elif output_format == 'jpg':
        print(f"  nvJPEG encode:{transfer_time*1000:.1f}ms")
        print(f"  File save:    {save_time:.2f}s")
    else:
        print(f"  CPU transfer: {transfer_time*1000:.1f}ms")
        print(f"  File save:    {save_time:.2f}s")
    print(f"  Total:        {total_time:.2f}s ({total_images/total_time:.1f} FPS)")


def list_cameras(args):
    """List available cameras in the scene."""
    device = torch.device('cuda')
    print(f"Loading scene: {args.scene}")
    scene = load_scene(args.scene, device,
                       use_gpu_decoder=_get_gpu_decoder_flag(args))
    
    cameras = get_available_cameras(scene)
    print(f"\nAvailable cameras ({len(cameras)}):")
    for cam in cameras:
        print(f"  {cam}")


def render_single_frame_multi_camera(args, camera_names: list):
    """Render a single frame for multiple cameras."""
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.render_utils import render_frame
    
    device = torch.device('cuda')
    
    print(f"Loading scene: {args.scene}")
    include_ego_traj = getattr(args, 'ego_trajectory', False)
    scene = load_scene(args.scene, device, include_ego_obstacle=args.bev,
                       include_ego_trajectory=include_ego_traj,
                       use_gpu_decoder=_get_gpu_decoder_flag(args))
    
    # Resample timestamps to 10Hz
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, 100000, 20000000)
    print(f"Timestamps: {len(timestamps)} frames at 10Hz")
    
    if args.frame >= len(timestamps):
        print(f"Error: Frame {args.frame} out of range (max {len(timestamps) - 1})")
        return
    
    # Create context
    ctx = _create_context(args, device)
    ctx.set_depth_scaling(True)
    
    width, height = args.width, args.height
    
    # Upload all cameras
    all_cameras = []
    for cam_name in camera_names:
        is_bev = (cam_name == 'bev')
        cam = create_camera(width, height, device, bev=is_bev,
                           bev_height=args.bev_height, bev_fov=args.bev_fov,
                           scene=scene, camera_name=cam_name)
        all_cameras.append(cam)
    ctx.upload_cameras(all_cameras)
    
    # Upload scene
    scene_id = ctx.upload_scene(scene.timestamped_scene)
    
    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../_images")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering frame {args.frame} for {len(camera_names)} cameras...")
    
    for cam_idx, cam_name in enumerate(camera_names):
        bev_h = args.bev_height if cam_name == 'bev' else None
        img = render_frame(ctx, scene, scene_id, timestamps, args.frame, 
                          width, height, device, bev_height=bev_h,
                          camera_name=cam_name, camera_id=cam_idx)
        
        suffix = "_bev" if args.bev else ""
        cam_suffix = cam_name.replace(':', '_')
        output_path = os.path.join(output_dir, f"hdmap_scene_{args.frame:04d}{suffix}_{cam_suffix}.png")
        img.save(output_path)
        print(f"  [{cam_idx+1}/{len(camera_names)}] Saved: {output_path}")


def _render_png_all_cameras(ctx, scene_id, timestamps, all_poses, camera_type_id,
                            output_dirs, camera_names, width, height, n_frames, n_cameras,
                            batch_size, device):
    """PNG rendering pipeline: GPU -> CPU transfer -> PNG files."""
    from ludus_renderer.render_utils import gpu_to_numpy, save_frames
    
    print("\nStep 4: GPU rendering...")
    t0 = time.time()
    
    all_gpu_images = []
    for cam_idx, cam_name in enumerate(camera_names):
        camera_poses = all_poses[cam_idx]
        gpu_image_batches = []
        
        for i in range(0, n_frames, batch_size):
            end_idx = min(i + batch_size, n_frames)
            batch_n = end_idx - i
            
            scene_ids = torch.full((batch_n,), scene_id, dtype=torch.int32, device=device)
            camera_ids = torch.full((batch_n,), cam_idx, dtype=torch.int32, device=device)
            timestamps_batch = timestamps[i:end_idx].to(torch.int64)
            camera_type_ids = torch.full((batch_n,), camera_type_id, dtype=torch.int32, device=device)
            poses_batch = camera_poses[i:end_idx]
            
            images = ctx.render(
                scene_ids, camera_ids, timestamps_batch, camera_type_ids,
                poses_batch, resolution=(height, width)
            )
            images_rgb = images[:, :, :, :3]
            if ctx.needs_vflip:
                images_rgb = images_rgb.flip(1)
            gpu_image_batches.append(images_rgb.contiguous())
        
        gpu_images = torch.cat(gpu_image_batches, dim=0)
        all_gpu_images.append(gpu_images)
    
    torch.cuda.synchronize()
    gpu_render_time = time.time() - t0
    total_images = n_frames * n_cameras
    print(f"  Rendered {total_images} images")
    print(f"  Time: {gpu_render_time*1000:.1f}ms ({total_images/gpu_render_time:.0f} FPS)")
    
    print("\nStep 5: GPU -> CPU transfer...")
    t0 = time.time()
    all_cpu_images = []
    for gpu_images in all_gpu_images:
        cpu_images = gpu_to_numpy(gpu_images)
        all_cpu_images.append(cpu_images)
    transfer_time = time.time() - t0
    print(f"  Transferred {total_images} images")
    print(f"  Time: {transfer_time*1000:.1f}ms")
    
    print("\nStep 6: Saving PNG files...")
    t0 = time.time()
    for cam_idx, cam_name in enumerate(camera_names):
        save_frames(all_cpu_images[cam_idx], output_dir=output_dirs[cam_idx])
        print(f"  [{cam_idx+1}/{n_cameras}] {cam_name} -> {output_dirs[cam_idx]}")
    save_time = time.time() - t0
    print(f"  Time: {save_time:.2f}s")
    
    return gpu_render_time, transfer_time, save_time


def _render_jpg_all_cameras(ctx, scene_id, timestamps, all_poses, camera_type_id,
                            output_dirs, camera_names, width, height, n_frames, n_cameras,
                            batch_size, device, quality=85):
    """JPG rendering pipeline: GPU -> nvJPEG encode -> JPG files.

    Uses the package-level ``ludus_renderer.nvjpeg.encode`` GPU encoder, which
    accepts an arbitrary CUDA tensor batch.
    """
    from ludus_renderer import nvjpeg
    if not nvjpeg.is_available():
        print("Warning: nvJPEG not available, falling back to PNG")
        return _render_png_all_cameras(
            ctx, scene_id, timestamps, all_poses, camera_type_id,
            output_dirs, camera_names, width, height, n_frames, n_cameras,
            batch_size, device
        )

    print(f"\nStep 4: GPU rendering + nvJPEG encoding (quality={quality})...")
    total_images = n_frames * n_cameras
    encode_time = 0.0
    t0 = time.time()

    all_jpeg_data = []

    for cam_idx, cam_name in enumerate(camera_names):
        camera_poses = all_poses[cam_idx]
        camera_jpegs = []

        for i in range(0, n_frames, batch_size):
            end_idx = min(i + batch_size, n_frames)
            batch_n = end_idx - i

            scene_ids = torch.full((batch_n,), scene_id, dtype=torch.int32, device=device)
            camera_ids = torch.full((batch_n,), cam_idx, dtype=torch.int32, device=device)
            timestamps_batch = timestamps[i:end_idx].to(torch.int64)
            camera_type_ids = torch.full((batch_n,), camera_type_id, dtype=torch.int32, device=device)
            poses_batch = camera_poses[i:end_idx]

            images = ctx.render(
                scene_ids, camera_ids, timestamps_batch, camera_type_ids,
                poses_batch, resolution=(height, width)
            )
            images_rgb = images[:, :, :, :3]
            if ctx.needs_vflip:
                images_rgb = images_rgb.flip(1)
            images_rgb = images_rgb.permute(0, 3, 1, 2).contiguous()
            jpeg_list = nvjpeg.encode(images_rgb, quality=quality)

            camera_jpegs.extend(jpeg_list)

        all_jpeg_data.append(camera_jpegs)

    torch.cuda.synchronize()
    gpu_render_time = time.time() - t0
    print(f"  Rendered and encoded {total_images} images")
    print(f"  Time: {gpu_render_time*1000:.1f}ms ({total_images/gpu_render_time:.0f} FPS)")

    print("\nStep 5: Saving JPG files...")
    t0 = time.time()
    for cam_idx, cam_name in enumerate(camera_names):
        output_dir = output_dirs[cam_idx]
        for frame_idx, jpeg_bytes in enumerate(all_jpeg_data[cam_idx]):
            output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
            with open(output_path, 'wb') as f:
                f.write(jpeg_bytes)
        print(f"  [{cam_idx+1}/{n_cameras}] {cam_name} -> {output_dir}")
    save_time = time.time() - t0
    print(f"  Time: {save_time:.2f}s")

    return gpu_render_time, encode_time, save_time


def _get_ffmpeg_binary():
    """Find the ffmpeg binary: system PATH first, then imageio-ffmpeg bundled."""
    import shutil
    path = shutil.which('ffmpeg')
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "ffmpeg not found. Install via system package manager or "
            "'pip install imageio-ffmpeg'"
        )




def _render_mp4_ffmpeg(ctx, scene_id, timestamps, all_poses, camera_type_id,
                       camera_names, output_base, width, height, n_frames, n_cameras,
                       batch_size, device, fps=30, bitrate=10_000_000):
    """MP4 via ffmpeg software encoding."""
    import subprocess

    ffmpeg_bin = _get_ffmpeg_binary()
    print(f"\nStep 4: GPU rendering + ffmpeg SW encoding (fps={fps}, bitrate={bitrate//1_000_000}Mbps)...")
    total_images = n_frames * n_cameras
    gpu_render_time = 0.0
    encode_time = 0.0

    output_paths = []
    for cam_idx, cam_name in enumerate(camera_names):
        camera_poses = all_poses[cam_idx]
        cam_subdir = cam_name.replace(':', '_')
        output_path = os.path.join(output_base, f"{cam_subdir}.mp4")
        output_paths.append(output_path)

        ffmpeg_cmd = [
            ffmpeg_bin, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-b:v', str(bitrate),
            output_path,
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        assert ffmpeg_proc.stdin is not None

        for i in range(0, n_frames, batch_size):
            end_idx = min(i + batch_size, n_frames)
            batch_n = end_idx - i

            scene_ids = torch.full((batch_n,), scene_id, dtype=torch.int32, device=device)
            camera_ids = torch.full((batch_n,), cam_idx, dtype=torch.int32, device=device)
            timestamps_batch = timestamps[i:end_idx].to(torch.int64)
            camera_type_ids = torch.full((batch_n,), camera_type_id, dtype=torch.int32, device=device)
            poses_batch = camera_poses[i:end_idx]

            torch.cuda.synchronize()
            t_render = time.time()
            images = ctx.render(
                scene_ids, camera_ids, timestamps_batch, camera_type_ids,
                poses_batch, resolution=(height, width)
            )
            torch.cuda.synchronize()
            gpu_render_time += time.time() - t_render

            t_enc = time.time()
            frames_rgb = images[:, :, :, :3]
            if ctx.needs_vflip:
                frames_rgb = frames_rgb.flip(1)
            frames_rgb = frames_rgb.contiguous().cpu().numpy()
            for frame_idx in range(batch_n):
                ffmpeg_proc.stdin.write(frames_rgb[frame_idx].tobytes())
            encode_time += time.time() - t_enc

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed for {cam_name}")

        print(f"  [{cam_idx+1}/{n_cameras}] {cam_name} -> {output_path}")

    print(f"  Rendered and encoded {total_images} images to {n_cameras} MP4 files")
    print(f"  Time: render={gpu_render_time:.2f}s, encode={encode_time:.2f}s")

    return gpu_render_time, encode_time



def _render_mp4_all_cameras(ctx, scene_id, timestamps, all_poses, camera_type_id,
                            camera_names, output_base, width, height, n_frames, n_cameras,
                            batch_size, device, fps=30, bitrate=10_000_000):
    """MP4 rendering via ffmpeg software encoding."""
    return _render_mp4_ffmpeg(
        ctx, scene_id, timestamps, all_poses, camera_type_id,
        camera_names, output_base, width, height, n_frames, n_cameras,
        batch_size, device, fps=fps, bitrate=bitrate,
    )


def render_overlay_sequence(args):
    """Render HD map overlay on top of an input video.
    
    Renders the scene at the video's native frame rate (1:1 frame mapping),
    blends each rendered frame 50:50 with the corresponding video frame,
    and writes the composite as png/jpg frames or an mp4 video.
    
    Blending and JPEG encoding are performed on GPU via PyTorch and nvjpeg.
    """
    import cv2
    from ludus_renderer.util import resample_timestamps
    from ludus_renderer.render_utils import compute_camera_poses
    from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR, CAMERA_TYPE_BEV

    device = torch.device('cuda')
    video_path = args.overlay_video
    camera_name = args.camera or DEFAULT_CAMERA
    output_format = args.output_format

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_offset = getattr(args, 'video_offset', 0)
    print(f"Input video: {video_path}")
    print(f"  {video_w}x{video_h}, {video_fps:.2f} fps, {video_frame_count} frames")

    # Skip leading video frames to align with ego timestamps
    if video_offset > 0:
        for _ in range(video_offset):
            cap.read()
        video_frame_count -= video_offset
        print(f"  Skipping first {video_offset} video frames (aligned count: {video_frame_count})")

    width, height = args.width, args.height
    fps = args.fps if args.fps is not None else int(round(video_fps))

    # For jpg output, setup nvjpeg
    use_nvjpeg = False
    if output_format == 'jpg':
        from ludus_renderer import nvjpeg
        if nvjpeg.is_available():
            use_nvjpeg = True
            print("  Using nvjpeg GPU encoder for JPEG output")
        else:
            print("  Warning: nvjpeg not available, falling back to PIL")

    # Load scene
    print(f"Loading scene: {args.scene}")
    include_ego_traj = getattr(args, 'ego_trajectory', False)
    scene = load_scene(args.scene, device, include_ego_obstacle=args.bev,
                       include_ego_trajectory=include_ego_traj,
                       use_gpu_decoder=_get_gpu_decoder_flag(args))

    # Resample timestamps
    timestep_us = 1000000 // fps
    duration_us = (scene.ego_tracks.timestamps[-1] - scene.ego_tracks.timestamps[0]).item()
    timestamps = resample_timestamps(scene.ego_tracks.timestamps, timestep_us, duration_us)
    n_render = len(timestamps)
    n_frames = min(n_render, video_frame_count)
    timestamps = timestamps[:n_frames]
    print(f"Overlay: {n_frames} frames (render={n_render}, video={video_frame_count})")
    print(f"  Resolution: {width}x{height}, fps={fps}, format={output_format.upper()}")

    # Setup renderer
    ctx = _create_context(args, device)
    ctx.set_depth_scaling(True)

    cam = create_camera(width, height, device, bev=args.bev,
                        bev_height=args.bev_height, bev_fov=args.bev_fov,
                        scene=scene, camera_name=camera_name)
    ctx.upload_cameras([cam])
    scene_id = ctx.upload_scene(scene.timestamped_scene)

    bev_height_val = args.bev_height if args.bev else None
    camera_type_id = CAMERA_TYPE_BEV if args.bev else CAMERA_TYPE_REGULAR

    print("  Computing camera poses...")
    poses, _ = compute_camera_poses(scene, timestamps, device,
                                    bev_height=bev_height_val, camera_name=camera_name)

    # Setup output paths
    cam_tag = camera_name.replace(':', '_')
    output_base = os.path.join(os.path.dirname(__file__), "../_images")
    os.makedirs(output_base, exist_ok=True)

    ffmpeg_proc = None
    if output_format == 'mp4':
        import subprocess
        ffmpeg_bin = _get_ffmpeg_binary()
        output_dest = os.path.join(output_base, f"overlay_{cam_tag}.mp4")
        ffmpeg_cmd = [
            ffmpeg_bin, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', output_dest
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        assert ffmpeg_proc.stdin is not None
    else:
        output_dest = os.path.join(output_base, f"overlay_{cam_tag}")
        os.makedirs(output_dest, exist_ok=True)

    # Render + blend in batches
    max_batch = ctx.max_batch_size
    batch_size = args.batch_size or n_frames
    if batch_size > max_batch:
        print(f"  Clamping batch_size {batch_size} -> {max_batch}")
        batch_size = max_batch
    quality = args.quality
    print("  Rendering and blending...")
    t0 = time.time()
    frame_counter = 0

    for i in range(0, n_frames, batch_size):
        end_idx = min(i + batch_size, n_frames)
        batch_n = end_idx - i

        scene_ids = torch.full((batch_n,), scene_id, dtype=torch.int32, device=device)
        camera_ids = torch.zeros(batch_n, dtype=torch.int32, device=device)
        ts_batch = timestamps[i:end_idx].to(torch.int64)
        type_ids = torch.full((batch_n,), camera_type_id, dtype=torch.int32, device=device)
        poses_batch = poses[i:end_idx]

        images = ctx.render(
            scene_ids, camera_ids, ts_batch, type_ids,
            poses_batch, resolution=(height, width)
        )
        # [B, H, W, 3] uint8 on GPU — extract RGB and flip vertically
        rendered_gpu = images[:, :, :, :3]
        if ctx.needs_vflip:
            rendered_gpu = rendered_gpu.flip(1)

        # Read video frames for this batch, resize, convert BGR→RGB, upload to GPU
        video_batch = torch.empty(batch_n, height, width, 3, dtype=torch.uint8, device=device)
        actual_n = batch_n
        for j in range(batch_n):
            ret, bgr_frame = cap.read()
            if not ret:
                print(f"  Video ended at frame {i + j}, stopping.")
                actual_n = j
                break
            if bgr_frame.shape[1] != width or bgr_frame.shape[0] != height:
                bgr_frame = cv2.resize(bgr_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            video_batch[j] = torch.from_numpy(rgb_frame).to(device, non_blocking=True)

        if actual_n == 0:
            break
        if actual_n < batch_n:
            rendered_gpu = rendered_gpu[:actual_n]
            video_batch = video_batch[:actual_n]

        # 50:50 blend on GPU using integer arithmetic (no float conversion)
        blended = ((rendered_gpu.to(torch.int32) + video_batch.to(torch.int32)) >> 1).to(torch.uint8)

        # Write output
        if output_format == 'jpg' and use_nvjpeg:
            # nvjpeg expects [B, 3, H, W] uint8 contiguous
            blended_chw = blended.permute(0, 3, 1, 2).contiguous()
            jpeg_list = nvjpeg.encode(blended_chw, quality=quality)
            for j, jpeg_bytes in enumerate(jpeg_list):
                path = os.path.join(output_dest, f"frame_{frame_counter:04d}.jpg")
                with open(path, 'wb') as f:
                    f.write(jpeg_bytes)
                frame_counter += 1
        else:
            blended_cpu = blended.cpu().numpy()
            for j in range(actual_n):
                if output_format == 'mp4':
                    bgr = cv2.cvtColor(blended_cpu[j], cv2.COLOR_RGB2BGR)
                    assert ffmpeg_proc is not None
                    assert ffmpeg_proc.stdin is not None
                    ffmpeg_proc.stdin.write(bgr.tobytes())
                elif output_format == 'jpg':
                    path = os.path.join(output_dest, f"frame_{frame_counter:04d}.jpg")
                    Image.fromarray(blended_cpu[j]).save(path, quality=quality)
                else:
                    path = os.path.join(output_dest, f"frame_{frame_counter:04d}.png")
                    Image.fromarray(blended_cpu[j]).save(path)
                frame_counter += 1

        if frame_counter % 100 == 0:
            print(f"    {frame_counter}/{n_frames} frames...")

        if actual_n < batch_n:
            break

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    cap.release()
    if ffmpeg_proc is not None:
        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    print(f"\nDone! Overlay saved to: {output_dest}")
    print(f"  {frame_counter} frames in {elapsed:.2f}s ({frame_counter / elapsed:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description='HDMap Scene Renderer')
    parser.add_argument('--scene', type=str, default=DEFAULT_SCENE_PATH,
                        help='Path to clipgt scene directory')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to render (default: 0)')
    parser.add_argument('--sequence', action='store_true',
                        help='Render full sequence instead of single frame')
    parser.add_argument('--width', type=int, default=1280,
                        help='Output image width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Output image height (default: 720)')
    parser.add_argument('--bev', action='store_true',
                        help='Use bird\'s eye view camera')
    parser.add_argument('--ego-trajectory', action='store_true',
                        help='Enable ego trajectory rendering (disabled by default)')
    parser.add_argument('--bev-height', type=float, default=80.0,
                        help='BEV camera height in meters (default: 80)')
    parser.add_argument('--bev-fov', type=float, default=60.0,
                        help='BEV camera vertical FOV in degrees (default: 60)')
    parser.add_argument('--camera', type=str, default=None,
                        help=f'Scene camera name (default: {DEFAULT_CAMERA})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for sequence rendering (default: all frames at once)')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available cameras in the scene and exit')
    parser.add_argument('--all-cameras', action='store_true',
                        help='Render with all available cameras')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output frame rate (default: 10 for rendering, video fps for overlay)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory name under _images/ (default: hdmap_sequence)')
    parser.add_argument('--output-format', type=str, default='png',
                        choices=['png', 'jpg', 'mp4'],
                        help='Output format: png (default), jpg (nvJPEG), mp4 (H264)')
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG quality 1-100 (default: 90)')
    parser.add_argument('--bitrate', type=int, default=10_000_000,
                        help='Video bitrate in bps (default: 10Mbps)')
    parser.add_argument('--overlay-video', type=str, default=None,
                        help='Input video path for overlay compositing')
    parser.add_argument('--video-offset', type=int, default=4,
                        help='Number of video frames to skip before aligning with ego timestamps (default: 4)')
    parser.add_argument('--msaa', type=int, default=0, choices=[0, 4],
                        help='MSAA sample count (0=disabled, 4=4x antialiasing)')
    parser.add_argument('--loader', type=str, default='gpu', choices=['gpu', 'cpu'],
                        help='Scene loader: gpu (GPU-native parquet, default) or cpu (PyArrow)')

    args = parser.parse_args()
    
    
    if args.list_cameras:
        list_cameras(args)
        return
    
    if args.overlay_video:
        render_overlay_sequence(args)
        return
    
    # Render sequence or single frame
    if args.sequence:
        # render_sequence handles --all-cameras internally
        render_sequence(args, all_cameras=args.all_cameras)
    elif args.all_cameras:
        # Single frame for all cameras
        device = torch.device('cuda')
        include_ego_traj = getattr(args, 'ego_trajectory', False)
        scene = load_scene(args.scene, device, include_ego_obstacle=args.bev,
                           include_ego_trajectory=include_ego_traj,
                           use_gpu_decoder=_get_gpu_decoder_flag(args))
        camera_names = get_available_cameras(scene)
        print(f"Using all {len(camera_names)} cameras")
        render_single_frame_multi_camera(args, camera_names)
    else:
        render_single_frame(args)


if __name__ == "__main__":
    main()
