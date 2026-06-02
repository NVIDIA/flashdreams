#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Ludus condition rasterizer in isolation.

This exercises the exact rasterizer path used by
:class:`omnidreams.interactive_drive.backends.world_model.WorldModelRenderBackend`
(``LudusConditionRasterizer.render_chunk``) but skips the world model
entirely, so the per-chunk timings reflect the rasterizer + host plumbing
only.

Useful for confirming the ``raster_ms=...`` numbers printed by the
interactive-drive demo on GB200 vs RTX 6000 are explained by the CUDA
software rasterizer / host launch overhead rather than the world model.

Example::

    uv run --no-sync --package flashdreams-omnidreams \
        python integrations/omnidreams/scripts/bench_ludus_raster.py \
        --frames-per-chunk 8 --num-chunks 30 --width 1280 --height 704
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omnidreams.interactive_drive._sample_assets import SAMPLE_SCENE
from omnidreams.interactive_drive.config import BevConfig, RasterConfig
from omnidreams.interactive_drive.rasterizer import LudusConditionRasterizer
from omnidreams.interactive_drive.scene_loader import load_scene_bundle
from omnidreams.scenes import local_scene_archive_path


def _resolve_scene_path(args: argparse.Namespace) -> Path:
    if args.scene is not None:
        path = Path(args.scene).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--scene file does not exist: {path}")
        return path
    if args.scene_uuid is not None:
        path = local_scene_archive_path(args.scene_uuid)
        if not path.is_file():
            raise SystemExit(
                f"Staged scene not found for uuid={args.scene_uuid!r}: {path}\n"
                f"Run ``omnidreams-prepare`` to stage it first."
            )
        return path
    if SAMPLE_SCENE.is_file():
        return SAMPLE_SCENE
    raise SystemExit(
        "No scene available. Pass --scene <path> or --scene-uuid <uuid>, or run "
        "``omnidreams-prepare`` to stage the default sample scene."
    )


def _build_synthetic_trajectory(
    *,
    initial_rig_to_world: np.ndarray,
    initial_timestamp_us: int,
    initial_yaw_rad: float,
    speed_mps: float,
    fps: float,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-moving rig poses + timestamps for ``num_frames`` frames.

    Pose values do not affect kernel timing in any meaningful way (the
    rasterizer does view-frustum culling with a fixed cull radius), so
    a constant-velocity straight-line trajectory is sufficient for a
    benchmark.
    """
    dt = 1.0 / float(fps)
    forward = np.array(
        [np.cos(initial_yaw_rad), np.sin(initial_yaw_rad), 0.0], dtype=np.float64
    )
    poses = np.empty((num_frames, 4, 4), dtype=np.float32)
    timestamps = np.empty((num_frames,), dtype=np.int64)
    base = initial_rig_to_world.astype(np.float64, copy=True)
    base_translation = base[:3, 3].copy()
    rotation = base[:3, :3].copy()
    timestamp_step_us = int(round(1_000_000.0 * dt))
    for i in range(num_frames):
        delta = forward * speed_mps * dt * float(i)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = base_translation + delta
        poses[i] = pose.astype(np.float32)
        timestamps[i] = initial_timestamp_us + i * timestamp_step_us
    return poses, timestamps


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(np.floor(k))
    hi = int(np.ceil(k))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _summarize(label: str, samples_ms: list[float], frames: int) -> None:
    if not samples_ms:
        print(f"  {label}: <no samples>")
        return
    avg = statistics.fmean(samples_ms)
    std = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    print(
        f"  {label}: "
        f"avg={avg:.2f}ms std={std:.2f} "
        f"min={min(samples_ms):.2f} "
        f"p50={_percentile(samples_ms, 50):.2f} "
        f"p95={_percentile(samples_ms, 95):.2f} "
        f"max={max(samples_ms):.2f} "
        f"per_frame_avg={avg / max(frames, 1):.2f}ms "
        f"({frames * 1000.0 / avg:.0f} fps eff.)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=str, default=None, help="Path to a .usdz scene.")
    parser.add_argument(
        "--scene-uuid",
        type=str,
        default=None,
        help="Scene UUID (resolved via omnidreams.scenes.local_scene_archive_path).",
    )
    parser.add_argument("--camera", type=str, default="camera_front_wide_120fov")
    parser.add_argument("--variant", type=str, default="1")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=8,
        help="Frames per render_chunk call (matches WorldModelRenderBackend's chunk_frames).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=30,
        help="Steady-state chunks to time (after warmup).",
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=5,
        help="Untimed warmup chunks before measurement.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=15.0,
        help="Forward speed used to step the synthetic trajectory.",
    )
    parser.add_argument(
        "--no-bev",
        action="store_true",
        help="Disable BEV rasterizer dispatch (default mirrors interactive-drive demo).",
    )
    parser.add_argument(
        "--bev-width", type=int, default=BevConfig.__dataclass_fields__["width"].default
    )
    parser.add_argument(
        "--bev-height",
        type=int,
        default=BevConfig.__dataclass_fields__["height"].default,
    )
    parser.add_argument(
        "--materialize-rgb",
        action="store_true",
        help=(
            "Force a host-side numpy materialization of every frame "
            "(simulates the demo's HUD path)."
        ),
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-chunk timing instead of aggregate stats only.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Ludus rasterizer benchmark.")

    scene_path = _resolve_scene_path(args)

    raster = RasterConfig(width=args.width, height=args.height)
    bev = (
        BevConfig(enabled=False)
        if args.no_bev
        else BevConfig(
            enabled=True,
            width=args.bev_width,
            height=args.bev_height,
        )
    )

    print(f"[bench] scene={scene_path}")
    print(
        f"[bench] resolution={args.width}x{args.height} "
        f"frames_per_chunk={args.frames_per_chunk} "
        f"warmup={args.warmup_chunks} timed={args.num_chunks} "
        f"bev={'on' if bev.enabled else 'off'} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    load_t0 = time.perf_counter()
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name=args.camera,
        variant=args.variant,
        prompt_override=None,
        raster=raster,
    )
    load_ms = (time.perf_counter() - load_t0) * 1000.0
    print(f"[bench] load_scene_bundle_ms={load_ms:.1f}")

    rasterizer = LudusConditionRasterizer(raster, bev=bev)
    upload_t0 = time.perf_counter()
    rasterizer.load_scene(bundle)
    torch.cuda.synchronize()
    upload_ms = (time.perf_counter() - upload_t0) * 1000.0
    print(f"[bench] load_scene_ms={upload_ms:.1f}")

    rig_poses, timestamps = _build_synthetic_trajectory(
        initial_rig_to_world=bundle.initial_rig_to_world,
        initial_timestamp_us=bundle.initial_timestamp_us,
        initial_yaw_rad=bundle.initial_yaw_rad,
        speed_mps=args.speed_mps,
        fps=args.fps,
        num_frames=args.frames_per_chunk
        * (args.warmup_chunks + args.num_chunks),
    )

    chunk_size = args.frames_per_chunk
    total_chunks = args.warmup_chunks + args.num_chunks

    timings_ms: list[float] = []
    materialize_ms: list[float] = []
    try:
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            stop = start + chunk_size
            chunk_poses = rig_poses[start:stop]
            chunk_ts = timestamps[start:stop]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = rasterizer.render_chunk(
                rig_poses_world=chunk_poses,
                timestamps_us=chunk_ts,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            mat_ms = 0.0
            if args.materialize_rgb:
                mt0 = time.perf_counter()
                for frame in chunk.frames:
                    np.asarray(frame.rgb_host_uint8)
                torch.cuda.synchronize()
                mat_ms = (time.perf_counter() - mt0) * 1000.0

            phase = "warmup" if chunk_idx < args.warmup_chunks else "timed"
            if args.print_each or phase == "warmup":
                print(
                    f"[bench] chunk={chunk_idx} phase={phase} "
                    f"render_ms={elapsed_ms:.2f} "
                    f"materialize_ms={mat_ms:.2f}"
                )
            if phase == "timed":
                timings_ms.append(elapsed_ms)
                if args.materialize_rgb:
                    materialize_ms.append(mat_ms)
    finally:
        # ``LudusConditionRasterizer`` owns a single GL worker thread + EGL
        # context; release it deterministically so re-running the script
        # in the same process is safe.
        with contextlib.suppress(Exception):
            rasterizer.cleanup()

    print()
    print("[bench] === results (timed phase only) ===")
    _summarize("render_chunk wallclock", timings_ms, chunk_size)
    if args.materialize_rgb:
        _summarize("materialize_rgb host", materialize_ms, chunk_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
