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

"""Command-line Real-ESRGAN image/video upsampler."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

import cv2
import torch

from realesrgan.upsampler import (
    RealESRGANUpsampler,
    default_model_name,
    write_bgr_image,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def main() -> None:
    """Run the Real-ESRGAN upsampler CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--scale", "-s", type=int, choices=(2, 4), default=2)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--pre-pad", type=int, default=10)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile().",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="Mode passed to torch.compile() when --compile is used.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--profile-warmup-frames",
        type=int,
        default=10,
        help="Frames to exclude from steady FPS profiling.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for video smoke tests.",
    )
    args = parser.parse_args()

    scale: Literal[2, 4] = 2 if args.scale == 2 else 4
    model_name = args.model_name or default_model_name(scale)
    upsampler = RealESRGANUpsampler(
        scale=scale,
        model_name=model_name,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        device=args.device,
    )

    suffix = args.input.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        _upsample_image(args.input, args.output, upsampler)
        return
    if suffix in VIDEO_EXTENSIONS:
        _upsample_video(
            args.input,
            args.output,
            upsampler,
            max_frames=args.max_frames,
            profile_warmup_frames=args.profile_warmup_frames,
        )
        return
    raise SystemExit(f"Unsupported input extension {suffix!r}.")


def _upsample_image(
    input_path: Path,
    output_path: Path,
    upsampler: RealESRGANUpsampler,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image {input_path}")
    output, mode = upsampler.upsample_bgr_image(image)
    if mode == "RGBA":
        output_path = output_path.with_suffix(".png")
    write_bgr_image(output_path, output)
    print(f"wrote {output.shape[1]}x{output.shape[0]} {output_path}")


def _upsample_video(
    input_path: Path,
    output_path: Path,
    upsampler: RealESRGANUpsampler,
    *,
    max_frames: int | None,
    profile_warmup_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_factory = getattr(cv2, "VideoWriter_fourcc")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc_factory(*"mp4v"),
        fps,
        (width * upsampler.scale, height * upsampler.scale),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer {output_path}")

    start = time.perf_counter()
    frame_idx = 0
    process_durations: list[float] = []
    pipeline_durations: list[float] = []
    model_durations_ms: list[float | None] = []
    try:
        while True:
            process_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            pipeline_start = time.perf_counter()
            output, _mode, profile = upsampler.upsample_bgr_image_profiled(frame)
            _synchronize(upsampler)
            pipeline_durations.append(time.perf_counter() - pipeline_start)
            model_durations_ms.append(profile.model_ms)
            writer.write(output)
            process_durations.append(time.perf_counter() - process_start)
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    total_note = f"/{total_frames}" if total_frames else ""
    warmup_frames = min(max(profile_warmup_frames, 0), len(process_durations))
    steady_process = process_durations[warmup_frames:]
    steady_pipeline = pipeline_durations[warmup_frames:]
    steady_model_ms = model_durations_ms[warmup_frames:]
    print(f"wrote {frame_idx}{total_note} frames -> {output_path}")
    print(
        f"profile total_s={elapsed:.3f} "
        f"end_to_end_fps={_fps(frame_idx, elapsed):.2f} "
        f"video_loop_fps={_duration_fps(process_durations):.2f} "
        f"pipeline_fps={_duration_fps(pipeline_durations):.2f} "
        f"model_fps={_duration_ms_fps(model_durations_ms):.2f}"
    )
    print(
        f"profile_steady warmup_frames={warmup_frames} "
        f"steady_frames={len(steady_process)} "
        f"steady_video_loop_fps={_duration_fps(steady_process):.2f} "
        f"steady_pipeline_fps={_duration_fps(steady_pipeline):.2f} "
        f"steady_model_fps={_duration_ms_fps(steady_model_ms):.2f}"
    )


def _synchronize(upsampler: RealESRGANUpsampler) -> None:
    if upsampler.device.type == "cuda":
        torch.cuda.synchronize(upsampler.device)


def _duration_fps(durations: list[float]) -> float:
    return _fps(len(durations), sum(durations))


def _duration_ms_fps(durations_ms: list[float | None]) -> float:
    durations = [duration for duration in durations_ms if duration is not None]
    return _fps(len(durations), sum(durations) / 1000.0)


def _fps(frames: int, elapsed: float) -> float:
    return frames / elapsed if elapsed > 0 else 0.0


if __name__ == "__main__":
    main()
