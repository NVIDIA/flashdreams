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

"""Run the FlashDreams SANA-WM streaming benchmark helper."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from run_flashdreams_common import (
    apply_backend_defaults,
    compile_streaming_refiner,
    install_stage1_compile_hook,
    preload_pipeline_components,
    prepare_streaming_inputs,
    resolve_device,
    sum_stage_ms,
    write_json,
    write_video,
)
from sana_wm.conditioning import (
    SanaWMStreamingI2VConditioningRequest,
    streaming_chunk_boundaries,
)
from sana_wm.config import RUNNER_SANA_WM_STREAMING
from sana_wm.constants import (
    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
    SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)
from sana_wm.decoder import SanaWMDecodedVideo
from sana_wm.runner import (
    _active_quantized_precisions,
    _resolve_quant_backend,
    _streaming_pipeline_config,
    _validate_precision_request,
)

from flashdreams.infra.config import derive_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, default=None)
    parser.add_argument(
        "--camera-source",
        choices=["camera", "action"],
        default="camera",
    )
    parser.add_argument("--action", default=None)
    parser.add_argument("--translation-speed", type=float, default=0.025)
    parser.add_argument("--rotation-speed-deg", type=float, default=0.6)
    parser.add_argument("--intrinsics-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="flashdreams")
    parser.add_argument("--dump-frames", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=241)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--no-refiner", action="store_true")
    parser.add_argument("--save-stage1", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--intrinsics-hfov-deg", type=float, default=90.0)
    parser.add_argument(
        "--stage1-precision",
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
    )
    parser.add_argument(
        "--refiner-precision",
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
    )
    parser.add_argument(
        "--quant-backend",
        choices=["auto", "torch", "torch-fp8", "torch-fp4"],
        default="auto",
    )
    parser.add_argument("--force-cudnn-sdpa", action="store_true")
    parser.add_argument("--compile-stage1", action="store_true")
    parser.add_argument("--compile-streaming-refiner", action="store_true")
    parser.add_argument("--output-mode", choices=["mp4", "discard"], default="mp4")
    parser.add_argument(
        "--denoising-step-list",
        default="1000,960,889,727,0",
    )
    parser.add_argument(
        "--num-frame-per-block",
        type=int,
        default=SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
    )
    parser.add_argument("--num-cached-blocks", type=int, default=2)
    parser.add_argument("--sink-size", type=int, default=1)
    parser.add_argument("--no-sink-token", action="store_true")
    parser.add_argument(
        "--refiner-block-size",
        type=int,
        default=SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--refiner-kv-max-frames",
        type=int,
        default=SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES,
    )
    parser.add_argument("--refiner-seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.force_cudnn_sdpa:
        apply_backend_defaults()

    prompt = args.prompt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt_path}")
    if args.refiner_seed is None:
        args.refiner_seed = args.seed

    device = resolve_device(args.device)
    quantized = _active_quantized_precisions(
        stage1_precision=args.stage1_precision,
        refiner_precision=args.refiner_precision,
        refiner_enabled=not args.no_refiner,
    )
    quant_backend = _resolve_quant_backend(args.quant_backend, quantized)
    _validate_precision_request(
        device=device,
        stage1_precision=args.stage1_precision,
        refiner_precision=args.refiner_precision,
        refiner_enabled=not args.no_refiner,
        quant_backend=args.quant_backend,
    )

    snap_stride = SANA_WM_VAE_TEMPORAL_COMPRESSION * args.refiner_block_size
    image, c2w, intrinsics_vec4, num_frames = prepare_streaming_inputs(
        image_path=args.image_path,
        camera_path=args.camera_path,
        camera_source=args.camera_source,
        action=args.action,
        translation_speed=args.translation_speed,
        rotation_speed_deg=args.rotation_speed_deg,
        intrinsics_path=args.intrinsics_path,
        intrinsics_hfov_deg=args.intrinsics_hfov_deg,
        num_frames_requested=args.num_frames,
        snap_stride=snap_stride,
    )
    denoising_step_list = tuple(
        int(timestep.strip())
        for timestep in args.denoising_step_list.split(",")
        if timestep.strip()
    )
    if not denoising_step_list or denoising_step_list[-1] != 0:
        raise ValueError("--denoising-step-list must end with 0.")

    runner_cfg = derive_config(
        RUNNER_SANA_WM_STREAMING,
        output_dir=args.output_dir,
        runner_name=args.name,
        num_frames=num_frames,
        fps=args.fps,
        step=len(denoising_step_list) - 1,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        no_refiner=args.no_refiner,
        save_stage1=args.save_stage1,
        refiner_seed=args.refiner_seed,
        sink_size=args.sink_size,
        stage1_precision=args.stage1_precision,
        refiner_precision=args.refiner_precision,
        quant_backend=args.quant_backend,
        num_frame_per_block=args.num_frame_per_block,
        num_cached_blocks=args.num_cached_blocks,
        no_sink_token=args.no_sink_token,
        denoising_step_list=denoising_step_list,
        refiner_block_size=args.refiner_block_size,
        refiner_kv_max_frames=args.refiner_kv_max_frames,
    )
    pipeline_cfg = _streaming_pipeline_config(runner_cfg, quant_backend=quant_backend)
    pipeline_cfg = derive_config(pipeline_cfg, enable_sync_and_profile=True)
    pipeline = pipeline_cfg.setup().to(device).eval()
    if args.compile_stage1:
        install_stage1_compile_hook(pipeline)
    preload_pipeline_components(pipeline)
    if args.compile_streaming_refiner:
        compile_streaming_refiner(pipeline)

    latent_frames = (num_frames - 1) // SANA_WM_VAE_TEMPORAL_COMPRESSION + 1
    chunk_boundaries = streaming_chunk_boundaries(
        latent_frames,
        args.num_frame_per_block,
    )
    request = SanaWMStreamingI2VConditioningRequest(
        image=image,
        prompt=prompt,
        poses_c2w=c2w,
        intrinsics_vec4=intrinsics_vec4,
        num_frames=num_frames,
        fps=args.fps,
        steps=len(denoising_step_list) - 1,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        num_frame_per_block=args.num_frame_per_block,
    )
    cache = pipeline.initialize_cache(
        decoder_context={
            "prompt": prompt,
            "fps": args.fps,
            "save_stage1": args.save_stage1,
            "refiner_seed": args.refiner_seed,
            "sink_size": runner_cfg.sink_size,
            "block_size": args.refiner_block_size,
            "refiner_kv_max_frames": args.refiner_kv_max_frames,
        }
    )

    decoded_chunks: list[np.ndarray] = []
    stage1_chunks: list[np.ndarray] = []
    per_chunk_stats: list[dict[str, float]] = []
    per_chunk_wall_s: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        for ar_idx in range(len(chunk_boundaries) - 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            chunk_start = time.perf_counter()
            decoded = pipeline.generate(ar_idx, cache, input=request)
            stats = pipeline.finalize(ar_idx, cache) or {}
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            per_chunk_wall_s.append(time.perf_counter() - chunk_start)
            per_chunk_stats.append(stats)
            if not isinstance(decoded, SanaWMDecodedVideo):
                raise TypeError(
                    f"expected SanaWMDecodedVideo, got {type(decoded).__name__}"
                )
            if decoded.video_hwc.size:
                decoded_chunks.append(np.asarray(decoded.video_hwc, dtype=np.uint8))
            if decoded.stage1_video_hwc is not None and decoded.stage1_video_hwc.size:
                stage1_chunks.append(
                    np.asarray(decoded.stage1_video_hwc, dtype=np.uint8)
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    stream_wall_seconds = time.perf_counter() - generation_start
    first_chunk_seconds = per_chunk_wall_s[0] if per_chunk_wall_s else None
    steady_state_seconds = (
        sum(per_chunk_wall_s[1:]) if len(per_chunk_wall_s) > 1 else None
    )

    if not decoded_chunks:
        raise RuntimeError("FlashDreams SANA-WM streaming produced no decoded frames.")
    video_hwc = np.concatenate(decoded_chunks, axis=0)
    first_chunk_frames = int(decoded_chunks[0].shape[0]) if decoded_chunks else 0
    n_pixel_frames = int(video_hwc.shape[0])
    video_path = None
    if args.output_mode != "discard":
        video_path = write_video(args.output_dir, args.name, video_hwc, args.fps)
    if args.dump_frames is not None:
        args.dump_frames.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_frames, video_hwc)
    if stage1_chunks and args.output_mode != "discard":
        write_video(
            args.output_dir,
            f"{args.name}_stage1",
            np.concatenate(stage1_chunks, axis=0),
            args.fps,
        )

    steady_frames = max(0, n_pixel_frames - first_chunk_frames)
    frames_per_second = (
        n_pixel_frames / stream_wall_seconds if stream_wall_seconds > 0 else None
    )
    realtime_factor = (
        frames_per_second / args.fps
        if frames_per_second is not None and args.fps > 0
        else None
    )
    steady_state_frames_per_second = (
        steady_frames / steady_state_seconds
        if steady_state_seconds and steady_state_seconds > 0
        else None
    )
    steady_state_realtime_factor = (
        steady_state_frames_per_second / args.fps
        if steady_state_frames_per_second is not None and args.fps > 0
        else None
    )
    stage1_cuda_ms = sum_stage_ms(per_chunk_stats, "diffuse_ms")
    decode_cuda_ms = sum_stage_ms(per_chunk_stats, "decode_ms")
    if args.stats_json is not None:
        write_json(
            args.stats_json,
            {
                "backend": "flashdreams",
                "variant": "streaming",
                "runner": "sana-wm-streaming",
                "output_mode": args.output_mode,
                "output_path": str(video_path) if video_path is not None else None,
                "video_shape": list(video_hwc.shape),
                "requested_num_frames": args.num_frames,
                "actual_num_frames": num_frames,
                "num_frames": num_frames,
                "seed": args.seed,
                "fps": args.fps,
                "cfg_scale": args.cfg_scale,
                "flow_shift": args.flow_shift,
                "no_refiner": args.no_refiner,
                "save_stage1": args.save_stage1,
                "stage1_precision": args.stage1_precision,
                "refiner_precision": args.refiner_precision,
                "quant_backend": args.quant_backend,
                "compile_stage1": args.compile_stage1,
                "compile_streaming_refiner": args.compile_streaming_refiner,
                "force_cudnn_sdpa": args.force_cudnn_sdpa,
                "denoising_step_list": list(denoising_step_list),
                "num_frame_per_block": args.num_frame_per_block,
                "num_cached_blocks": args.num_cached_blocks,
                "sink_token": not args.no_sink_token,
                "refiner_block_size": args.refiner_block_size,
                "refiner_kv_max_frames": args.refiner_kv_max_frames,
                "refiner_seed": args.refiner_seed,
                "stream_wall_seconds": stream_wall_seconds,
                "wall_s": stream_wall_seconds,
                "end_to_end_seconds": stream_wall_seconds,
                "first_chunk_seconds": first_chunk_seconds,
                "first_chunk_frames": first_chunk_frames,
                "steady_state_seconds": steady_state_seconds,
                "steady_state_frames_per_second": steady_state_frames_per_second,
                "steady_state_realtime_factor": steady_state_realtime_factor,
                "frames_per_second": frames_per_second,
                "realtime_factor": realtime_factor,
                "n_decode_chunks": len(chunk_boundaries) - 1,
                "n_refiner_blocks": len(chunk_boundaries) - 1,
                "n_pixel_frames": n_pixel_frames,
                "stage1_cuda_seconds": (
                    stage1_cuda_ms / 1000.0 if stage1_cuda_ms is not None else None
                ),
                "refiner_cuda_seconds": None,
                "decode_cuda_seconds": (
                    decode_cuda_ms / 1000.0 if decode_cuda_ms is not None else None
                ),
                "per_chunk_wall_seconds": per_chunk_wall_s,
                "per_chunk_stats_ms": per_chunk_stats,
                "mem_peak_gib": (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        )
    if video_path is not None:
        print(f"[flashdreams] wrote {video_path}")
    else:
        print(f"[flashdreams] streaming output discarded ({n_pixel_frames} frames)")


if __name__ == "__main__":
    main()
