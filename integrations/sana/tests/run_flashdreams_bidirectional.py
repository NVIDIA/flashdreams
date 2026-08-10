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

"""Run the FlashDreams SANA-WM bidirectional benchmark helper."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from run_flashdreams_common import (
    apply_backend_defaults,
    install_stage1_compile_hook,
    preload_pipeline_components,
    prepare_bidirectional_inputs,
    resolve_device,
    write_json,
    write_video,
)
from sana_wm.conditioning import SanaWMI2VConditioningRequest
from sana_wm.config import RUNNER_SANA_WM_BIDIRECTIONAL
from sana_wm.decoder import SanaWMDecodedVideo
from sana_wm.runner import (
    _active_quantized_precisions,
    _pipeline_config,
    _resolve_quant_backend,
    _validate_precision_request,
)

from flashdreams.infra.config import derive_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument("--intrinsics-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="flashdreams")
    parser.add_argument("--dump-frames", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=161)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=60)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=None)
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
    parser.add_argument("--warmup-generations", type=int, default=0)
    parser.add_argument("--measured-generations", type=int, default=1)
    args = parser.parse_args(argv)

    if args.warmup_generations < 0:
        raise ValueError("--warmup-generations must be >= 0")
    if args.measured_generations < 1:
        raise ValueError("--measured-generations must be >= 1")

    if args.force_cudnn_sdpa:
        apply_backend_defaults()

    prompt = args.prompt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt_path}")

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
    image, c2w, intrinsics_vec4, num_frames = prepare_bidirectional_inputs(
        image_path=args.image_path,
        camera_path=args.camera_path,
        intrinsics_path=args.intrinsics_path,
        intrinsics_hfov_deg=args.intrinsics_hfov_deg,
        num_frames_requested=args.num_frames,
    )
    runner_cfg = derive_config(
        RUNNER_SANA_WM_BIDIRECTIONAL,
        output_dir=args.output_dir,
        runner_name=args.name,
        num_frames=num_frames,
        fps=args.fps,
        step=args.step,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        no_refiner=args.no_refiner,
        save_stage1=args.save_stage1,
        refiner_seed=args.seed,
        stage1_precision=args.stage1_precision,
        refiner_precision=args.refiner_precision,
        quant_backend=args.quant_backend,
    )
    pipeline_cfg = _pipeline_config(runner_cfg, quant_backend=quant_backend)
    pipeline_cfg = derive_config(pipeline_cfg, enable_sync_and_profile=True)
    pipeline = pipeline_cfg.setup().to(device).eval()
    if args.compile_stage1:
        install_stage1_compile_hook(pipeline)
    preload_pipeline_components(pipeline)

    request = SanaWMI2VConditioningRequest(
        image=image,
        prompt=prompt,
        poses_c2w=c2w,
        intrinsics_vec4=intrinsics_vec4,
        num_frames=num_frames,
        fps=args.fps,
        steps=args.step,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )
    generation_records: list[dict[str, object]] = []
    final_record: dict[str, object] | None = None
    final_frames: np.ndarray | None = None
    total_generations = args.warmup_generations + args.measured_generations
    with torch.inference_mode():
        for generation_index in range(total_generations):
            warmup = generation_index < args.warmup_generations
            cache = pipeline.initialize_cache(
                decoder_context={
                    "prompt": prompt,
                    "fps": args.fps,
                    "save_stage1": args.save_stage1,
                    "refiner_seed": args.seed,
                    "sink_size": runner_cfg.sink_size,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            generation_start = time.perf_counter()
            decoded = pipeline.generate(0, cache, input=request)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - generation_start
            stats = pipeline.finalize(0, cache) or {}

            if not isinstance(decoded, SanaWMDecodedVideo):
                raise TypeError(
                    f"expected SanaWMDecodedVideo, got {type(decoded).__name__}"
                )

            if torch.cuda.is_available():
                stats.setdefault(
                    "mem_peak_gib",
                    torch.cuda.max_memory_allocated(device) / (1024**3),
                )
            frames = np.asarray(decoded.video_hwc, dtype=np.uint8)
            record: dict[str, object] = {
                "generation_index": generation_index,
                "warmup": warmup,
                "wall_s": wall_s,
                "stats_ms": stats,
                "video_shape": list(frames.shape),
                "num_frames": num_frames,
                "seed": args.seed,
            }
            generation_records.append(record)
            if not warmup:
                final_record = record
                final_frames = frames

    if final_record is None or final_frames is None:
        raise RuntimeError("bidirectional benchmark produced no measured generations")

    video_path = write_video(args.output_dir, args.name, final_frames, args.fps)
    final_record["video_path"] = str(video_path)
    if args.dump_frames is not None:
        args.dump_frames.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_frames, final_frames)
    if args.stats_json is not None:
        write_json(
            args.stats_json,
            {
                "backend": "flashdreams",
                "variant": "bidirectional",
                "runner": "sana-wm-bidirectional",
                "video_path": str(video_path),
                "video_shape": list(final_frames.shape),
                "num_frames": num_frames,
                "seed": args.seed,
                "fps": args.fps,
                "step": args.step,
                "cfg_scale": args.cfg_scale,
                "no_refiner": args.no_refiner,
                "stage1_precision": args.stage1_precision,
                "refiner_precision": args.refiner_precision,
                "compile_stage1": args.compile_stage1,
                "force_cudnn_sdpa": args.force_cudnn_sdpa,
                "warmup_generations": args.warmup_generations,
                "measured_generations": args.measured_generations,
                "wall_s": final_record["wall_s"],
                "stats_ms": final_record["stats_ms"],
                "generation_records": generation_records,
            },
        )
    print(f"[flashdreams] wrote {video_path}")


if __name__ == "__main__":
    main()
