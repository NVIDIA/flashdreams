# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark one SwiftVR compile configuration in a fresh process."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from swiftvr.impl.pipeline import SwiftVRPipeline
from swiftvr.impl.postprocess import SwiftVRStream
from torch import Tensor

_CHECKPOINT_REVISION = "743ed2530c550764905400f38eb6cc41af5abc80"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--input-tensor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="H-oliday/SwiftVR")
    parser.add_argument("--output-height", type=int, default=1408)
    parser.add_argument("--output-width", type=int, default=2560)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--warmup-chunks", type=int, default=5)
    parser.add_argument("--measured-chunks", type=int, default=20)
    parser.add_argument("--compile-transformer", action="store_true")
    parser.add_argument("--compile-encoder", action="store_true")
    parser.add_argument("--compile-decoder", action="store_true")
    return parser.parse_args()


def _load_frames(path: Path, chunk_size: int) -> Tensor:
    frames = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(frames, Tensor):
        raise TypeError(f"expected a tensor in {path}, got {type(frames).__name__}")
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
        raise ValueError(
            "input tensor must be uint8 [T, H, W, 3], got "
            f"shape={tuple(frames.shape)} dtype={frames.dtype}"
        )
    if frames.shape[0] < chunk_size or frames.shape[0] % chunk_size:
        raise ValueError(
            f"input frame count must be a positive multiple of {chunk_size}, "
            f"got {frames.shape[0]}"
        )
    return frames.pin_memory()


def _chunk(frames: Tensor, index: int, chunk_size: int, device: torch.device) -> Tensor:
    chunks = frames.shape[0] // chunk_size
    start = index % chunks * chunk_size
    return frames[start : start + chunk_size].to(device, non_blocking=True)


def _percentile90(samples: list[float]) -> float:
    return sorted(samples)[max(0, int(0.9 * len(samples)) - 1)]


def _restore_visual(
    pipeline: SwiftVRPipeline,
    frames: Tensor,
    *,
    output_height: int,
    output_width: int,
    chunk_size: int,
) -> Tensor:
    stream = SwiftVRStream(
        pipeline,
        output_height=output_height,
        output_width=output_width,
        overlap=0,
    )
    outputs = []
    for start in range(0, frames.shape[0], chunk_size):
        output = stream.step(
            frames[start : start + chunk_size].to(pipeline.device, non_blocking=True)
        )
        if output is not None:
            outputs.append(output)
    # A complete four-frame group leaves SwiftVR's three-frame causal startup
    # trim outstanding. Feed one replicated frame so flush emits those frames.
    output = stream.step(frames[-1:].to(pipeline.device, non_blocking=True))
    if output is not None:
        outputs.append(output)
    output = stream.flush()
    if output is not None:
        outputs.append(output)
    restored = torch.cat(outputs, dim=1)[:, : frames.shape[0]]
    if restored.shape[1] != frames.shape[0]:
        raise RuntimeError(
            f"restored {restored.shape[1]} frames for {frames.shape[0]} inputs"
        )
    return restored


def main() -> None:
    args = _arguments()
    if min(args.chunk_size, args.warmup_chunks, args.measured_chunks) <= 0:
        raise ValueError("chunk and iteration counts must be positive")
    if args.chunk_size % 4:
        raise ValueError("chunk size must be a multiple of four")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(args.input_tensor, args.chunk_size)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    pipeline = SwiftVRPipeline.from_pretrained(
        args.checkpoint,
        revision=_CHECKPOINT_REVISION,
        device="cuda",
        dtype=torch.bfloat16,
        attention_window=(16, 16),
        compile_blocks=args.compile_transformer,
        compile_reae_encoder=args.compile_encoder,
        compile_reae_decoder=args.compile_decoder,
        chunk_size=args.chunk_size,
    )
    model_load_seconds = time.perf_counter() - load_started

    stream = SwiftVRStream(
        pipeline,
        output_height=args.output_height,
        output_width=args.output_width,
        overlap=0,
    )
    prepare_started = time.perf_counter()
    for index in range(args.warmup_chunks):
        stream.step(_chunk(frames, index, args.chunk_size, device))
    stream.step(frames[-1:].to(device, non_blocking=True))
    stream.flush()
    torch.cuda.synchronize(device)
    prepare_seconds = time.perf_counter() - prepare_started

    stream = SwiftVRStream(
        pipeline,
        output_height=args.output_height,
        output_width=args.output_width,
        overlap=0,
    )
    # Populate causal state before measuring the steady path.
    stream.step(_chunk(frames, 0, args.chunk_size, device))
    torch.cuda.synchronize(device)
    samples = []
    for index in range(args.measured_chunks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        stream.step(_chunk(frames, index + 1, args.chunk_size, device))
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    visual = _restore_visual(
        pipeline,
        frames,
        output_height=args.output_height,
        output_width=args.output_width,
        chunk_size=args.chunk_size,
    )
    torch.cuda.synchronize(device)
    output_path = args.output_dir / f"{args.label}.pt"
    torch.save(visual.to(device="cpu", dtype=torch.float16), output_path)

    median_ms = statistics.median(samples)
    result: dict[str, Any] = {
        "label": args.label,
        "checkpoint_revision": _CHECKPOINT_REVISION,
        "input_shape": list(frames.shape),
        "output_shape": list(visual.shape),
        "chunk_size": args.chunk_size,
        "warmup_chunks": args.warmup_chunks,
        "measured_chunks": args.measured_chunks,
        "compile_transformer": args.compile_transformer,
        "compile_encoder": args.compile_encoder,
        "compile_decoder": args.compile_decoder,
        "model_load_seconds": model_load_seconds,
        "prepare_seconds": prepare_seconds,
        "median_chunk_ms": median_ms,
        "p90_chunk_ms": _percentile90(samples),
        "effective_fps": args.chunk_size * 1000 / median_ms,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "samples_ms": samples,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "output_tensor": str(output_path),
    }
    result_path = args.output_dir / f"{args.label}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
