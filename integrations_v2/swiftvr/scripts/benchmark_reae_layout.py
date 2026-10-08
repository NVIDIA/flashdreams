# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark experimental SwiftVR ReAE memory layouts in a fresh process."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from swiftvr.config import _resolve_checkpoint
from swiftvr.impl.decoder.network import SwiftVRTAEHV, _SwiftVRCompiledDecoder
from swiftvr.impl.encoder import SwiftVREncoder, SwiftVREncoderConfig
from torch import Tensor, nn

from flashdreams.infra.compile import compile_module
from flashdreams.infra.cuda_graph import set_or_copy
from flashdreams.recipes.taehv.impl import Encoder, MemBlock, TPool

_CHECKPOINT_REVISION = "743ed2530c550764905400f38eb6cc41af5abc80"
_Layout = Literal["contiguous", "channels-last", "channels-last-3d", "combined"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("encoder", "decoder"), required=True)
    parser.add_argument("--layout", choices=_Layout.__args__, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--input-tensor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="H-oliday/SwiftVR")
    parser.add_argument("--output-height", type=int, default=1408)
    parser.add_argument("--output-width", type=int, default=2560)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--warmup-chunks", type=int, default=5)
    parser.add_argument("--measured-chunks", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def _percentile90(samples: list[float]) -> float:
    return sorted(samples)[max(0, int(0.9 * len(samples)) - 1)]


def _uses_channels_last(layout: _Layout) -> bool:
    return layout in ("channels-last", "combined")


def _uses_channels_last_3d(layout: _Layout) -> bool:
    return layout in ("channels-last-3d", "combined")


def _convert_weights(module: nn.Module, layout: _Layout) -> None:
    temporal_convs = {
        id(child)
        for boundary in module.modules()
        if isinstance(boundary, TPool)
        for child in boundary.modules()
        if isinstance(child, nn.Conv2d)
    }
    if _uses_channels_last(layout):
        for child in module.modules():
            if isinstance(child, nn.Conv2d) and id(child) not in temporal_convs:
                child.weight.data = child.weight.data.contiguous(
                    memory_format=torch.channels_last
                )


def _memblock_step(
    block: MemBlock,
    tensor: Tensor,
    state: dict[int, Tensor],
    batch: int,
    *,
    channels_last: bool,
) -> Tensor:
    key = id(block)
    bt, channels, height, width = tensor.shape
    time_steps = bt // batch
    video = tensor.reshape(batch, time_steps, channels, height, width)
    if key not in state:
        state[key] = tensor.new_zeros(batch, 1, channels, height, width)
    past = torch.cat([state[key], video[:, :-1]], dim=1).reshape_as(tensor)
    set_or_copy(state, key, video[:, -1:])
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
        past = past.contiguous(memory_format=torch.channels_last)
    return block(tensor, past)


class _EncoderRunner(nn.Module):
    def __init__(self, network: Encoder, *, channels_last: bool) -> None:
        super().__init__()
        self.network = network
        self.channels_last = channels_last

    def forward(self, tensor: Tensor, state: dict[int, Tensor]) -> Tensor:
        batch, time_steps, channels, height, width = tensor.shape
        tensor = tensor.reshape(batch * time_steps, channels, height, width)
        if self.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        for block in self.network.blocks:
            if isinstance(block, MemBlock):
                tensor = _memblock_step(
                    block,
                    tensor,
                    state,
                    batch,
                    channels_last=self.channels_last,
                )
            elif isinstance(block, TPool):
                tensor = block(tensor.contiguous())
                if self.channels_last:
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
            else:
                tensor = block(tensor)
        tensor = tensor.contiguous()
        _, channels, height, width = tensor.shape
        return tensor.reshape(batch, -1, channels, height, width)


def _load_frames(
    path: Path,
    *,
    output_height: int,
    output_width: int,
    chunk_size: int,
    device: torch.device,
) -> Tensor:
    frames = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(frames, Tensor):
        raise TypeError(f"expected a tensor in {path}, got {type(frames).__name__}")
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
        raise ValueError(
            "encoder input must be uint8 [T,H,W,3], got "
            f"shape={tuple(frames.shape)} dtype={frames.dtype}"
        )
    if frames.shape[0] < chunk_size or frames.shape[0] % chunk_size:
        raise ValueError(
            f"frame count must be a positive multiple of {chunk_size}, "
            f"got {frames.shape[0]}"
        )
    frames = frames.to(device=device, dtype=torch.bfloat16).permute(0, 3, 1, 2)
    frames = F.interpolate(
        frames,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    ).div_(255)
    frames = F.pixel_unshuffle(frames, 2)
    return frames.reshape(1, frames.shape[0], *frames.shape[1:])


def _load_latents(path: Path, chunk_size: int, device: torch.device) -> Tensor:
    latents = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(latents, Tensor) or latents.ndim != 5:
        raise ValueError(f"decoder input must be a 5D tensor, got {type(latents)}")
    latent_chunk = chunk_size // 4
    if latents.shape[1] < latent_chunk or latents.shape[1] % latent_chunk:
        raise ValueError(
            f"latent frame count must be a multiple of {latent_chunk}, "
            f"got {latents.shape[1]}"
        )
    return latents.to(device=device, dtype=torch.bfloat16)


def _measure(
    step: Any,
    chunks: list[Tensor],
    *,
    new_state: Any,
    warmup_chunks: int,
    measured_chunks: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    state = new_state()
    started = time.perf_counter()
    for index in range(warmup_chunks):
        step(chunks[index % len(chunks)], state)
    torch.cuda.synchronize(device)
    prepare_seconds = time.perf_counter() - started

    state = new_state()
    step(chunks[0], state)
    torch.cuda.synchronize(device)
    samples = []
    for index in range(measured_chunks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step(chunks[(index + 1) % len(chunks)], state)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return prepare_seconds, samples


@torch.inference_mode()
def _benchmark_encoder(
    args: argparse.Namespace,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[float, list[float], Tensor]:
    encoder = SwiftVREncoder(
        SwiftVREncoderConfig(
            checkpoint_path=checkpoint_path,
            dtype=torch.bfloat16,
            use_compile=False,
        )
    ).to(device=device, dtype=torch.bfloat16)
    _convert_weights(encoder.network, args.layout)
    runner: nn.Module = _EncoderRunner(
        encoder.network,
        channels_last=_uses_channels_last(args.layout),
    )
    if args.compile:
        runner = compile_module(runner, mode="default")
    packed = _load_frames(
        args.input_tensor,
        output_height=args.output_height,
        output_width=args.output_width,
        chunk_size=args.chunk_size,
        device=device,
    )
    chunks = list(packed.split(args.chunk_size, dim=1))

    def step(chunk: Tensor, state: dict[int, Tensor]) -> Tensor:
        return runner(chunk, state)

    prepare_seconds, samples = _measure(
        step,
        chunks,
        new_state=dict,
        warmup_chunks=args.warmup_chunks,
        measured_chunks=args.measured_chunks,
        device=device,
    )
    state: dict[int, Tensor] = {}
    output = torch.cat([step(chunk, state) for chunk in chunks], dim=1)
    return prepare_seconds, samples, output


@torch.inference_mode()
def _benchmark_decoder(
    args: argparse.Namespace,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[float, list[float], Tensor]:
    network = SwiftVRTAEHV(checkpoint_path, use_compile=False).to(
        device=device, dtype=torch.bfloat16
    )
    runner: nn.Module = _SwiftVRCompiledDecoder(
        network.decoder,
        channels_last=_uses_channels_last(args.layout),
        channels_last_3d=_uses_channels_last_3d(args.layout),
    )
    if args.compile:
        runner = compile_module(runner)
    latents = _load_latents(args.input_tensor, args.chunk_size, device)
    chunks = list(latents.split(args.chunk_size // 4, dim=1))

    def new_state() -> dict[int, Tensor]:
        return {}

    def step(chunk: Tensor, state: dict[int, Tensor]) -> Tensor:
        first = not state
        if first:
            batch, time_steps, channels, height, width = chunk.shape
            network.decoder.initialize_state(
                (batch, time_steps, channels, height, width),
                chunk.dtype,
                chunk.device,
                state,
            )
        output = runner(chunk, state, chunk.shape[0]).clamp_(0, 1)
        n, time_steps, channels, height, width = output.shape
        output = F.pixel_shuffle(
            output.reshape(n * time_steps, channels, height, width),
            network.patch_size,
        ).reshape(n, time_steps, 3, height * 2, width * 2)
        return output[:, network.frames_to_trim :] if first else output

    prepare_seconds, samples = _measure(
        step,
        chunks,
        new_state=new_state,
        warmup_chunks=args.warmup_chunks,
        measured_chunks=args.measured_chunks,
        device=device,
    )
    state = new_state()
    output = torch.cat([step(chunk, state) for chunk in chunks], dim=1)
    return prepare_seconds, samples, output


def main() -> None:
    args = _arguments()
    if min(args.chunk_size, args.warmup_chunks, args.measured_chunks) <= 0:
        raise ValueError("chunk and iteration counts must be positive")
    if args.chunk_size % 4:
        raise ValueError("chunk size must be a multiple of four")
    if args.stage == "encoder" and _uses_channels_last_3d(args.layout):
        raise ValueError("the encoder has no Conv3d path")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = _resolve_checkpoint(
        args.checkpoint,
        revision=_CHECKPOINT_REVISION,
    )
    checkpoint_path = str(checkpoint_root / "reae.safetensors")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.stage == "encoder":
        prepare_seconds, samples, output = _benchmark_encoder(
            args, checkpoint_path, device
        )
    else:
        prepare_seconds, samples, output = _benchmark_decoder(
            args, checkpoint_path, device
        )
    total_seconds = time.perf_counter() - started
    torch.cuda.synchronize(device)

    output_path = args.output_dir / f"{args.label}.pt"
    torch.save(output.to(device="cpu", dtype=torch.float16), output_path)
    median_ms = statistics.median(samples)
    result: dict[str, Any] = {
        "label": args.label,
        "stage": args.stage,
        "layout": args.layout,
        "compiled": args.compile,
        "input_shape": list(
            torch.load(args.input_tensor, map_location="cpu", weights_only=True).shape
        ),
        "output_shape": list(output.shape),
        "chunk_size": args.chunk_size,
        "warmup_chunks": args.warmup_chunks,
        "measured_chunks": args.measured_chunks,
        "prepare_seconds": prepare_seconds,
        "total_seconds": total_seconds,
        "median_chunk_ms": median_ms,
        "p90_chunk_ms": _percentile90(samples),
        "effective_fps": args.chunk_size * 1000 / median_ms,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "samples_ms": samples,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "checkpoint_revision": _CHECKPOINT_REVISION,
        "output_tensor": str(output_path),
    }
    result_path = args.output_dir / f"{args.label}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
