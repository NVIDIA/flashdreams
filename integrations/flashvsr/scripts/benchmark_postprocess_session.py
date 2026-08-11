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

"""Standalone multi-GPU FlashVSR postprocessor benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

_FIRST_CHUNK_FRAMES = 13
"""Frame count for the initial call of the FlashVSR 16-frame chunk mode."""

_STEADY_CHUNK_FRAMES = 16
"""Frame count for steady calls of the FlashVSR 16-frame chunk mode."""


def _parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a warmed FlashVSR full-attention postprocessor session. "
            "Launch this script with torchrun for multi-GPU context parallelism."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=1152)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=16)
    parser.add_argument(
        "--compile-cache-root",
        type=Path,
        default=None,
        help=(
            "Persistent compile-cache root. Defaults to "
            "$FLASHVSR_BENCHMARK_CACHE_ROOT or "
            "$FLASHDREAMS_CACHE_DIR/compile/flashvsr-postprocess."
        ),
    )
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.measured_steps <= 0:
        parser.error("--measured-steps must be positive")
    return args


def _default_cache_root() -> Path:
    """Return the default persistent FlashVSR compile-cache root."""
    explicit = os.environ.get("FLASHVSR_BENCHMARK_CACHE_ROOT")
    if explicit:
        return Path(explicit)
    flashdreams_cache = os.environ.get("FLASHDREAMS_CACHE_DIR")
    if flashdreams_cache:
        return Path(flashdreams_cache) / "compile" / "flashvsr-postprocess"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_parent = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return cache_parent / "flashdreams" / "compile" / "flashvsr-postprocess"


def _configure_compile_cache(cache_root: Path | None) -> tuple[dict[str, str], bool]:
    """Set persistent, rank-scoped compiler cache environment variables.

    Args:
        cache_root: Explicit cache root; ``None`` selects the persistent default.

    Returns:
        Effective compiler cache environment variables and whether this rank's
        Inductor cache contained artifacts before launch.
    """
    root = (cache_root or _default_cache_root()).expanduser().resolve()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank_suffix = Path(f"world-{world_size}") / f"rank-{local_rank}"
    defaults = {
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor" / rank_suffix,
        "TRITON_CACHE_DIR": root / "triton" / rank_suffix,
        "TORCH_EXTENSIONS_DIR": root / "torch-extensions" / rank_suffix,
        "CUDA_CACHE_PATH": root / "cuda" / rank_suffix,
    }
    inductor_path = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", defaults["TORCHINDUCTOR_CACHE_DIR"])
    )
    cache_preexisting = inductor_path.is_dir() and any(inductor_path.iterdir())
    effective: dict[str, str] = {}
    for name, default in defaults.items():
        value = os.environ.setdefault(name, str(default))
        Path(value).mkdir(parents=True, exist_ok=True)
        effective[name] = value
    return effective, cache_preexisting


def _percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile from non-empty values."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _git_commit() -> str:
    """Return the current Git commit, or ``unknown`` outside a checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    """Run the distributed FlashVSR postprocessor benchmark."""
    args = _parse_args()
    process_started = time.perf_counter()

    # Configure caches before importing Torch or FlashVSR so Inductor and
    # Triton observe the persistent paths during their module initialization.
    cache_environment, cache_preexisting = _configure_compile_cache(
        args.compile_cache_root
    )

    import torch
    import torch.distributed as dist
    from flashvsr.postprocess import POSTPROCESS_PRESET_FLASHVSR_V1_1_FULL_ATTN

    from flashdreams.infra.postprocess import VideoChunk, VideoSpec

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = POSTPROCESS_PRESET_FLASHVSR_V1_1_FULL_ATTN

    try:
        setup_started = time.perf_counter()
        session = config.setup().start(
            VideoSpec(height=args.height, width=args.width, fps=args.fps)
        )
        session.prepare()
        dist.barrier()
        prepare_seconds = time.perf_counter() - setup_started

        first = torch.zeros(
            (1, 3, _FIRST_CHUNK_FRAMES, args.height, args.width),
            device=device,
            dtype=torch.bfloat16,
        )
        steady = torch.zeros(
            (1, 3, _STEADY_CHUNK_FRAMES, args.height, args.width),
            device=device,
            dtype=torch.bfloat16,
        )

        assert session.reset()
        session.process(VideoChunk(tensor=first, layout="bcthw"))
        torch.cuda.synchronize()
        for _ in range(args.warmup_steps):
            session.process(VideoChunk(tensor=steady, layout="bcthw"))
            torch.cuda.synchronize()
        dist.barrier()
        warmup_seconds = time.perf_counter() - setup_started - prepare_seconds
        startup_seconds = time.perf_counter() - process_started

        torch.cuda.reset_peak_memory_stats(device)
        records: list[dict[str, float | int]] = []
        output_shape: list[int] | None = None
        for step in range(args.measured_steps):
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = session.process(VideoChunk(tensor=steady, layout="bcthw"))
            torch.cuda.synchronize()
            local_elapsed_ms = (time.perf_counter() - started) * 1000.0

            # Report the slowest rank because every context-parallel call is
            # gated by that rank even when host-side timers differ slightly.
            elapsed = torch.tensor(local_elapsed_ms, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            elapsed_ms = float(elapsed.item())
            output_frames = sum(int(chunk.tensor.shape[2]) for chunk in outputs)
            if outputs:
                output_shape = list(outputs[-1].tensor.shape)
            records.append(
                {
                    "step": step,
                    "elapsed_ms": elapsed_ms,
                    "input_frames": _STEADY_CHUNK_FRAMES,
                    "output_frames": output_frames,
                    "fps": output_frames * 1000.0 / elapsed_ms,
                }
            )

        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peaks: list[float | None] = [None] * world_size
        gpu_names: list[str | None] = [None] * world_size
        cache_environments: list[dict[str, str] | None] = [None] * world_size
        cache_preexisting_by_rank: list[bool | None] = [None] * world_size
        dist.all_gather_object(peaks, peak_mib)
        dist.all_gather_object(gpu_names, torch.cuda.get_device_name(device))
        dist.all_gather_object(cache_environments, cache_environment)
        dist.all_gather_object(cache_preexisting_by_rank, cache_preexisting)

        if rank == 0:
            times = [float(record["elapsed_ms"]) for record in records]
            total_output_frames = sum(
                int(record["output_frames"]) for record in records
            )
            result: dict[str, Any] = {
                "kind": f"standalone_flashvsr_{world_size}gpu",
                "commit": _git_commit(),
                "world_size": world_size,
                "gpu_names_by_rank": gpu_names,
                "input": {
                    "height": args.height,
                    "width": args.width,
                    "frames": _STEADY_CHUNK_FRAMES,
                },
                "output_tensor_shape": output_shape,
                "dtype": "bfloat16",
                "compile_network": config.compile_network,
                "use_cuda_graph": config.use_cuda_graph,
                "compile_cache_by_rank": cache_environments,
                "compile_cache_preexisting_by_rank": cache_preexisting_by_rank,
                "prepare_seconds_excluded": prepare_seconds,
                "warmup_seconds_excluded": warmup_seconds,
                "startup_seconds": startup_seconds,
                "warmup_steps_excluded": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "median_ms": statistics.median(times),
                "p90_ms": _percentile(times, 0.90),
                "median_fps": _STEADY_CHUNK_FRAMES * 1000.0 / statistics.median(times),
                "aggregate_fps": total_output_frames * 1000.0 / sum(times),
                "peak_memory_mib_by_rank": peaks,
                "software": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                },
                "records": records,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
