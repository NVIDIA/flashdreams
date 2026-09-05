# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture an operator profile for one warmed LongSANA Runtime V2 block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from longsana.config import PIPELINE_LONGSANA_2B_480P
from longsana.impl.transformer import LongSanaTransformerCache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/longsana_profile"),
    )
    parser.add_argument(
        "--prompt",
        default="A red panda walks through a misty bamboo forest, tracking shot.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Warm one block, then profile the complete next-block lifecycle."""
    args = _parse_args()
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The LongSANA operator profile requires a CUDA GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PIPELINE_LONGSANA_2B_480P.setup().to(args.device).eval()
    try:
        cache = pipeline.initialize_cache(text=[args.prompt])
        _ = pipeline.generate(0, cache)
        _ = pipeline.finalize(0, cache)
        torch.cuda.reset_peak_memory_stats()

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            frames = pipeline.generate(1, cache)
            metrics = pipeline.finalize(1, cache)

        if metrics is None:
            raise RuntimeError("LongSANA profiling must be enabled.")
        transformer_cache = cache.transformer_cache
        if not isinstance(transformer_cache, LongSanaTransformerCache):
            raise TypeError("Profile requires a LongSanaTransformerCache.")

        trace_path = args.output_dir / "steady_state_trace.json"
        table_path = args.output_dir / "steady_state_operators.txt"
        summary_path = args.output_dir / "summary.json"
        profile.export_chrome_trace(str(trace_path))
        table = profile.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
        table_path.write_text(table + "\n")
        summary = {
            "pipeline": PIPELINE_LONGSANA_2B_480P.name,
            "prompt": args.prompt,
            "profiled_block": 1,
            "frames": int(frames.shape[0]),
            "shape": list(frames.shape),
            "finite": bool(torch.isfinite(frames).all()),
            "cache_mib": transformer_cache.state_bytes() / 1024**2,
            "metrics": metrics,
            "trace": str(trace_path.resolve()),
            "operators": str(table_path.resolve()),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(table)
        print(f"Results: {summary_path.resolve()}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
