# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and visually validate LongSANA with a diverse prompt suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import mediapy as media
import torch

from longsana.config import PIPELINE_LONGSANA_2B_480P
from longsana.impl.constants import DEFAULT_VIDEO_FPS
from longsana.impl.transformer import LongSanaTransformerCache


@dataclass(frozen=True, kw_only=True)
class ValidationCase:
    """One prompt and rollout length in the validation matrix."""

    slug: str
    prompt: str
    blocks: int
    seed: int
    category: str


DEFAULT_PROMPTS = (
    (
        "animal_motion",
        "A red panda walks briskly through a misty bamboo forest at sunrise, "
        "tracking shot, realistic fur moving in the wind.",
        "subject identity and articulated motion",
    ),
    (
        "dance_camera",
        "Three street dancers perform fast synchronized choreography in a neon "
        "alley at night while the camera smoothly circles them, cinematic.",
        "multiple subjects, fast motion, and camera orbit",
    ),
    (
        "cooking_interaction",
        "A chef flips vegetables in a flaming wok, pours sauce, and plates the "
        "dish in a busy restaurant kitchen, close-up documentary camera.",
        "object interaction and ordered actions",
    ),
    (
        "coastal_aerial",
        "A continuous aerial shot flies over sea cliffs toward a lighthouse as "
        "waves crash below and storm clouds roll across the coast, photorealistic.",
        "large camera translation and scene continuity",
    ),
    (
        "surreal_long",
        "An astronaut riding a white horse crosses a moonlit salt flat; reflections "
        "ripple under the hooves as the camera follows from behind, cinematic.",
        "long-horizon composition and identity",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/longsana_benchmark"),
    )
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--long-blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only this case slug; repeat to select more than one.",
    )
    parser.add_argument(
        "--no-contact-sheets",
        action="store_true",
        help="Skip ffmpeg contact-sheet creation.",
    )
    args = parser.parse_args()
    if args.blocks <= 0 or args.long_blocks <= 0:
        parser.error("--blocks and --long-blocks must be positive")
    return args


def _cases(args: argparse.Namespace) -> list[ValidationCase]:
    selected = set(args.cases or ())
    known = {slug for slug, _prompt, _category in DEFAULT_PROMPTS}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown validation case(s): {', '.join(sorted(unknown))}")
    return [
        ValidationCase(
            slug=slug,
            prompt=prompt,
            blocks=args.long_blocks if slug == "surreal_long" else args.blocks,
            seed=args.seed + index,
            category=category,
        )
        for index, (slug, prompt, category) in enumerate(DEFAULT_PROMPTS)
        if not selected or slug in selected
    ]


def _gpu_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "name": properties.name,
                "total_memory_gib": properties.total_memory / 1024**3,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    return info


def _contact_sheet(video_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=2,scale=416:240,tile=4x5",
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )


def _reset_generator(pipeline: Any, seed: int) -> None:
    pipeline.diffusion_model._rng = torch.Generator(  # noqa: SLF001
        device=pipeline.device
    ).manual_seed(seed)


def _run_case(
    pipeline: Any,
    case: ValidationCase,
    output_dir: Path,
    *,
    contact_sheets: bool,
) -> dict[str, Any]:
    _reset_generator(pipeline, case.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompt_started = time.perf_counter()
    cache = pipeline.initialize_cache(text=[case.prompt])
    prompt_encode_s = time.perf_counter() - prompt_started
    if not isinstance(cache.transformer_cache, LongSanaTransformerCache):
        raise TypeError("Benchmark requires a LongSanaTransformerCache.")

    chunks: list[torch.Tensor] = []
    block_metrics: list[dict[str, float]] = []
    cache_mib_history: list[float] = []
    for index in range(case.blocks):
        frames = pipeline.generate(index, cache)
        metrics = pipeline.finalize(index, cache)
        if metrics is None:
            raise RuntimeError("LongSANA profiling must be enabled for this benchmark.")
        if not bool(torch.isfinite(frames).all()):
            raise RuntimeError(f"Non-finite video output in {case.slug} block {index}.")
        chunks.append(frames.detach().cpu())
        block_metrics.append(metrics)
        cache_mib_history.append(cache.transformer_cache.state_bytes() / 1024**2)

    video = torch.cat(chunks).clamp(0, 1)
    video_path = output_dir / f"{case.slug}_{case.blocks}blocks.mp4"
    media.write_video(
        video_path,
        video.permute(0, 2, 3, 1).contiguous().numpy(),
        fps=DEFAULT_VIDEO_FPS,
    )
    if contact_sheets:
        _contact_sheet(
            video_path,
            output_dir / f"{case.slug}_{case.blocks}blocks_contact.jpg",
        )

    steady = block_metrics[1:] or block_metrics
    steady_total = [item["total_ms"] for item in steady]
    steady_frames = sum(int(chunk.shape[0]) for chunk in chunks[1:] or chunks)
    steady_elapsed_ms = sum(steady_total)
    result = {
        **asdict(case),
        "video": str(video_path.resolve()),
        "prompt_encode_s": prompt_encode_s,
        "frames": int(video.shape[0]),
        "shape": list(video.shape),
        "finite": True,
        "cache_mib_history": cache_mib_history,
        "cache_constant": len({round(value, 6) for value in cache_mib_history}) == 1,
        "block_metrics": block_metrics,
        "steady_state": {
            "blocks": len(steady),
            "total_ms_median": statistics.median(steady_total),
            "total_ms_mean": statistics.mean(steady_total),
            "end_to_end_fps": steady_frames / (steady_elapsed_ms / 1000),
            "diffusion_fps": steady_frames
            / (sum(item["diffuse_ms"] + item["finalize_ms"] for item in steady) / 1000),
            "decode_ms_median": statistics.median(item["decode_ms"] for item in steady),
            "peak_memory_gib": max(item["mem_peak_gib"] for item in steady),
        },
    }
    case_path = output_dir / f"{case.slug}_{case.blocks}blocks.json"
    case_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run selected validation cases through the public LongSANA pipeline."""
    args = _parse_args()
    cases = _cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The released LongSANA benchmark requires a CUDA GPU.")

    setup_started = time.perf_counter()
    pipeline = PIPELINE_LONGSANA_2B_480P.setup().to(args.device).eval()
    setup_s = time.perf_counter() - setup_started
    try:
        results = [
            _run_case(
                pipeline,
                case,
                args.output_dir,
                contact_sheets=not args.no_contact_sheets,
            )
            for case in cases
        ]
    finally:
        pipeline.close()

    steady = [result["steady_state"] for result in results]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": PIPELINE_LONGSANA_2B_480P.name,
        "setup_s": setup_s,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "hardware": _gpu_info(args.device),
        "cases": results,
        "aggregate_steady_state": {
            "end_to_end_fps_median": statistics.median(
                item["end_to_end_fps"] for item in steady
            ),
            "diffusion_fps_median": statistics.median(
                item["diffusion_fps"] for item in steady
            ),
            "total_ms_median": statistics.median(
                item["total_ms_median"] for item in steady
            ),
            "peak_memory_gib_max": max(item["peak_memory_gib"] for item in steady),
            "cache_mib": results[0]["cache_mib_history"][-1],
            "all_outputs_finite": all(result["finite"] for result in results),
            "all_caches_constant": all(result["cache_constant"] for result in results),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate_steady_state"], indent=2))
    print(f"Results: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
