#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an OmniDreams ``flashdreams-run --batch-inputs-path`` manifest.

Default input layout:

    ~/data/omni-dreams-samples/data/single_view/<clip_id>/
        *_hdmap.mp4
        first_frame.png
        prompt.txt

Example:

    python3 scripts/generate_omnidreams_sweep_json.py \
        --output sweep.json \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = Path("~/data/omni-dreams-samples").expanduser()
DEFAULT_CAMERA_NAME = "camera_front_wide_120fov"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="OmniDreams dataset root. Defaults to ~/data/omni-dreams-samples.",
    )
    parser.add_argument(
        "--split",
        default="single_view",
        help="Dataset split under <data-root>/data/. Defaults to single_view.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sweep.json"),
        help="Output manifest path. Defaults to sweep.json.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="One or more seeds to emit per clip. Defaults to 0.",
    )
    parser.add_argument(
        "--dataset",
        default="omni-dreams-samples",
        help="Dataset label written into each manifest item.",
    )
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help="Camera name written into each single-view manifest item.",
    )
    parser.add_argument(
        "--prompt-id",
        default="vlm",
        help="Prompt id for prompt.txt rows. Defaults to vlm.",
    )
    parser.add_argument(
        "--prompt-source",
        default="vlm-only",
        help="Prompt source metadata for prompt.txt rows. Defaults to vlm-only.",
    )
    parser.add_argument(
        "--include-simple-prompt",
        action="store_true",
        help="Also emit a simple prompt variant for every clip/seed.",
    )
    parser.add_argument(
        "--simple-prompt-id",
        default="simple",
        help="Prompt id for --include-simple-prompt rows.",
    )
    parser.add_argument(
        "--simple-prompt",
        default="Urban driving scene from a front-facing dashcam.",
        help="Prompt text for --include-simple-prompt rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only emit the first N valid clips after sorting. 0 means all.",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        default=0,
        help=(
            "Only write the first N manifest items after seed/prompt expansion. "
            "0 means all. Useful for smoke tests."
        ),
    )
    parser.add_argument(
        "--match-hdmap-duration",
        action="store_true",
        help=(
            "Write per-item total_blocks from each HDMap frame count. By "
            "default this uses enough chunks to cover the whole HDMap; pass "
            "--pad-final-hdmap-chunk True to flashdreams-run so the runner "
            "pads the final conditioning chunk and crops the saved video back "
            "to the original HDMap length."
        ),
    )
    parser.add_argument(
        "--duration-block-mode",
        choices=("ceil", "floor"),
        default="ceil",
        help=(
            "How to map HDMap frame counts to chunk counts. ceil covers the "
            "whole HDMap and may require final-chunk padding; floor uses only "
            "the longest exact chunked prefix. Default: ceil."
        ),
    )
    parser.add_argument(
        "--first-chunk-frames",
        type=int,
        default=5,
        help=(
            "Decoded frame count for AR chunk 0 when --match-hdmap-duration "
            "is enabled. Default 5 for len_t=2 with temporal compression 4."
        ),
    )
    parser.add_argument(
        "--next-chunk-frames",
        type=int,
        default=8,
        help=(
            "Decoded frame count for AR chunks after chunk 0 when "
            "--match-hdmap-duration is enabled. Default 8 for len_t=2 with "
            "temporal compression 4."
        ),
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable used for --match-hdmap-duration.",
    )
    parser.add_argument(
        "--strict-hdmap-duration",
        action="store_true",
        help=(
            "Fail if an HDMap frame count cannot be represented exactly by "
            "the configured first/next chunk frame schedule."
        ),
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to the current directory instead of absolute paths.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def _json_path(path: Path, *, relative_paths: bool) -> str:
    path = path.expanduser()
    if relative_paths:
        return str(path)
    return str(path.resolve())


def _find_one(path: Path, pattern: str, *, name: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    if not matches:
        print(f"[skip] {path.name}: missing {name} ({pattern})")
        return None
    if len(matches) > 1:
        print(f"[warn] {path.name}: multiple {name} matches; using {matches[0].name}")
    return matches[0]


def _probe_video_frames(path: Path, *, ffprobe_bin: str) -> int:
    """Return the best available video frame count from ffprobe."""
    common_cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(common_cmd, text=True))
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"no video stream found: {path}")
    stream = streams[0]
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A", ""):
            frames = int(value)
            if frames > 0:
                return frames
    raise ValueError(f"could not determine frame count for {path}")


def _chunked_blocks_for_frames(
    frame_count: int,
    *,
    first_chunk_frames: int,
    next_chunk_frames: int,
    mode: str,
) -> tuple[int, int, int]:
    """Return ``(total_blocks, chunk_schedule_frames, frame_delta)``.

    ``frame_delta`` is ``chunk_schedule_frames - frame_count``. Positive values
    mean final-chunk padding/cropping is needed; negative values mean the floor
    schedule leaves HDMap frames unused.
    """
    if first_chunk_frames <= 0:
        raise ValueError("--first-chunk-frames must be > 0")
    if next_chunk_frames <= 0:
        raise ValueError("--next-chunk-frames must be > 0")
    if frame_count < first_chunk_frames:
        if mode == "ceil":
            return 1, first_chunk_frames, first_chunk_frames - frame_count
        return 0, 0, -frame_count
    remaining = frame_count - first_chunk_frames
    if mode == "ceil":
        extra_blocks = (remaining + next_chunk_frames - 1) // next_chunk_frames
    else:
        extra_blocks = remaining // next_chunk_frames
    total_blocks = 1 + extra_blocks
    chunk_schedule_frames = first_chunk_frames + extra_blocks * next_chunk_frames
    return total_blocks, chunk_schedule_frames, chunk_schedule_frames - frame_count


def _clip_items(
    *,
    clip_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    hdmap_path = _find_one(clip_dir, "*_hdmap.mp4", name="HDMap video")
    first_frame_path = clip_dir / "first_frame.png"
    prompt_path = clip_dir / "prompt.txt"
    if hdmap_path is None:
        return []
    if not first_frame_path.exists():
        print(f"[skip] {clip_dir.name}: missing first_frame.png")
        return []
    if not prompt_path.exists():
        print(f"[skip] {clip_dir.name}: missing prompt.txt")
        return []

    metadata: dict[str, Any] = {
        "source_dataset_root": _json_path(
            args.data_root,
            relative_paths=args.relative_paths,
        ),
        "split": args.split,
    }
    total_blocks: int | None = None
    if args.match_hdmap_duration:
        hdmap_frame_count = _probe_video_frames(hdmap_path, ffprobe_bin=args.ffprobe_bin)
        total_blocks, chunk_schedule_frames, frame_delta = _chunked_blocks_for_frames(
            hdmap_frame_count,
            first_chunk_frames=args.first_chunk_frames,
            next_chunk_frames=args.next_chunk_frames,
            mode=args.duration_block_mode,
        )
        if total_blocks <= 0:
            print(
                f"[skip] {clip_dir.name}: HDMap has only {hdmap_frame_count} "
                f"frame(s), shorter than first chunk"
            )
            return []
        if args.strict_hdmap_duration and frame_delta:
            raise SystemExit(
                f"{clip_dir.name}: HDMap has {hdmap_frame_count} frames, but "
                f"chunk schedule generates {chunk_schedule_frames}; "
                f"delta={frame_delta} frame(s)."
            )
        if frame_delta:
            action = "pad/crop" if frame_delta > 0 else "leave unused"
            print(
                f"[warn] {clip_dir.name}: HDMap has {hdmap_frame_count} frames; "
                f"chunk schedule is {chunk_schedule_frames} "
                f"({action} {abs(frame_delta)} frame(s))"
            )
        metadata.update(
            {
                "hdmap_frame_count": hdmap_frame_count,
                "hdmap_duration_mode": "chunked_cover"
                if args.duration_block_mode == "ceil"
                else "chunked_prefix",
                "duration_block_mode": args.duration_block_mode,
                "expected_output_frame_count": hdmap_frame_count
                if args.duration_block_mode == "ceil"
                else chunk_schedule_frames,
                "chunk_schedule_frame_count": chunk_schedule_frames,
                "final_chunk_pad_frame_count": max(frame_delta, 0),
                "unused_hdmap_frame_count": max(-frame_delta, 0),
                "first_chunk_frames": args.first_chunk_frames,
                "next_chunk_frames": args.next_chunk_frames,
            }
        )

    base = {
        "dataset": args.dataset,
        "clip_id": clip_dir.name,
        "hdmap_path": _json_path(hdmap_path, relative_paths=args.relative_paths),
        "first_frame_path": _json_path(
            first_frame_path,
            relative_paths=args.relative_paths,
        ),
        "camera_name": args.camera_name,
        "metadata": metadata,
    }
    if total_blocks is not None:
        base["total_blocks"] = total_blocks

    items: list[dict[str, Any]] = []
    for seed in args.seeds:
        items.append(
            {
                **base,
                "prompt_id": args.prompt_id,
                "prompt_source": args.prompt_source,
                "prompt_path": _json_path(
                    prompt_path,
                    relative_paths=args.relative_paths,
                ),
                "seed": seed,
            }
        )
        if args.include_simple_prompt:
            items.append(
                {
                    **base,
                    "prompt_id": args.simple_prompt_id,
                    "prompt_source": "simple",
                    "prompt": args.simple_prompt,
                    "seed": seed,
                }
            )
    return items


def main() -> None:
    args = _parse_args()
    args.data_root = args.data_root.expanduser()
    split_root = args.data_root / "data" / args.split
    if not split_root.is_dir():
        raise SystemExit(f"Input split directory not found: {split_root}")

    clip_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
    if args.limit > 0:
        clip_dirs = clip_dirs[: args.limit]

    items: list[dict[str, Any]] = []
    for clip_dir in clip_dirs:
        items.extend(_clip_items(clip_dir=clip_dir, args=args))
        if args.limit_items > 0 and len(items) >= args.limit_items:
            items = items[: args.limit_items]
            break

    payload = {"items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent == 0 else args.indent
    args.output.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")

    print(
        f"Wrote {len(items)} item(s) from {len(clip_dirs)} clip dir(s) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
