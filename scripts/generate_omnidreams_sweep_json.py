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

    base = {
        "dataset": args.dataset,
        "clip_id": clip_dir.name,
        "hdmap_path": _json_path(hdmap_path, relative_paths=args.relative_paths),
        "first_frame_path": _json_path(
            first_frame_path,
            relative_paths=args.relative_paths,
        ),
        "camera_name": args.camera_name,
        "metadata": {
            "source_dataset_root": _json_path(
                args.data_root,
                relative_paths=args.relative_paths,
            ),
            "split": args.split,
        },
    }

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
