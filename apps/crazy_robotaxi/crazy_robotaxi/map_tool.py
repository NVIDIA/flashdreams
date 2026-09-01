# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline validation, compilation, and preview commands for game maps."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from omnidreams_game_engine.game_map import (
    compile_game_map,
    load_game_map,
    write_game_map_preview,
    write_spawn_first_frame_preview,
)

from crazy_robotaxi.spawn_images import generate_spawn_images


def main(argv: Sequence[str] | None = None) -> None:
    """Run one map-authoring command without constructing a model."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        game_map = load_game_map(args.map)
        print(f"valid: {game_map.map_id}")
        return
    if args.command == "compile":
        result = compile_game_map(args.map, force=args.force_map_recompile)
        print(result.archive_path)
        return
    if args.command == "preview":
        write_game_map_preview(args.map, args.output)
        print(args.output)
        return
    if args.command == "preview-spawn":
        write_spawn_first_frame_preview(
            args.map,
            args.output,
            spawn_id=args.spawn,
        )
        print(args.output)
        return
    if args.command == "generate-spawns":
        from qwen_image_edit_v2 import QwenImageEditor

        output = generate_spawn_images(
            args.map,
            QwenImageEditor(device=args.device),
            output_map=args.output_map,
            resolution_wh=(args.width, args.height),
            force=args.force,
            num_inference_steps=args.steps,
            progress=lambda index, total, spawn, variant: print(
                f"generating {index}/{total}: {spawn}/{variant}", flush=True
            ),
        )
        print(output)
        return
    parser.error(f"Unknown command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crazy-robotaxi-map")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("map", type=Path)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("map", type=Path)
    compile_parser.add_argument("--force-map-recompile", action="store_true")
    preview = subparsers.add_parser("preview")
    preview.add_argument("map", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    spawn = subparsers.add_parser("preview-spawn")
    spawn.add_argument("map", type=Path)
    spawn.add_argument("--spawn", required=True)
    spawn.add_argument("--output", type=Path, required=True)
    generate = subparsers.add_parser(
        "generate-spawns",
        help="generate Qwen-only spawn drafts without world-model settlement",
    )
    generate.add_argument("map", type=Path)
    generate.add_argument("--output-map", type=Path)
    generate.add_argument("--width", type=int, default=1280)
    generate.add_argument("--height", type=int, default=704)
    generate.add_argument("--steps", type=int)
    generate.add_argument("--device", default="cuda")
    generate.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    main()
