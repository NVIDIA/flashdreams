# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Command-line validation and previews for Crazy Robotaxi maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from omnidreams_game_engine.game_map import (
    compile_game_map,
    load_game_map,
    write_game_map_preview,
    write_spawn_first_frame_preview,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the semantic-map utility parser."""
    parser = argparse.ArgumentParser(prog="crazy-robotaxi-map")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a semantic YAML map")
    validate.add_argument("map", type=Path)
    preview = subparsers.add_parser("preview", help="write a top-down SVG preview")
    preview.add_argument("map", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview_spawn = subparsers.add_parser(
        "preview-spawn", help="write a spawn-aligned synthetic first-frame PNG"
    )
    preview_spawn.add_argument("map", type=Path)
    preview_spawn.add_argument("--output", type=Path, required=True)
    preview_spawn.add_argument("--spawn", help="spawn id (defaults to the first)")
    compile_parser = subparsers.add_parser(
        "compile", help="populate the private renderer cache"
    )
    compile_parser.add_argument("map", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a semantic-map utility command."""
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        game_map = load_game_map(args.map)
        print(
            f"valid: {game_map.name} ({len(game_map.elements)} elements, "
            f"{len(game_map.lanes)} directed lanes)"
        )
    elif args.command == "preview":
        print(write_game_map_preview(args.map, args.output))
    elif args.command == "preview-spawn":
        print(
            write_spawn_first_frame_preview(args.map, args.output, spawn_id=args.spawn)
        )
    else:
        compiled = compile_game_map(args.map)
        print(compiled.archive_path)
