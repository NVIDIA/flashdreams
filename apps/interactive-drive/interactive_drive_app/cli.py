# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line launcher for discovered plug-compatible driving adapters."""

from __future__ import annotations

import argparse

from flashdreams.runtime.demo import DemoRoute, DemoSpec, LocalWindowOutputSpec
from flashdreams.runtime.demo.registry import discover_demo_adapters

from interactive_drive_app.application import InteractiveDriveApplication


def build_parser() -> argparse.ArgumentParser:
    """Build the plug-compatible interactive-drive argument parser."""
    parser = argparse.ArgumentParser(prog="flashdreams-interactive-drive")
    parser.add_argument("--model-id")
    parser.add_argument("--preset-id")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument("--window-title", default="FlashDreams interactive drive")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Discover a compatible adapter and run its default driving session."""
    parser = build_parser()
    args = parser.parse_args(argv)
    adapters = discover_demo_adapters()
    compatible = {
        model_id: adapter
        for model_id, adapter in adapters.items()
        if DemoRoute(
            input_mode="keyboard-driving",
            output_mode="local-window",
        )
        in adapter.supported_routes()
    }
    if args.list_models:
        for model_id in sorted(compatible):
            print(model_id)
        return
    if not args.model_id:
        parser.error("--model-id is required unless --list-models is used.")
    adapter = compatible.get(args.model_id)
    if adapter is None:
        parser.error(
            f"model {args.model_id!r} is not installed or does not support "
            "keyboard-driving + local-window."
        )

    spec = DemoSpec(
        model_id=args.model_id,
        input_mode="keyboard-driving",
        preset_id=args.preset_id,
        output=LocalWindowOutputSpec(
            width=args.window_width,
            height=args.window_height,
            title=args.window_title,
        ),
    )
    sessions = adapter.list_sessions(spec)
    if not sessions:
        raise RuntimeError(f"Adapter {adapter.model_id!r} returned no sessions.")
    index = 0
    app = InteractiveDriveApplication(adapter=adapter, initial_spec=sessions[index])
    try:
        while True:
            selected = sessions[index]
            outcome = app.run_session(
                spec=selected,
                session_id=f"session-{index}",
            )
            if outcome.action == "reset":
                continue
            if outcome.action == "next":
                index = (index + 1) % len(sessions)
                continue
            if outcome.action == "previous":
                index = (index - 1) % len(sessions)
                continue
            if outcome.action in {"exit", "closed", "completed", "stopped"}:
                break
    finally:
        app.close()


__all__ = ["build_parser", "main"]
