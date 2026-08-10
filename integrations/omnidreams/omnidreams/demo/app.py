# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the experimental shared OmniDreams demo path."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from omnidreams.runner import DEFAULT_EXAMPLE_DATA_UUID_1V

from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.app import DemoApplication

from .adapter import OmnidreamsDemoAdapter
from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    OMNIDREAMS_CONDITIONING_LUDUS,
    OMNIDREAMS_CONDITIONING_MODES,
    OMNIDREAMS_CONDITIONING_PRECOMPUTED,
    OMNIDREAMS_MODEL_ID,
    OmnidreamsWebRTCScenario,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental OmniDreams demo using flashdreams.runtime.demo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Run a finite replay demo.")
    replay.add_argument("--preset-id", default=DEFAULT_OMNIDREAMS_PRESET)
    replay.add_argument("--device", default="cuda")
    replay.add_argument("--seed", type=int, default=42)
    replay.add_argument(
        "--conditioning-mode",
        choices=OMNIDREAMS_CONDITIONING_MODES,
        default=OMNIDREAMS_CONDITIONING_PRECOMPUTED,
    )
    replay.add_argument("--prompt", default=None)
    replay.add_argument("--hdmap-video-paths", type=_split_paths, default=())
    replay.add_argument("--first-frame-paths", type=_split_paths, default=())
    replay.add_argument("--camera-names", type=_split_strings, default=())
    replay.add_argument("--keyboard-trace", type=Path, default=None)
    replay.add_argument("--scene-path", type=Path, default=None)
    replay.add_argument("--scene-dir", type=Path, default=None)
    replay.add_argument("--scene-uuid", default=DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID)
    replay.add_argument("--scene-variant", default="default")
    replay.add_argument("--camera-name", default="camera_front_wide_120fov")
    replay.add_argument("--move-speed-per-s", type=float, default=6.0)
    replay.add_argument(
        "--rotate-speed-rad-per-s",
        type=float,
        default=math.radians(35.0),
    )
    replay.add_argument("--ludus-backend", choices=("cuda", "vulkan"), default="cuda")
    replay.add_argument(
        "--example-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the bundled single-view HF sample when asset paths are omitted "
            "(default: auto)."
        ),
    )
    replay.add_argument("--example-data-uuid", default=DEFAULT_EXAMPLE_DATA_UUID_1V)
    replay.add_argument("--total-blocks", type=int, default=60)
    replay.add_argument("--pixel-height", type=int, default=704)
    replay.add_argument("--pixel-width", type=int, default=1280)
    replay.add_argument("--fps", type=int, default=30)
    replay.add_argument("--output-mode", choices=("mp4", "null"), default="mp4")
    replay.add_argument("--output", type=Path, default=None)

    webrtc = subparsers.add_parser("webrtc", help="Serve a WebRTC driving demo.")
    webrtc.add_argument("--preset-id", default=DEFAULT_OMNIDREAMS_PRESET)
    webrtc.add_argument("--host", default="0.0.0.0")
    webrtc.add_argument("--port", type=int, default=8082)
    webrtc.add_argument("--device", default="cuda:0")
    webrtc.add_argument("--seed", type=int, default=42)
    webrtc.add_argument("--scene-dir", type=Path, default=None)
    webrtc.add_argument("--scene-uuid", default=None)
    webrtc.add_argument("--scene-variant", default="default")
    webrtc.add_argument("--camera-name", default="camera_front_wide_120fov")
    webrtc.add_argument("--fps", type=int, default=30)
    webrtc.add_argument("--video-height", type=int, default=704)
    webrtc.add_argument("--video-width", type=int, default=1280)
    webrtc.add_argument("--warmup-chunks", type=int, default=10)
    webrtc.add_argument("--warmup-timeout-s", type=float, default=600.0)
    webrtc.add_argument("--client-liveness-timeout-s", type=float, default=10.0)
    webrtc.add_argument("--debug-serve-hdmaps", action="store_true")
    webrtc.add_argument("--prefer-sw-encoder", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "replay":
        if args.output_mode == "mp4" and args.output is None:
            parser.error("replay --output is required when --output-mode=mp4.")
        if args.output_mode == "null" and args.output is not None:
            parser.error("replay --output is only valid when --output-mode=mp4.")
        if (
            args.conditioning_mode == OMNIDREAMS_CONDITIONING_LUDUS
            and args.keyboard_trace is None
        ):
            parser.error(
                "replay --keyboard-trace is required when "
                "--conditioning-mode=ludus-scene-driving."
            )
    return args


class OmnidreamsDemoApplication(DemoApplication):
    """OmniDreams replay and WebRTC demo application."""

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        return parse_args(argv)

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        return _replay_spec(args)

    def replay_adapter(self) -> OmnidreamsDemoAdapter:
        return OmnidreamsDemoAdapter()

    def serve_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        from .webrtc import serve_omnidreams_webrtc_demo

        serve_omnidreams_webrtc_demo(
            spec=_webrtc_spec(args, device=str(context.device)),
            world_rank=context.world_rank,
        )


_APPLICATION = OmnidreamsDemoApplication()


def main(argv: list[str] | None = None) -> None:
    """Run the OmniDreams demo application."""
    _APPLICATION.main(argv)


def _replay_spec(args: argparse.Namespace) -> DemoSpec:
    scenario: dict[str, object] = {
        "conditioning_mode": args.conditioning_mode,
        "example_data": args.example_data,
        "example_data_uuid": args.example_data_uuid,
        "total_blocks": args.total_blocks,
        "pixel_height": args.pixel_height,
        "pixel_width": args.pixel_width,
        "fps": args.fps,
    }
    if args.prompt:
        scenario["prompt"] = args.prompt
    if args.conditioning_mode == OMNIDREAMS_CONDITIONING_LUDUS:
        scenario.update(
            {
                "keyboard_trace_path": args.keyboard_trace,
                "scene_path": args.scene_path,
                "scene_dir": args.scene_dir,
                "scene_uuid": args.scene_uuid,
                "scene_variant": args.scene_variant,
                "camera_name": args.camera_name,
                "move_speed_per_s": args.move_speed_per_s,
                "rotate_speed_rad_per_s": args.rotate_speed_rad_per_s,
                "ludus_backend": args.ludus_backend,
            }
        )
    else:
        if args.hdmap_video_paths:
            scenario["hdmap_video_paths"] = args.hdmap_video_paths
        if args.first_frame_paths:
            scenario["first_frame_paths"] = args.first_frame_paths
        if args.camera_names:
            scenario["camera_names"] = args.camera_names

    return DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=args.preset_id,
        input_mode="replay",
        scenario=scenario,
        output=_replay_output_spec(args),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=args.preset_id,
            device=args.device,
            seed=args.seed,
            runtime_options={"seed": args.seed},
        ),
    )


def _replay_output_spec(args: argparse.Namespace) -> Mp4OutputSpec | NullOutputSpec:
    if args.output_mode == "mp4":
        if args.output is None:
            raise ValueError("OmniDreams MP4 replay requires --output.")
        return Mp4OutputSpec(path=args.output, fps=args.fps)
    if args.output_mode == "null":
        return NullOutputSpec()
    raise ValueError(
        f"Unsupported OmniDreams replay output mode: {args.output_mode!r}."
    )


def _webrtc_spec(args: argparse.Namespace, *, device: str) -> DemoSpec:
    return DemoSpec(
        model_id=OMNIDREAMS_MODEL_ID,
        preset_id=args.preset_id,
        input_mode="keyboard-driving",
        scenario=OmnidreamsWebRTCScenario(
            scene_dir=args.scene_dir,
            scene_uuid=args.scene_uuid,
            scene_variant=args.scene_variant,
            camera_name=args.camera_name,
            debug_serve_hdmaps=args.debug_serve_hdmaps,
            prefer_sw_encoder=args.prefer_sw_encoder,
        ),
        output=WebRTCOutputSpec(
            host=args.host,
            port=args.port,
            fps=args.fps,
            video_width=args.video_width,
            video_height=args.video_height,
            warmup_chunks=args.warmup_chunks,
            warmup_timeout_s=args.warmup_timeout_s,
            client_liveness_timeout_s=args.client_liveness_timeout_s,
            preload_name="Omnidreams",
        ),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=args.preset_id,
            device=device,
            runtime_options={"seed": args.seed},
        ),
    )


def _split_paths(value: str) -> tuple[Path, ...]:
    return tuple(Path(part) for part in value.split(",") if part)


def _split_strings(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(",") if part)


if __name__ == "__main__":
    main()
