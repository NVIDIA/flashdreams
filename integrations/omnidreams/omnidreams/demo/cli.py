# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the experimental shared OmniDreams demo path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from omnidreams.runner import DEFAULT_EXAMPLE_DATA_UUID_1V

from flashdreams.core.distributed import init as distributed_init
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    WebRTCOutputSpec,
    run_flashdreams_demo,
    serve_flashdreams_demo,
)
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)

from .adapter import OmnidreamsDemoAdapter
from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_MODEL_ID,
    OmnidreamsWebRTCScenario,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental OmniDreams demo using flashdreams.runtime.demo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Run an MP4 replay demo.")
    replay.add_argument("--preset-id", default=DEFAULT_OMNIDREAMS_PRESET)
    replay.add_argument("--device", default="cuda")
    replay.add_argument("--prompt", default=None)
    replay.add_argument("--hdmap-video-paths", type=_split_paths, default=())
    replay.add_argument("--first-frame-paths", type=_split_paths, default=())
    replay.add_argument("--camera-names", type=_split_strings, default=())
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
    replay.add_argument("--output", type=Path, required=True)

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
    webrtc.add_argument("--postprocess-preset", default="")
    webrtc.add_argument("--prefer-sw-encoder", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    adapter = OmnidreamsDemoAdapter()
    if args.command == "replay":
        run_flashdreams_demo(spec=_replay_spec(args), adapter=adapter)
        return
    if args.command == "webrtc":
        context = initialize_cuda_distributed(
            default_device=args.device,
            distributed_init_fn=distributed_init,
            configure_logging_fn=configure_logging,
            torch_module=torch,
            dist_module=dist,
        )
        serve_flashdreams_demo(
            spec=_webrtc_spec(args, device=str(context.device)),
            adapter=adapter,
            world_rank=context.world_rank,
        )
        return
    raise AssertionError(f"Unhandled command: {args.command}")


def _replay_spec(args: argparse.Namespace) -> DemoSpec:
    scenario: dict[str, object] = {
        "example_data": args.example_data,
        "example_data_uuid": args.example_data_uuid,
        "total_blocks": args.total_blocks,
        "pixel_height": args.pixel_height,
        "pixel_width": args.pixel_width,
        "fps": args.fps,
    }
    if args.prompt:
        scenario["prompt"] = args.prompt
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
        output=Mp4OutputSpec(path=args.output, fps=args.fps),
        config=InferenceConfig(
            model_id=OMNIDREAMS_MODEL_ID,
            preset_id=args.preset_id,
            device=args.device,
        ),
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
            postprocess_preset=args.postprocess_preset,
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
