# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the experimental shared OmniDreams demo path."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from omnidreams.runner import DEFAULT_EXAMPLE_DATA_UUID_1V

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.app import DemoApplication
from flashdreams.runtime.demo.benchmark import run_benchmark_demo
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)

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


def launch_from_runner(
    *,
    config: RunnerConfig,
    mode: Literal["mp4", "null", "webrtc"],
    scenario: dict[str, object],
    output: dict[str, object],
    host: str | None = None,
    port: int | None = None,
    prefer_sw_encoder: bool = False,
) -> object:
    """Launch an OmniDreams demo directly from a resolved runner config."""
    configure_logging()
    preset_id = str(getattr(config.pipeline, "name", config.runner_name))
    seed = _runner_seed(config)
    if mode in {"mp4", "null"}:
        output_path = output.get("path") or output.get("output")
        if mode == "mp4" and output_path is None:
            raise ValueError("OmniDreams mp4 mode requires output.path.")
        args = argparse.Namespace(
            preset_id=preset_id,
            device=str(config.device),
            seed=seed,
            conditioning_mode=scenario.get(
                "conditioning_mode", OMNIDREAMS_CONDITIONING_PRECOMPUTED
            ),
            prompt=scenario.get("prompt"),
            hdmap_video_paths=_as_tuple(scenario.get("hdmap_video_paths", ())),
            first_frame_paths=_as_tuple(scenario.get("first_frame_paths", ())),
            camera_names=_as_tuple(scenario.get("camera_names", ())),
            keyboard_trace=_optional_path(scenario.get("keyboard_trace")),
            scene_path=_optional_path(scenario.get("scene_path")),
            scene_dir=_optional_path(scenario.get("scene_dir")),
            scene_uuid=scenario.get("scene_uuid", DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID),
            scene_variant=str(scenario.get("scene_variant", "default")),
            camera_name=str(scenario.get("camera_name", "camera_front_wide_120fov")),
            move_speed_per_s=_as_float(scenario.get("move_speed_per_s", 6.0)),
            rotate_speed_rad_per_s=_as_float(
                scenario.get("rotate_speed_rad_per_s", math.radians(35.0))
            ),
            ludus_backend=str(scenario.get("ludus_backend", "cuda")),
            example_data=scenario.get("example_data"),
            example_data_uuid=scenario.get(
                "example_data_uuid", DEFAULT_EXAMPLE_DATA_UUID_1V
            ),
            total_blocks=_as_int(
                scenario.get("total_blocks", getattr(config, "total_blocks", 60))
            ),
            pixel_height=_as_int(
                scenario.get("pixel_height", getattr(config, "pixel_height", 704))
            ),
            pixel_width=_as_int(
                scenario.get("pixel_width", getattr(config, "pixel_width", 1280))
            ),
            fps=_as_int(
                output.get(
                    "fps", scenario.get("fps", getattr(config, "output_fps", 30))
                )
            ),
            output_mode=mode,
            output=None if output_path is None else Path(cast(Any, output_path)),
        )
        spec = _replay_spec(args)
        stats_path = _optional_path(output.get("stats_path"))
        stats_dir = _optional_path(output.get("stats_dir"))
        if stats_path is not None or stats_dir is not None:
            return run_benchmark_demo(
                spec=spec,
                adapter=OmnidreamsDemoAdapter(),
                stats_path=stats_path,
                stats_dir=stats_dir,
                capture_output=True,
            )
        return run_replay_demo(spec=spec, adapter=OmnidreamsDemoAdapter())
    if mode != "webrtc":
        raise ValueError(f"Unsupported OmniDreams launch mode: {mode!r}.")

    context = initialize_cuda_distributed(default_device=str(config.device))
    args = argparse.Namespace(
        preset_id=preset_id,
        device=str(context.device),
        seed=seed,
        scene_dir=_optional_path(scenario.get("scene_dir")),
        scene_uuid=scenario.get("scene_uuid"),
        scene_variant=str(scenario.get("scene_variant", "default")),
        camera_name=str(scenario.get("camera_name", "camera_front_wide_120fov")),
        fps=_as_int(output.get("fps", getattr(config, "output_fps", 30))),
        video_height=_as_int(
            output.get("video_height", getattr(config, "pixel_height", 704))
        ),
        video_width=_as_int(
            output.get("video_width", getattr(config, "pixel_width", 1280))
        ),
        warmup_chunks=_as_int(output.get("warmup_chunks", 10)),
        warmup_timeout_s=_as_float(output.get("warmup_timeout_s", 600.0)),
        client_liveness_timeout_s=_as_float(
            output.get("client_liveness_timeout_s", 10.0)
        ),
        debug_serve_hdmaps=bool(output.get("debug_serve_hdmaps", False)),
        prefer_sw_encoder=bool(output.get("prefer_sw_encoder", prefer_sw_encoder)),
        host=str(host or output.get("host", "0.0.0.0")),
        port=_as_int(port if port is not None else output.get("port", 8082)),
    )
    from .webrtc import serve_omnidreams_webrtc_demo

    return serve_omnidreams_webrtc_demo(
        spec=_webrtc_spec(args, device=str(context.device)),
        world_rank=context.world_rank,
    )


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("Expected a sequence value in the launch manifest.")
    return tuple(value)


def _runner_seed(config: RunnerConfig) -> int:
    diffusion_model = getattr(config.pipeline, "diffusion_model", None)
    seed = getattr(diffusion_model, "seed", 42)
    return 42 if seed is None else int(seed)


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
