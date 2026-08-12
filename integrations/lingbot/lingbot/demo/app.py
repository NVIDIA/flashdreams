# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the experimental shared Lingbot demo path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, cast

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.app import DemoApplication
from flashdreams.runtime.demo.replay import run_replay_demo
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from lingbot.example_data import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    ensure_example_data_downloaded,
)
from lingbot.runtime import (
    FIELD_CAMERA_INTRINSICS_PATH,
    FIELD_CAMERA_POSES_PATH,
    FIELD_FIRST_FRAME_PATH,
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
)

from .adapter import LingbotDemoAdapter
from .spec import (
    DEFAULT_FPS,
    DEFAULT_LINGBOT_PRESET,
    DEFAULT_PIXEL_HEIGHT,
    DEFAULT_PIXEL_WIDTH,
    LINGBOT_MODEL_ID,
    LingbotWebRTCScenario,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Lingbot demo using flashdreams.runtime.demo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Run a finite replay demo.")
    replay.add_argument("--preset-id", "--config-name", default=DEFAULT_LINGBOT_PRESET)
    replay.add_argument("--device", default="cuda")
    replay.add_argument("--prompt", default=None)
    replay.add_argument("--prompt-path", type=Path, default=None)
    replay.add_argument("--image-path", type=Path, default=None)
    replay.add_argument("--pose-path", type=Path, default=None)
    replay.add_argument(
        "--intrinsic-path",
        "--intrinsics-path",
        type=Path,
        default=None,
    )
    replay.add_argument(
        "--example-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the bundled Lingbot example when asset paths are omitted "
            "(default: auto)."
        ),
    )
    replay.add_argument(
        "--example-idx",
        "--example_idx",
        type=int,
        default=0,
        choices=EXAMPLE_DATA_AVAILABLE_IDXS,
    )
    replay.add_argument("--total-blocks", type=int, default=20)
    replay.add_argument("--pixel-height", type=int, default=DEFAULT_PIXEL_HEIGHT)
    replay.add_argument("--pixel-width", type=int, default=DEFAULT_PIXEL_WIDTH)
    replay.add_argument("--fps", type=int, default=DEFAULT_FPS)
    replay.add_argument("--output-mode", choices=("mp4", "null"), default="mp4")
    replay.add_argument("--output", type=Path, default=None)

    webrtc = subparsers.add_parser("webrtc", help="Serve a WebRTC driving demo.")
    webrtc.add_argument("--preset-id", "--config-name", default=DEFAULT_LINGBOT_PRESET)
    webrtc.add_argument("--host", default="0.0.0.0")
    webrtc.add_argument("--port", type=int, default=8080)
    webrtc.add_argument("--device", default="cuda:0")
    webrtc.add_argument("--seed", type=int, default=42)
    webrtc.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable torch.compile for the Lingbot transformer.",
    )
    webrtc.add_argument("--fps", type=int, default=DEFAULT_FPS)
    webrtc.add_argument("--video-height", type=int, default=DEFAULT_PIXEL_HEIGHT)
    webrtc.add_argument("--video-width", type=int, default=DEFAULT_PIXEL_WIDTH)
    webrtc.add_argument("--warmup-chunks", type=int, default=10)
    webrtc.add_argument("--warmup-timeout-s", type=float, default=600.0)
    webrtc.add_argument("--client-liveness-timeout-s", type=float, default=30.0)
    webrtc.add_argument("--prefer-sw-encoder", action="store_true")
    webrtc.add_argument(
        "--example-idx",
        "--example_idx",
        type=int,
        default=0,
        choices=EXAMPLE_DATA_AVAILABLE_IDXS,
    )
    args = parser.parse_args(argv)
    if args.command == "replay":
        if args.output_mode == "mp4" and args.output is None:
            parser.error("replay --output is required when --output-mode=mp4.")
        if args.output_mode == "null" and args.output is not None:
            parser.error("replay --output is only valid when --output-mode=mp4.")
    return args


class LingbotDemoApplication(DemoApplication):
    """Lingbot replay and WebRTC demo application."""

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        return parse_args(argv)

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        return _replay_spec(args)

    def replay_adapter(self) -> LingbotDemoAdapter:
        return LingbotDemoAdapter()

    def prepare_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        ensure_example_data_downloaded(
            is_rank_zero=(context.world_rank == 0),
            example_idx=args.example_idx,
        )

    def serve_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        from .webrtc import serve_lingbot_webrtc_demo

        serve_lingbot_webrtc_demo(
            spec=_webrtc_spec(
                args,
                device=str(context.device),
                context_parallel_size=context.world_size,
            ),
            world_rank=context.world_rank,
        )


_APPLICATION = LingbotDemoApplication()


def main(argv: list[str] | None = None) -> None:
    """Run the Lingbot demo application."""
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
    """Launch a LingBot demo directly from a resolved runner configuration."""
    configure_logging()
    preset_id = str(getattr(config.pipeline, "name", config.runner_name))
    if mode in {"mp4", "null"}:
        output_path = output.get("path") or output.get("output")
        if mode == "mp4" and output_path is None:
            raise ValueError("LingBot mp4 mode requires output.path.")
        if mode == "null" and output_path is not None:
            raise ValueError("LingBot null mode does not write output.path.")
        args = argparse.Namespace(
            preset_id=preset_id,
            device=str(config.device),
            prompt=scenario.get("prompt"),
            prompt_path=_optional_path(scenario.get("prompt_path")),
            image_path=_optional_path(scenario.get("image_path")),
            pose_path=_optional_path(scenario.get("pose_path")),
            intrinsic_path=_optional_path(scenario.get("intrinsic_path")),
            example_data=scenario.get("example_data"),
            example_idx=_as_int(
                scenario.get("example_idx", getattr(config, "example_idx", 0))
            ),
            total_blocks=_as_int(
                scenario.get("total_blocks", getattr(config, "total_blocks", 20))
            ),
            pixel_height=_as_int(
                scenario.get(
                    "pixel_height",
                    getattr(config, "pixel_height", DEFAULT_PIXEL_HEIGHT),
                )
            ),
            pixel_width=_as_int(
                scenario.get(
                    "pixel_width", getattr(config, "pixel_width", DEFAULT_PIXEL_WIDTH)
                )
            ),
            fps=_as_int(
                output.get(
                    "fps", scenario.get("fps", getattr(config, "fps", DEFAULT_FPS))
                )
            ),
            output_mode=mode,
            output=None if output_path is None else Path(cast(Any, output_path)),
        )
        spec = _replay_spec(args)
        return run_replay_demo(spec=spec, adapter=LingbotDemoAdapter())
    if mode != "webrtc":
        raise ValueError(f"Unsupported LingBot launch mode: {mode!r}.")

    context = initialize_cuda_distributed(default_device=str(config.device))
    example_idx = _as_int(
        scenario.get("example_idx", getattr(config, "example_idx", 0))
    )
    ensure_example_data_downloaded(
        is_rank_zero=(context.world_rank == 0),
        example_idx=example_idx,
    )
    args = argparse.Namespace(
        preset_id=preset_id,
        device=str(context.device),
        seed=_as_int(output.get("seed", 42)),
        compile=_runner_compile(config),
        fps=_as_int(output.get("fps", getattr(config, "fps", DEFAULT_FPS))),
        video_height=_as_int(
            output.get(
                "video_height", getattr(config, "pixel_height", DEFAULT_PIXEL_HEIGHT)
            )
        ),
        video_width=_as_int(
            output.get(
                "video_width", getattr(config, "pixel_width", DEFAULT_PIXEL_WIDTH)
            )
        ),
        warmup_chunks=_as_int(output.get("warmup_chunks", 10)),
        warmup_timeout_s=_as_float(output.get("warmup_timeout_s", 600.0)),
        client_liveness_timeout_s=_as_float(
            output.get("client_liveness_timeout_s", 30.0)
        ),
        prefer_sw_encoder=bool(output.get("prefer_sw_encoder", prefer_sw_encoder)),
        example_idx=example_idx,
        host=str(host or output.get("host", "0.0.0.0")),
        port=_as_int(port if port is not None else output.get("port", 8080)),
    )
    from .webrtc import serve_lingbot_webrtc_demo

    return serve_lingbot_webrtc_demo(
        spec=_webrtc_spec(
            args,
            device=str(context.device),
            context_parallel_size=context.world_size,
        ),
        world_rank=context.world_rank,
    )


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _runner_compile(config: RunnerConfig) -> bool:
    transformer = getattr(
        getattr(config.pipeline, "diffusion_model", None),
        "transformer",
        None,
    )
    return bool(getattr(transformer, "compile_network", True))


def _replay_spec(args: argparse.Namespace) -> DemoSpec:
    scenario: dict[str, object] = {
        "example_data": args.example_data,
        "example_idx": args.example_idx,
        FIELD_TOTAL_BLOCKS: args.total_blocks,
        FIELD_PIXEL_HEIGHT: args.pixel_height,
        FIELD_PIXEL_WIDTH: args.pixel_width,
        FIELD_FPS: args.fps,
    }
    if args.prompt:
        scenario[FIELD_PROMPT] = args.prompt
    if args.prompt_path is not None:
        scenario["prompt_path"] = args.prompt_path
    if args.image_path is not None:
        scenario[FIELD_FIRST_FRAME_PATH] = args.image_path
    if args.pose_path is not None:
        scenario[FIELD_CAMERA_POSES_PATH] = args.pose_path
    if args.intrinsic_path is not None:
        scenario[FIELD_CAMERA_INTRINSICS_PATH] = args.intrinsic_path

    return DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=args.preset_id,
        input_mode="replay",
        scenario=scenario,
        output=_replay_output_spec(args),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=args.preset_id,
            device=args.device,
        ),
    )


def _replay_output_spec(args: argparse.Namespace) -> Mp4OutputSpec | NullOutputSpec:
    if args.output_mode == "mp4":
        if args.output is None:
            raise ValueError("Lingbot MP4 replay requires --output.")
        return Mp4OutputSpec(
            path=args.output,
            fps=args.fps,
            output_layout="tchw",
        )
    if args.output_mode == "null":
        return NullOutputSpec()
    raise ValueError(f"Unsupported Lingbot replay output mode: {args.output_mode!r}.")


def _webrtc_spec(
    args: argparse.Namespace,
    *,
    device: str,
    context_parallel_size: int = 1,
) -> DemoSpec:
    return DemoSpec(
        model_id=LINGBOT_MODEL_ID,
        preset_id=args.preset_id,
        input_mode="keyboard-driving",
        scenario=LingbotWebRTCScenario(
            example_idx=args.example_idx,
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
            preload_name="Lingbot",
        ),
        config=InferenceConfig(
            model_id=LINGBOT_MODEL_ID,
            preset_id=args.preset_id,
            device=device,
            compile=args.compile,
            runtime_options={
                "seed": args.seed,
                "context_parallel_size": context_parallel_size,
                "example_idx": args.example_idx,
            },
        ),
    )
