# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for native FlashVSR runtime API demos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from loguru import logger

from flashdreams.infra.runner_io import read_video_fps, resolve_input_path
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    WebRTCOutputSpec,
)
from flashdreams.runtime.demo.app import DemoApplication
from flashvsr.runtime import DEFAULT_FLASHVSR_PRESET, FLASHVSR_MODEL_ID

from .adapter import FlashVSRDemoAdapter
from .spec import (
    DEFAULT_FLASHVSR_INPUT_URL,
    FLASHVSR_INPUT_CACHE_DIR,
    FlashVSRVideoScenario,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse replay or WebRTC demo arguments."""
    parser = argparse.ArgumentParser(
        description="FlashVSR demos using the native flashdreams.runtime API."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Upscale a finite MP4 input.")
    _add_model_arguments(
        replay,
        default_device="cuda",
        default_input=DEFAULT_FLASHVSR_INPUT_URL,
    )
    replay.add_argument("--output-mode", choices=("mp4", "null"), default="mp4")
    replay.add_argument("--output", type=Path, default=None)

    webrtc = subparsers.add_parser(
        "webrtc",
        help="Upload an MP4 in the browser and stream the upscaled result.",
    )
    _add_model_arguments(
        webrtc,
        default_device="cuda:0",
        default_input=None,
    )
    webrtc.add_argument("--host", default="0.0.0.0")
    webrtc.add_argument("--port", type=int, default=8082)
    webrtc.add_argument(
        "--loop-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop the selected source until the browser disconnects (default: true).",
    )
    webrtc.add_argument("--warmup-chunks", type=int, default=1)
    webrtc.add_argument("--warmup-timeout-s", type=float, default=600.0)
    webrtc.add_argument("--client-liveness-timeout-s", type=float, default=30.0)
    webrtc.add_argument(
        "--prefer-sw-encoder",
        action="store_true",
        help=(
            "Use aiortc's software encoder. Browser-upload mode already selects "
            "this resolution-agnostic backend."
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "replay":
        if args.output_mode == "mp4" and args.output is None:
            parser.error("replay --output is required when --output-mode=mp4.")
        if args.output_mode == "null" and args.output is not None:
            parser.error("replay --output is valid only when --output-mode=mp4.")
    return args


def _add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_device: str,
    default_input: str | None,
) -> None:
    parser.add_argument("--preset-id", default=DEFAULT_FLASHVSR_PRESET)
    parser.add_argument(
        "--input",
        "--input-path",
        dest="input_path",
        default=default_input,
        help=(
            "Optional server-side input video. WebRTC can instead upload an MP4 "
            "from the browser."
        ),
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--chunk-size", type=int, choices=(8, 16), default=16)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Override playback FPS. Defaults to source metadata, or 30 for "
            "upload-only WebRTC startup."
        ),
    )
    parser.add_argument("--scale", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--crop-region",
        choices=("none", "bottom_half", "top_half"),
        default="none",
    )
    parser.add_argument("--tail-policy", choices=("drop", "pad"), default="drop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the preset's torch.compile setting.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CUDA graphs for the encoder, DiT, and decoder.",
    )
    parser.add_argument(
        "--color-corrector",
        choices=("cuda", "torch"),
        default=None,
        help="Override the preset's color-correction implementation.",
    )


class FlashVSRDemoApplication(DemoApplication):
    """Dispatch FlashVSR finite replay and realtime WebRTC demos."""

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        return parse_args(argv)

    def replay_spec(self, args: argparse.Namespace) -> DemoSpec:
        return _replay_spec(args)

    def replay_adapter(self) -> FlashVSRDemoAdapter:
        return FlashVSRDemoAdapter()

    def serve_webrtc(self, args: argparse.Namespace, *, context: Any) -> None:
        from .webrtc import serve_flashvsr_webrtc_demo

        serve_flashvsr_webrtc_demo(
            spec=_webrtc_spec(args, device=str(context.device)),
            world_rank=context.world_rank,
        )


_APPLICATION = FlashVSRDemoApplication()


def main(argv: list[str] | None = None) -> None:
    """Run the FlashVSR demo application."""
    _APPLICATION.main(argv)


def _replay_spec(args: argparse.Namespace) -> DemoSpec:
    fps = _resolve_demo_fps(args, webrtc=False)
    return DemoSpec(
        model_id=FLASHVSR_MODEL_ID,
        preset_id=args.preset_id,
        input_mode="replay",
        scenario=_video_scenario(args, loop_input=False, fps=fps),
        output=_replay_output_spec(args, fps=fps),
        config=_inference_config(args, device=args.device, fps=fps),
    )


def _replay_output_spec(
    args: argparse.Namespace,
    *,
    fps: float,
) -> Mp4OutputSpec | NullOutputSpec:
    if args.output_mode == "null":
        return NullOutputSpec()
    if args.output is None:
        raise ValueError("FlashVSR MP4 replay requires --output.")
    return Mp4OutputSpec(
        path=args.output,
        fps=fps,
        output_layout="bcthw",
    )


def _webrtc_spec(args: argparse.Namespace, *, device: str) -> DemoSpec:
    fps = _resolve_demo_fps(args, webrtc=True)
    return DemoSpec(
        model_id=FLASHVSR_MODEL_ID,
        preset_id=args.preset_id,
        input_mode="replay",
        scenario=_video_scenario(args, loop_input=args.loop_input, fps=fps),
        # Upload-only startup has no resolution yet. The adapter replaces
        # these placeholders with the decoded target dimensions per session.
        output=WebRTCOutputSpec(
            host=args.host,
            port=args.port,
            fps=int(fps),
            video_width=128,
            video_height=128,
            warmup_chunks=args.warmup_chunks,
            warmup_timeout_s=args.warmup_timeout_s,
            client_liveness_timeout_s=args.client_liveness_timeout_s,
            preload_name="FlashVSR",
        ),
        config=_inference_config(
            args,
            device=device,
            fps=fps,
            extra_options={"prefer_sw_encoder": args.prefer_sw_encoder},
        ),
    )


def _video_scenario(
    args: argparse.Namespace,
    *,
    loop_input: bool,
    fps: float,
) -> FlashVSRVideoScenario:
    return FlashVSRVideoScenario(
        input_path=args.input_path,
        chunk_size=args.chunk_size,
        fps=fps,
        crop_region=args.crop_region,
        tail_policy=args.tail_policy,
        loop_input=loop_input,
    )


def _inference_config(
    args: argparse.Namespace,
    *,
    device: str,
    fps: float,
    extra_options: dict[str, Any] | None = None,
) -> InferenceConfig:
    runtime_options: dict[str, Any] = {
        "fps": fps,
        "chunk_size": args.chunk_size,
        "scale": args.scale,
    }
    if args.cuda_graph is not None:
        runtime_options["use_cuda_graph"] = args.cuda_graph
    if args.color_corrector is not None:
        runtime_options["color_corrector_implementation"] = args.color_corrector
    if extra_options:
        runtime_options.update(extra_options)
    return InferenceConfig(
        model_id=FLASHVSR_MODEL_ID,
        preset_id=args.preset_id,
        device=device,
        seed=args.seed,
        compile=args.compile,
        runtime_options=runtime_options,
    )


def _resolve_demo_fps(
    args: argparse.Namespace,
    *,
    webrtc: bool,
) -> float:
    fps = args.fps
    if fps is None and args.input_path is None:
        fps = 30.0
    elif fps is None:
        resolved_path = resolve_input_path(
            args.input_path,
            cache_dir=FLASHVSR_INPUT_CACHE_DIR,
        )
        try:
            fps = float(read_video_fps(resolved_path))
        except Exception:
            logger.warning("Could not read input fps; using 30 fps.")
            fps = 30.0
    fps = float(fps)
    if fps <= 0:
        raise ValueError("FlashVSR fps must be > 0.")
    if not webrtc:
        return fps
    transport_fps = int(round(fps))
    if transport_fps <= 0:
        raise ValueError("FlashVSR WebRTC fps must round to at least 1.")
    if not math.isclose(fps, transport_fps, rel_tol=0.0, abs_tol=1e-6):
        logger.warning(
            "WebRTC requires integer fps; using {} instead of {}.",
            transport_fps,
            fps,
        )
    return float(transport_fps)


if __name__ == "__main__":
    main()


__all__ = [
    "FlashVSRDemoApplication",
    "main",
    "parse_args",
]
