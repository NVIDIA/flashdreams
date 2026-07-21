# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from dataclasses import replace
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.interactive_drive.cli_args import (
    ExplicitArgTrackingArgumentParser,
    arg_was_explicit,
)
from omnidreams.interactive_drive.config import WorldModelProfileConfig
from omnidreams.interactive_drive.world_model.flashdreams_adapter import (
    _build_pipeline_config,
)
from omnidreams.interactive_drive.world_model.manifest import (
    load_world_model_manifest,
    resolve_world_model_manifest_path,
)
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.webrtc.session import (
    OmnidreamsRuntimeConfig,
    OmnidreamsWebRTCSessionManager,
)

from flashdreams.core.distributed import (
    init as distributed_init,
)
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    run_webrtc_server,
)
from flashdreams.serving.webrtc.server import WebRTCSessionManager, create_webrtc_app

WEB_DIR_RESOURCE = files("omnidreams.webrtc").joinpath("web")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = ExplicitArgTrackingArgumentParser(
        description=(
            "Omnidreams WebRTC server: serves /request_session and streams "
            "single-view WSAD-controlled video chunks over one peer connection."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--pipeline_config_name",
        type=str,
        default="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        choices=sorted(OMNIDREAMS_CONFIGS),
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=None,
        help=(
            "Local WebRTC scene directory containing clipgt/first_image.* "
            "and clipgt/prompt.txt. If omitted, the server downloads and "
            "stages the selected Hugging Face scene."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Omnidreams world-model manifest (YAML). Accepts a path or a "
            "bundled config filename such as example_world_model_perf.yaml. "
            "When set, WebRTC uses the same pipeline perf toggles as the "
            "interactive-drive world-model path."
        ),
    )
    parser.add_argument(
        "--scene-uuid",
        type=str,
        default=None,
        help=(
            "Scene UUID for nvidia/omni-dreams-scenes. Expected dataset asset: "
            "scenes/clipgt-<uuid>[-<variant>].usdz."
        ),
    )
    parser.add_argument(
        "--scene-variant",
        type=str,
        default="default",
        help=(
            "Weather variant to serve: 'default' (clear), 'rain', or 'snow'. "
            "Selects the matching sibling archive and weather prompt."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video_height", type=int, default=704)
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument(
        "--warmup_chunks",
        type=int,
        default=10,
        help="Number of synthetic startup chunks to generate for kernel autotuning.",
    )
    parser.add_argument(
        "--warmup_timeout_s",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for synthetic startup warmup chunks.",
    )
    parser.add_argument(
        "--debug_serve_hdmaps",
        action="store_true",
        help=(
            "Stream rendered HDMap conditioning frames instead of generated RGB "
            "video. This skips video model generation after initialization."
        ),
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="camera_front_wide_120fov",
    )
    return parser.parse_args(argv)


async def _close_package_resources(app: web.Application) -> None:
    app["package_resource_stack"].close()


def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or OmnidreamsWebRTCSessionManager()

    resource_stack = ExitStack()
    try:
        web_dir = resource_stack.enter_context(as_file(WEB_DIR_RESOURCE))

        app = create_webrtc_app(
            web_dir=web_dir,
            session_manager=manager,
            preload_name="Omnidreams",
            request_session_url=request_session_url,
        )
        app["package_resource_stack"] = resource_stack
        app.on_cleanup.append(_close_package_resources)
    except Exception:
        resource_stack.close()
        raise
    return app


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
) -> OmnidreamsRuntimeConfig:
    manifest_path = None
    manifest = None
    pipeline_config = None
    pipeline_config_name = args.pipeline_config_name
    device = args.device
    seed = args.seed
    fps = args.fps
    video_width = args.video_width
    video_height = args.video_height

    manifest_arg = getattr(args, "manifest", None)
    if manifest_arg is not None:
        manifest_path = resolve_world_model_manifest_path(manifest_arg)
        manifest = load_world_model_manifest(manifest_path)
        pipeline_config = _build_pipeline_config(
            manifest,
            profile=WorldModelProfileConfig(),
        )
        pipeline_config_name = str(pipeline_config.name)
        if (
            arg_was_explicit(args, "pipeline_config_name")
            and args.pipeline_config_name != pipeline_config_name
        ):
            raise ValueError(
                "--manifest selects pipeline config "
                f"{pipeline_config_name!r}, but --pipeline_config_name was "
                f"also set to {args.pipeline_config_name!r}."
            )

        if not arg_was_explicit(args, "device"):
            device = manifest.device
        if not arg_was_explicit(args, "seed"):
            seed = manifest.seed_for_every_rollout
        if not arg_was_explicit(args, "fps"):
            fps = manifest.fps
        if not arg_was_explicit(args, "video_width"):
            video_width = manifest.resolution_wh[0]
        if not arg_was_explicit(args, "video_height"):
            video_height = manifest.resolution_wh[1]

    return OmnidreamsRuntimeConfig(
        pipeline_config_name=pipeline_config_name,
        pipeline_config=pipeline_config,
        manifest_path=manifest_path,
        scene_dir=args.scene_dir,
        scene_uuid=args.scene_uuid,
        scene_variant=args.scene_variant,
        seed=seed,
        device=device_override or device,
        video_height=video_height,
        video_width=video_width,
        fps=fps,
        camera_name=args.camera_name,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
        debug_serve_hdmaps=args.debug_serve_hdmaps,
    )


def initialize_distributed(
    *,
    default_device: str | torch.device = "cuda:0",
) -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for inference in the Omnidreams WebRTC server."
        )

    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if has_rank != has_world_size:
        raise RuntimeError(
            "Distributed launch expects both RANK and WORLD_SIZE to be set."
        )

    distributed_launch = has_rank and has_world_size
    if distributed_launch:
        distributed_init()
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_rank = 0
        world_size = 1

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA device count must be >= 1 for inference.")
    if distributed_launch:
        local_rank = world_rank % device_count
        torch_device = torch.device(f"cuda:{local_rank}")
    else:
        torch_device = torch.device(default_device)
        if torch_device.type != "cuda":
            raise RuntimeError(
                f"CUDA device is required for inference, got {torch_device}."
            )
        if torch_device.index is None:
            torch_device = torch.device("cuda:0")
    torch.cuda.set_device(torch_device)

    configure_logging(world_rank=world_rank)
    logger.info(
        "Rank {} initialized Omnidreams runtime with context_parallel_size {}",
        world_rank,
        world_size,
    )
    return torch_device, world_rank, world_size


def _validate_single_view_config(
    config_name: str, pipeline_config: Any | None = None
) -> None:
    pipeline_cfg = pipeline_config or OMNIDREAMS_CONFIGS[config_name]
    transformer_cfg = pipeline_cfg.diffusion_model.transformer
    if not isinstance(transformer_cfg, CosmosTransformerConfig):
        raise TypeError("Omnidreams WebRTC requires a CosmosTransformerConfig.")
    if transformer_cfg.num_views != 1:
        raise ValueError(
            "Omnidreams WebRTC only serves single-view configs; "
            f"{config_name!r} has num_views={transformer_cfg.num_views}."
        )


def main() -> None:
    configure_logging()
    args = parse_args()
    runtime_config = build_runtime_config(args)
    _validate_single_view_config(
        runtime_config.pipeline_config_name,
        runtime_config.pipeline_config,
    )

    runtime_device, world_rank, _ = initialize_distributed(
        default_device=runtime_config.device
    )
    runtime_config = replace(runtime_config, device=str(runtime_device))
    session_manager = OmnidreamsWebRTCSessionManager(runtime_config=runtime_config)
    app = None
    if world_rank == 0:
        external_ip = get_external_ip()
        app = create_app(
            session_manager=session_manager,
            request_session_url=f"http://{external_ip}:{args.port}/request_session",
        )
        logger.info("Starting on external IP: {}", external_ip)
    run_webrtc_server(
        world_rank=world_rank,
        session_manager=session_manager,
        app=app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
