# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.webrtc.session import (
    OmnidreamsRuntimeConfig,
    OmnidreamsWebRTCSessionManager,
)

from flashdreams.core.distributed import (
    configure_loguru_for_distributed,
)
from flashdreams.core.distributed import (
    init as distributed_init,
)
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.server import create_webrtc_app

WEB_DIR = Path(__file__).resolve().parent / "web"


def configure_logging(*, world_rank: int | None = None) -> None:
    configure_loguru_for_distributed(world_rank=world_rank)
    for logger_name in ("aioice", "aioice.ice", "aiortc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Omnidreams WebRTC server: serves /request_session and streams "
            "single-view WSAD-controlled video chunks over one peer connection."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--pipeline_config_name",
        type=str,
        default="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        choices=sorted(OMNIDREAMS_CONFIGS),
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / "assets"
        / "example_data"
        / "omnidreams-webrtc"
        / "0d404ff7-2b66-498c-b047-1ed8cded60d4",
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
    return parser.parse_args()


def create_app(
    *,
    session_manager: OmnidreamsWebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or OmnidreamsWebRTCSessionManager()
    return create_webrtc_app(
        web_dir=WEB_DIR,
        session_manager=manager,
        preload_name="Omnidreams",
    )


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
) -> OmnidreamsRuntimeConfig:
    return OmnidreamsRuntimeConfig(
        pipeline_config_name=args.pipeline_config_name,
        scene_dir=args.scene_dir,
        seed=args.seed,
        device=device_override or args.device,
        video_height=args.video_height,
        video_width=args.video_width,
        fps=args.fps,
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


def _validate_single_view_config(config_name: str) -> None:
    pipeline_cfg = OMNIDREAMS_CONFIGS[config_name]
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
    _validate_single_view_config(args.pipeline_config_name)

    runtime_device, world_rank, _ = initialize_distributed(default_device=args.device)
    runtime_config = build_runtime_config(args, device_override=str(runtime_device))
    session_manager = OmnidreamsWebRTCSessionManager(runtime_config=runtime_config)
    if world_rank == 0:
        app = create_app(session_manager=session_manager)
        logger.info("Starting on external IP: {}", get_external_ip())
        try:
            web.run_app(app, host=args.host, port=args.port)
        finally:
            session_manager.send_exit_signal()
    else:
        try:
            session_manager.wait_for_termination()
        except KeyboardInterrupt:
            logger.warning("Worker rank interrupted, shutting down.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
