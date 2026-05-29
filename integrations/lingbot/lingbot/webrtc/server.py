# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from flashdreams.core.distributed import (
    configure_loguru_for_distributed,
)
from flashdreams.core.distributed import (
    init as distributed_init,
)
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.server import WebRTCSessionManager, create_webrtc_app
from lingbot.webrtc.session import (
    REPO_ROOT,
    LingbotRuntimeConfig,
    LingbotWebRTCSessionManager,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
EXAMPLE_DATA_AVAILABLE_IDXS = (0, 1, 2, 5)


def configure_logging(*, world_rank: int | None = None) -> None:
    configure_loguru_for_distributed(world_rank=world_rank)
    for logger_name in ("aioice", "aioice.ice", "aiortc"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lingbot WebRTC server: serves /request_session and streams action-bound "
            "video chunks over a single peer connection."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--config_name",
        type=str,
        default="lingbot-world-fast",
        help="Lingbot config preset from PIPELINE_CONFIGS.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile when building the Lingbot pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device used for the Lingbot runtime.",
    )
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
        "--fps",
        type=int,
        default=16,
        help="Output video framerate for WebRTC playback.",
    )
    parser.add_argument(
        "--example-idx",
        "--example_idx",
        type=int,
        default=0,
        choices=EXAMPLE_DATA_AVAILABLE_IDXS,
        help="Example folder index under assets/example_data/lingbot_world (allowed: 0, 1, 2, 5).",
    )
    return parser.parse_args()


def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or LingbotWebRTCSessionManager()
    return create_webrtc_app(
        web_dir=WEB_DIR,
        session_manager=manager,
        preload_name="Lingbot",
        request_session_url=request_session_url,
    )


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
    context_parallel_size: int = 1,
) -> LingbotRuntimeConfig:
    example_idx = getattr(args, "example_idx", 0)
    assert example_idx in EXAMPLE_DATA_AVAILABLE_IDXS, (
        f"--example_idx must be one of {EXAMPLE_DATA_AVAILABLE_IDXS}."
    )
    example_dir = REPO_ROOT / "assets/example_data/lingbot_world" / f"{example_idx:02d}"
    return LingbotRuntimeConfig(
        config_name=args.config_name,
        compile_network=not args.no_compile,
        context_parallel_size=context_parallel_size,
        device=device_override or args.device,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
        example_data_dir=example_dir,
    )


def initialize_distributed(
    *, default_device: str | torch.device = "cuda:0"
) -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for inference in the Lingbot WebRTC server."
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
        "Rank {} initialized Lingbot runtime with context_parallel_size {}",
        world_rank,
        world_size,
    )
    return torch_device, world_rank, world_size


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    runtime_device, world_rank, context_parallel_size = initialize_distributed(
        default_device=args.device
    )

    runtime_config = build_runtime_config(
        args,
        device_override=str(runtime_device),
        context_parallel_size=context_parallel_size,
    )
    session_manager = LingbotWebRTCSessionManager(
        runtime_config=runtime_config,
        fps=args.fps,
    )
    if world_rank == 0:
        external_ip = get_external_ip()
        app = create_app(
            session_manager=session_manager,
            request_session_url=f"http://{external_ip}:{args.port}/request_session",
        )
        logger.info("Starting on external IP: {}", external_ip)
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
        logger.info("[Rank {}] Destroying process group", world_rank)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
