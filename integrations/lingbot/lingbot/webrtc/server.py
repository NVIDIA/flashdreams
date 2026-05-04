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
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from aiohttp import web

from flashdreams.core.distributed import init as distributed_init
from lingbot.webrtc.session import (
    LingbotRuntimeConfig,
    LingbotWebRTCSessionManager,
    SessionBusyError,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
LOGGER = logging.getLogger(__name__)


def get_external_ip() -> str:
    """Get the external IP address of this machine.

    Uses a UDP socket trick to determine which interface would be used
    to reach an external address. No actual connection is made.

    Returns:
        The external IP address as a string, or "127.0.0.1" if detection fails.
    """
    try:
        # Create a UDP socket and "connect" to an external address
        # This doesn't send any data, but tells us which local IP would be used
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


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
        help="Lingbot config preset from LINGBOT_WORLD_CONFIGS.",
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
    return parser.parse_args()


def create_app(
    *,
    session_manager: LingbotWebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or LingbotWebRTCSessionManager()
    app = web.Application()
    app["session_manager"] = manager

    async def request_session_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "request_session.html")

    async def offer(request: web.Request) -> web.StreamResponse:
        try:
            payload = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(reason="Expected JSON offer payload.") from exc

        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="Offer payload must be a JSON object.")

        sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not isinstance(sdp, str) or not sdp:
            raise web.HTTPBadRequest(
                reason="Offer payload must include non-empty 'sdp'."
            )
        if not isinstance(offer_type, str) or not offer_type:
            raise web.HTTPBadRequest(
                reason="Offer payload must include non-empty 'type'."
            )

        manager = request.app["session_manager"]
        try:
            answer_payload = await manager.create_answer(
                offer_sdp=sdp,
                offer_type=offer_type,
            )
        except SessionBusyError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Failed to process WebRTC offer.")
            raise web.HTTPInternalServerError(reason=str(exc)) from exc

        return web.json_response(answer_payload)

    async def healthz(request: web.Request) -> web.StreamResponse:
        manager = request.app["session_manager"]
        return web.json_response(
            {
                "status": "ok",
                "runtime_ready": manager.is_runtime_ready(),
                "session_active": manager.has_active_session(),
            }
        )

    async def on_startup(app: web.Application) -> None:
        manager = app["session_manager"]
        LOGGER.info("Preloading Lingbot runtime on startup.")
        await manager.preload_runtime()
        LOGGER.info("Lingbot runtime preload complete.")

    async def on_shutdown(app: web.Application) -> None:
        manager = app["session_manager"]
        LOGGER.info("Shutting down Lingbot runtime.")
        await manager.shutdown()

    app.router.add_get("/request_session", request_session_page)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_get("/healthz", healthz)
    app.router.add_static("/static/", WEB_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
    context_parallel_size: int = 1,
) -> LingbotRuntimeConfig:
    return LingbotRuntimeConfig(
        config_name=args.config_name,
        compile_network=not args.no_compile,
        context_parallel_size=context_parallel_size,
        device=device_override or args.device,
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

    LOGGER.info(
        "Rank %s initialized Lingbot runtime with context_parallel_size %s",
        world_rank,
        world_size,
    )
    return torch_device, world_rank, world_size


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()

    runtime_device, world_rank, context_parallel_size = initialize_distributed(
        default_device=args.device
    )

    runtime_config = build_runtime_config(
        args,
        device_override=str(runtime_device),
        context_parallel_size=context_parallel_size,
    )
    session_manager = LingbotWebRTCSessionManager(runtime_config=runtime_config)
    if world_rank == 0:
        app = create_app(session_manager=session_manager)
        print(f"Starting on external IP: {get_external_ip()}")
        try:
            web.run_app(app, host=args.host, port=args.port)
        finally:
            session_manager.send_exit_signal()
    else:
        try:
            session_manager.wait_for_termination()
        except KeyboardInterrupt:
            LOGGER.warning("Worker rank interrupted, shutting down.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()
        LOGGER.info("[Rank %s] Destroying process group", world_rank)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
