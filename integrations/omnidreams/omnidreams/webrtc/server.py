# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from importlib.resources import as_file, files
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.distributed as dist
from aiohttp import web
from loguru import logger
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.transformer import CosmosTransformerConfig
from omnidreams.webrtc.session import (
    OmnidreamsRuntimeConfig,
    OmnidreamsSessionInput,
    OmnidreamsWebRTCSessionManager,
)

from flashdreams.core.distributed import (
    init as distributed_init,
)
from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
    run_webrtc_server,
)
from flashdreams.serving.webrtc.server import (
    SESSION_MANAGER_KEY,
    SessionBusyError,
    WebRTCSessionManager,
    create_packaged_webrtc_app,
    create_webrtc_app,
)
from flashdreams.serving.webrtc.server import (
    close_package_resources as _close_package_resources,
)

WEB_DIR_RESOURCE = files("omnidreams.webrtc").joinpath("web")


class _OmnidreamsSessionManager(WebRTCSessionManager, Protocol):
    runtime_config: OmnidreamsRuntimeConfig

    def set_pending_session_input(
        self, session_input: OmnidreamsSessionInput
    ) -> None: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
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
    parser.add_argument(
        "--postprocess-preset",
        "--postprocess_preset",
        dest="postprocess_preset",
        default="",
        choices=sorted(discover_postprocess_presets()),
        help=(
            "Default video post-process preset for WebRTC sessions. The browser "
            "can override this selection before connecting."
        ),
    )
    return parser.parse_args()


def _get_omnidreams_manager(app: web.Application) -> _OmnidreamsSessionManager:
    return cast(_OmnidreamsSessionManager, app[SESSION_MANAGER_KEY])


async def _postprocess_options(request: web.Request) -> web.StreamResponse:
    manager = _get_omnidreams_manager(request.app)
    presets = sorted(discover_postprocess_presets())
    return web.json_response(
        {
            "default_preset": manager.runtime_config.postprocess.preset,
            "presets": presets,
        }
    )


async def _session_input(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(reason="Expected JSON session input.") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(reason="Session input must be a JSON object.")
    preset = payload.get("postprocess_preset")
    if not isinstance(preset, str):
        raise web.HTTPBadRequest(
            reason="Session input must include string 'postprocess_preset'."
        )

    manager = _get_omnidreams_manager(request.app)
    try:
        manager.set_pending_session_input(
            OmnidreamsSessionInput(postprocess_preset=preset)
        )
    except SessionBusyError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc
    return web.json_response({"postprocess_preset": preset})


def _configure_app(app: web.Application) -> None:
    app.router.add_get("/api/postprocess/options", _postprocess_options)
    app.router.add_post("/api/session/input", _session_input)


def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or OmnidreamsWebRTCSessionManager()
    return create_packaged_webrtc_app(
        web_resource=WEB_DIR_RESOURCE,
        session_manager=manager,
        preload_name="Omnidreams",
        request_session_url=request_session_url,
        configure_app=_configure_app,
        as_file_fn=as_file,
        create_app_fn=create_webrtc_app,
        cleanup_callback=_close_package_resources,
    )


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
) -> OmnidreamsRuntimeConfig:
    return OmnidreamsRuntimeConfig(
        pipeline_config_name=args.pipeline_config_name,
        scene_dir=args.scene_dir,
        scene_uuid=args.scene_uuid,
        scene_variant=args.scene_variant,
        seed=args.seed,
        device=device_override or args.device,
        video_height=args.video_height,
        video_width=args.video_width,
        fps=args.fps,
        camera_name=args.camera_name,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
        debug_serve_hdmaps=args.debug_serve_hdmaps,
        postprocess=VideoPostprocessChainConfig(preset=args.postprocess_preset),
    )


def initialize_distributed(
    *,
    default_device: str | torch.device = "cuda:0",
) -> tuple[torch.device, int, int]:
    context = initialize_cuda_distributed(
        default_device=default_device,
        distributed_init_fn=distributed_init,
        configure_logging_fn=configure_logging,
        torch_module=torch,
        dist_module=dist,
    )
    logger.info(
        "Rank {} initialized Omnidreams runtime with context_parallel_size {}",
        context.world_rank,
        context.world_size,
    )
    return context.device, context.world_rank, context.world_size


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
