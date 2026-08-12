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

"""WebRTC server for interactive LingBot-World inference."""

from __future__ import annotations

import argparse
import asyncio
import json
from importlib.resources import as_file, files
from typing import Protocol, cast

import torch
import torch.distributed as dist
from aiohttp import web
from aiohttp.multipart import BodyPartReader
from loguru import logger

from flashdreams.core.distributed import (
    init as distributed_init,
)
from flashdreams.runtime import InferenceConfig
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
    run_webrtc_server,
)
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
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
from lingbot.example_data import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    ensure_example_data_downloaded,
)
from lingbot.runtime import (
    LINGBOT_MODEL_ID,
    LingbotModelAdapter,
    build_lingbot_webrtc_runtime_config,
)
from lingbot.webrtc.session import (
    LingbotImagePayload,
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
    LingbotSessionInput,
    LingbotWebRTCSessionController,
    create_lingbot_webrtc_session_manager,
    normalize_prompt_text,
    normalize_text_events,
)

WEB_DIR_RESOURCE = files("flashdreams.serving.webrtc").joinpath("web")
MODEL_WEB_DIR_RESOURCE = files("lingbot.webrtc").joinpath("web")
MAX_UPLOAD_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PROMPT_CHARS = 2_000


class LingbotSessionController(Protocol):
    def get_initial_scene(self) -> dict[str, object]: ...
    def get_first_frame(self) -> LingbotImagePayload: ...
    def set_pending_session_input(self, session_input: LingbotSessionInput) -> None: ...


LINGBOT_SESSION_CONTROLLER_KEY = web.AppKey(
    "lingbot_session_controller",
    LingbotSessionController,
)


def _get_lingbot_controller(app: web.Application) -> LingbotSessionController:
    return app[LINGBOT_SESSION_CONTROLLER_KEY]


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
        help="LingBot-World config preset from PIPELINE_CONFIGS.",
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
        "--seed",
        type=int,
        default=42,
        help="Base random seed for the Lingbot rollout.",
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
        "--video-height",
        "--video_height",
        type=int,
        default=464,
        help="Output video pixel height. Must be divisible by 16.",
    )
    parser.add_argument(
        "--video-width",
        "--video_width",
        type=int,
        default=832,
        help="Output video pixel width. Must be divisible by 16.",
    )
    parser.add_argument(
        "--prefer_sw_encoder",
        action="store_true",
        help=(
            "Force aiortc's default software encoder instead of probing NVENC. "
            "Without this flag, LingBot uses NVENC when the driver reports "
            "support at the target resolution and falls back to software "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--example-idx",
        "--example_idx",
        type=int,
        default=0,
        choices=EXAMPLE_DATA_AVAILABLE_IDXS,
        help="Example folder index under the LingBot example-data cache.",
    )
    return parser.parse_args()


def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager | None = None,
    session_controller: LingbotSessionController | None = None,
) -> web.Application:
    manager = session_manager or create_lingbot_webrtc_session_manager()
    if session_controller is None and not isinstance(manager, BaseWebRTCSessionManager):
        # Lightweight server tests may provide one object for both protocols.
        session_controller = cast(LingbotSessionController, manager)

    def configure_app(app: web.Application) -> None:
        configure_lingbot_webrtc_app(
            app,
            session_controller=session_controller,
        )

    return create_packaged_webrtc_app(
        web_resource=WEB_DIR_RESOURCE,
        model_web_resource=MODEL_WEB_DIR_RESOURCE,
        session_manager=manager,
        preload_name="Lingbot",
        request_session_url=request_session_url,
        configure_app=configure_app,
        as_file_fn=as_file,
        create_app_fn=create_webrtc_app,
        cleanup_callback=_close_package_resources,
    )


def configure_lingbot_webrtc_app(
    app: web.Application,
    *,
    session_controller: LingbotSessionController | None = None,
) -> None:
    """Register Lingbot-only initial-scene and session-input routes."""
    if session_controller is None:
        manager = app[SESSION_MANAGER_KEY]
        if not isinstance(manager, BaseWebRTCSessionManager):
            raise TypeError(
                "Lingbot routes require BaseWebRTCSessionManager or an "
                "explicit session_controller."
            )
        session_controller = LingbotWebRTCSessionController(
            cast(
                BaseWebRTCSessionManager[
                    LingbotInferenceRuntime,
                    LingbotRuntimeConfig,
                ],
                manager,
            )
        )
    app[LINGBOT_SESSION_CONTROLLER_KEY] = session_controller
    app.router.add_get("/api/session/initial_scene", _initial_scene)
    app.router.add_get("/api/session/first_frame", _first_frame)
    app.router.add_post("/api/session/input", _session_input)


async def _initial_scene(request: web.Request) -> web.StreamResponse:
    controller = _get_lingbot_controller(request.app)
    return web.json_response(controller.get_initial_scene())


async def _first_frame(request: web.Request) -> web.StreamResponse:
    controller = _get_lingbot_controller(request.app)
    payload = await asyncio.to_thread(controller.get_first_frame)
    if not isinstance(payload, LingbotImagePayload):
        raise web.HTTPInternalServerError(reason="Invalid Lingbot first-frame payload.")
    return web.Response(body=payload.data, content_type=payload.content_type)


async def _read_upload_bytes(field: BodyPartReader) -> bytes:
    data = bytearray()
    while True:
        chunk = await field.read_chunk(size=64 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_IMAGE_BYTES:
            raise web.HTTPRequestEntityTooLarge(
                max_size=MAX_UPLOAD_IMAGE_BYTES,
                actual_size=len(data),
            )
    return bytes(data)


async def _session_input(request: web.Request) -> web.StreamResponse:
    prompt: str | None = None
    image_bytes: bytes | None = None
    image_url: str | None = None
    image_content_type = "image/jpeg"
    text_events: object | None = None

    if request.content_type.startswith("multipart/"):
        try:
            reader = await request.multipart()
        except Exception as exc:
            raise web.HTTPBadRequest(
                reason="Expected multipart session input."
            ) from exc

        while True:
            field = await reader.next()
            if field is None:
                break
            if not isinstance(field, BodyPartReader):
                continue
            if field.name == "prompt":
                prompt = normalize_prompt_text(await field.text())
                if len(prompt) > MAX_PROMPT_CHARS:
                    raise web.HTTPBadRequest(
                        reason=f"Prompt must be <= {MAX_PROMPT_CHARS} characters."
                    )
                continue
            if field.name == "image_url":
                image_url = (await field.text()).strip() or None
                continue
            if field.name in {"text_events", "events"}:
                events_raw = (await field.text()).strip()
                if events_raw:
                    try:
                        text_events = json.loads(events_raw)
                    except json.JSONDecodeError as exc:
                        raise web.HTTPBadRequest(
                            reason="Text events must be valid JSON."
                        ) from exc
                continue
            if field.name == "image" and field.filename:
                image_content_type = field.headers.get(
                    "Content-Type", "application/octet-stream"
                )
                if not image_content_type.startswith("image/"):
                    raise web.HTTPBadRequest(
                        reason="Uploaded first frame must be an image."
                    )
                image_bytes = await _read_upload_bytes(field)
                if not image_bytes:
                    raise web.HTTPBadRequest(
                        reason="Uploaded first-frame image is empty."
                    )
    else:
        form = await request.post()
        prompt_raw = form.get("prompt")
        image_url_raw = form.get("image_url")
        text_events_raw = form.get("text_events", form.get("events"))
        if isinstance(prompt_raw, str):
            prompt = normalize_prompt_text(prompt_raw)
            if len(prompt) > MAX_PROMPT_CHARS:
                raise web.HTTPBadRequest(
                    reason=f"Prompt must be <= {MAX_PROMPT_CHARS} characters."
                )
        if isinstance(image_url_raw, str):
            image_url = image_url_raw.strip() or None
        if isinstance(text_events_raw, str) and text_events_raw.strip():
            try:
                text_events = json.loads(text_events_raw)
            except json.JSONDecodeError as exc:
                raise web.HTTPBadRequest(
                    reason="Text events must be valid JSON."
                ) from exc

    if image_bytes is not None:
        image_url = None

    try:
        normalized_text_events = (
            normalize_text_events(text_events) if text_events is not None else None
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc

    if (
        not prompt
        and image_bytes is None
        and image_url is None
        and normalized_text_events is None
    ):
        raise web.HTTPBadRequest(
            reason=(
                "Upload a prompt, an image file, an image URL, text events, "
                "or a combination."
            )
        )

    controller = _get_lingbot_controller(request.app)
    session_input = LingbotSessionInput(
        prompt=prompt or None,
        first_frame_image_bytes=image_bytes,
        first_frame_image_url=image_url,
        first_frame_content_type=image_content_type,
        text_events=normalized_text_events,
    )
    try:
        await asyncio.to_thread(controller.set_pending_session_input, session_input)
    except SessionBusyError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc
    return web.json_response(controller.get_initial_scene())


def build_runtime_config(
    args: argparse.Namespace,
    *,
    device_override: str | None = None,
    context_parallel_size: int = 1,
) -> LingbotRuntimeConfig:
    if args.video_height <= 0 or args.video_width <= 0:
        raise ValueError("--video-height and --video-width must be > 0")
    if args.video_height % 16 != 0 or args.video_width % 16 != 0:
        raise ValueError("--video-height and --video-width must be divisible by 16")

    inference_config = InferenceConfig(
        model_id=LINGBOT_MODEL_ID,
        preset_id=args.config_name,
        device=device_override or args.device,
        compile=not args.no_compile,
        runtime_options={"context_parallel_size": context_parallel_size},
    )
    adapter = LingbotModelAdapter()
    adapter.validate_config(inference_config)
    return build_lingbot_webrtc_runtime_config(
        preset_id=adapter.preset_id(inference_config),
        pipeline_config=adapter.pipeline_config(inference_config),
        device=inference_config.device or args.device,
        seed=int(getattr(args, "seed", 42)),
        compile_network=not args.no_compile,
        context_parallel_size=context_parallel_size,
        video_height=args.video_height,
        video_width=args.video_width,
        fps=args.fps,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
        example_idx=getattr(args, "example_idx", 0),
        prefer_sw_encoder=getattr(args, "prefer_sw_encoder", False),
        runtime_options=inference_config.runtime_options,
    )


def initialize_distributed(
    *, default_device: str | torch.device = "cuda:0"
) -> tuple[torch.device, int, int]:
    context = initialize_cuda_distributed(
        default_device=default_device,
        distributed_init_fn=distributed_init,
        configure_logging_fn=configure_logging,
        torch_module=torch,
        dist_module=dist,
    )
    logger.info(
        "Rank {} initialized Lingbot runtime with context_parallel_size {}",
        context.world_rank,
        context.world_size,
    )
    return context.device, context.world_rank, context.world_size


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    runtime_device, world_rank, context_parallel_size = initialize_distributed(
        default_device=args.device
    )

    # Pull the bundled example-data assets onto rank 0 (and barrier the
    # rest) before constructing the session manager: the manager's
    # initial-sync step checks the example_data_dir for the first frame
    # / intrinsics / poses / prompt files and raises FileNotFoundError
    # otherwise. Mirrors the offline runner's pre-flight behavior so the
    # WebRTC entry point is launchable on a fresh checkout with no
    # manual file staging.
    ensure_example_data_downloaded(
        is_rank_zero=(world_rank == 0),
        example_idx=args.example_idx,
    )

    runtime_config = build_runtime_config(
        args,
        device_override=str(runtime_device),
        context_parallel_size=context_parallel_size,
    )
    session_manager = create_lingbot_webrtc_session_manager(
        runtime_config=runtime_config,
        fps=args.fps,
    )
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
