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
import os
from contextlib import ExitStack
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
from flashdreams.serving.network import get_external_ip
from flashdreams.serving.webrtc.bootstrap import (
    configure_logging,
    run_webrtc_server,
)
from flashdreams.serving.webrtc.server import (
    SESSION_MANAGER_KEY,
    SessionBusyError,
    WebRTCSessionManager,
    create_webrtc_app,
)
from lingbot.runner import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    EXAMPLE_DATA_BASE_URL,
    EXAMPLE_DATA_DIR_LOCAL,
    ensure_example_data_downloaded,
    example_data_dirname,
)
from lingbot.webrtc.session import (
    LingbotImagePayload,
    LingbotRuntimeConfig,
    LingbotSessionInput,
    LingbotWebRTCSessionManager,
    normalize_prompt_text,
    normalize_text_events,
)

WEB_DIR_RESOURCE = files("lingbot.webrtc").joinpath("web")
MAX_UPLOAD_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PROMPT_CHARS = 2_000


class LingbotSessionManager(WebRTCSessionManager, Protocol):
    def get_initial_scene(self) -> dict[str, object]: ...
    def get_first_frame(self) -> LingbotImagePayload: ...
    def set_pending_session_input(self, session_input: LingbotSessionInput) -> None: ...


def _get_lingbot_manager(app: web.Application) -> LingbotSessionManager:
    return cast(LingbotSessionManager, app[SESSION_MANAGER_KEY])


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
        "--example-idx",
        "--example_idx",
        type=int,
        default=0,
        choices=EXAMPLE_DATA_AVAILABLE_IDXS,
        help="Example folder index under the LingBot example-data cache.",
    )
    return parser.parse_args()


async def _close_package_resources(app: web.Application) -> None:
    app["package_resource_stack"].close()


def create_app(
    *,
    request_session_url: str,
    session_manager: WebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or LingbotWebRTCSessionManager()
    resource_stack = ExitStack()
    try:
        web_dir = resource_stack.enter_context(as_file(WEB_DIR_RESOURCE))
        app = create_webrtc_app(
            web_dir=web_dir,
            session_manager=manager,
            preload_name="Lingbot",
            request_session_url=request_session_url,
        )
        app.router.add_get("/api/session/initial_scene", _initial_scene)
        app.router.add_get("/api/session/first_frame", _first_frame)
        app.router.add_post("/api/session/input", _session_input)
        app["package_resource_stack"] = resource_stack
        app.on_cleanup.append(_close_package_resources)
    except Exception:
        resource_stack.close()
        raise
    return app


async def _initial_scene(request: web.Request) -> web.StreamResponse:
    manager = _get_lingbot_manager(request.app)
    return web.json_response(manager.get_initial_scene())


async def _first_frame(request: web.Request) -> web.StreamResponse:
    manager = _get_lingbot_manager(request.app)
    payload = await asyncio.to_thread(manager.get_first_frame)
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

    manager = _get_lingbot_manager(request.app)
    session_input = LingbotSessionInput(
        prompt=prompt or None,
        first_frame_image_bytes=image_bytes,
        first_frame_image_url=image_url,
        first_frame_content_type=image_content_type,
        text_events=normalized_text_events,
    )
    try:
        await asyncio.to_thread(manager.set_pending_session_input, session_input)
    except SessionBusyError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc
    return web.json_response(manager.get_initial_scene())


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
    example_idx = getattr(args, "example_idx", 0)
    example_dirname = example_data_dirname(example_idx)
    example_dir = EXAMPLE_DATA_DIR_LOCAL / example_dirname
    if (
        example_idx == 0
        and not example_dir.exists()
        and (EXAMPLE_DATA_DIR_LOCAL / "image.jpg").exists()
    ):
        example_dir = EXAMPLE_DATA_DIR_LOCAL
    return LingbotRuntimeConfig(
        config_name=args.config_name,
        compile_network=not args.no_compile,
        context_parallel_size=context_parallel_size,
        device=device_override or args.device,
        warmup_chunks=args.warmup_chunks,
        warmup_timeout_s=args.warmup_timeout_s,
        video_height=args.video_height,
        video_width=args.video_width,
        example_data_dir=example_dir,
        default_image_url=f"{EXAMPLE_DATA_BASE_URL}/{example_dirname}/image.jpg",
        default_intrinsics_url=(
            f"{EXAMPLE_DATA_BASE_URL}/{example_dirname}/intrinsics.npy"
        ),
        default_poses_url=f"{EXAMPLE_DATA_BASE_URL}/{example_dirname}/poses.npy",
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
    session_manager = LingbotWebRTCSessionManager(
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


if __name__ == "__main__":
    main()
