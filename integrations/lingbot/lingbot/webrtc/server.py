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
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
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
    EXAMPLE_DATA_DIR_LOCAL,
    ensure_example_data_downloaded,
    example_data_dirname,
)
from lingbot.webrtc.session import (
    LingbotImagePayload,
    LingbotRuntimeConfig,
    LingbotSessionInput,
    LingbotWebRTCSessionManager,
    TextEventSpec,
    normalize_prompt_text,
    normalize_text_events,
)

WEB_DIR_RESOURCE = files("lingbot.webrtc").joinpath("web")
PRESET_DIR_RESOURCE = files("lingbot.webrtc").joinpath("presets")
MAX_UPLOAD_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PROMPT_CHARS = 2_000
PRESET_FIRST_FRAME_FILENAME = "first_frame.png"
PRESET_PROMPT_FILENAME = "prompt.txt"
PRESET_TEXT_EVENTS_FILENAME = "event_texts.json"
PRESET_ASSET_FILENAMES = (
    PRESET_FIRST_FRAME_FILENAME,
    PRESET_PROMPT_FILENAME,
    PRESET_TEXT_EVENTS_FILENAME,
)
"""Required filenames in a WebRTC preset-assets directory."""

BUNDLED_PRESET_IDS = (
    "golden-hour-portrait",
    "moonlit-portal",
    "cozy-reading-room",
    "misty-dinosaur-valley",
)
"""Stable UI order for bundled WebRTC presets."""


@dataclass(frozen=True, slots=True)
class PresetAsset:
    """In-memory assets for one selectable WebRTC scene preset."""

    preset_id: str
    """Stable identifier submitted by the WebRTC client."""

    label: str
    """Short display label for the preset picker."""

    prompt: str
    """Base scene prompt."""

    first_frame: LingbotImagePayload
    """Initial scene image."""

    text_events: tuple[TextEventSpec, ...]
    """Text-driven events available for the scene."""

    def as_public_dict(self) -> dict[str, str]:
        """Return metadata needed to render a preset picker card."""
        return {
            "preset_id": self.preset_id,
            "label": self.label,
            "first_frame_url": f"/api/presets/{self.preset_id}/first_frame",
        }


PRESET_CATALOG_KEY = web.AppKey("lingbot_preset_catalog", dict[str, PresetAsset])
"""Application key for the immutable bundled-preset lookup."""


class LingbotSessionManager(WebRTCSessionManager, Protocol):
    def get_initial_scene(self) -> dict[str, object]: ...
    def get_first_frame(self) -> LingbotImagePayload: ...
    def set_pending_session_input(self, session_input: LingbotSessionInput) -> None: ...


class _PresetResource(Protocol):
    """Readable interface shared by package resources and local paths."""

    def joinpath(
        self,
        *descendants: str | os.PathLike[str],
    ) -> _PresetResource: ...

    def is_file(self) -> bool: ...
    def read_bytes(self) -> bytes: ...
    def read_text(
        self,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str: ...


def _get_lingbot_manager(app: web.Application) -> LingbotSessionManager:
    return cast(LingbotSessionManager, app[SESSION_MANAGER_KEY])


def _get_preset_catalog(app: web.Application) -> dict[str, PresetAsset]:
    return app[PRESET_CATALOG_KEY]


def _initial_scene_payload(
    app: web.Application,
    manager: LingbotSessionManager,
) -> dict[str, object]:
    payload = dict(manager.get_initial_scene())
    payload.setdefault("active_preset_id", None)
    payload["presets"] = [
        preset.as_public_dict() for preset in _get_preset_catalog(app).values()
    ]
    return payload


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
    parser.add_argument(
        "--preset-assets-dir",
        "--preset_assets_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing first_frame.png, prompt.txt, and "
            "event_texts.json. Overrides --example-idx and skips example downloads."
        ),
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
        app.router.add_get(
            "/api/presets/{preset_id}/first_frame",
            _preset_first_frame,
        )
        app.router.add_post("/api/session/input", _session_input)
        app[PRESET_CATALOG_KEY] = {
            preset.preset_id: preset for preset in load_bundled_presets()
        }
        app["package_resource_stack"] = resource_stack
        app.on_cleanup.append(_close_package_resources)
    except Exception:
        resource_stack.close()
        raise
    return app


async def _initial_scene(request: web.Request) -> web.StreamResponse:
    manager = _get_lingbot_manager(request.app)
    return web.json_response(_initial_scene_payload(request.app, manager))


async def _first_frame(request: web.Request) -> web.StreamResponse:
    manager = _get_lingbot_manager(request.app)
    payload = await asyncio.to_thread(manager.get_first_frame)
    if not isinstance(payload, LingbotImagePayload):
        raise web.HTTPInternalServerError(reason="Invalid Lingbot first-frame payload.")
    return web.Response(body=payload.data, content_type=payload.content_type)


async def _preset_first_frame(request: web.Request) -> web.StreamResponse:
    preset_id = request.match_info["preset_id"]
    preset = _get_preset_catalog(request.app).get(preset_id)
    if preset is None:
        raise web.HTTPNotFound(reason=f"Unknown Lingbot preset: {preset_id}")
    return web.Response(
        body=preset.first_frame.data,
        content_type=preset.first_frame.content_type,
    )


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
    preset_id: str | None = None
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
            if field.name == "preset_id":
                preset_id = (await field.text()).strip() or None
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
        preset_id_raw = form.get("preset_id")
        prompt_raw = form.get("prompt")
        image_url_raw = form.get("image_url")
        text_events_raw = form.get("text_events", form.get("events"))
        if isinstance(preset_id_raw, str):
            preset_id = preset_id_raw.strip() or None
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

    preset = None
    if preset_id is not None:
        preset = _get_preset_catalog(request.app).get(preset_id)
        if preset is None:
            raise web.HTTPBadRequest(reason=f"Unknown Lingbot preset: {preset_id}")
        if prompt is None:
            prompt = preset.prompt
        if image_bytes is None and image_url is None:
            image_bytes = preset.first_frame.data
            image_content_type = preset.first_frame.content_type
        if text_events is None:
            text_events = preset.text_events

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
        preset_id=preset.preset_id if preset is not None else None,
    )
    try:
        await asyncio.to_thread(manager.set_pending_session_input, session_input)
    except SessionBusyError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc
    return web.json_response(_initial_scene_payload(request.app, manager))


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
    preset_assets_dir = getattr(args, "preset_assets_dir", None)
    if preset_assets_dir is not None:
        example_dir, text_events = load_preset_assets(preset_assets_dir)
        first_frame_filename = PRESET_FIRST_FRAME_FILENAME
        default_image_url = None
        default_preset_id = _bundled_preset_id_for_path(example_dir)
    else:
        example_idx = getattr(args, "example_idx", 0)
        example_dir = EXAMPLE_DATA_DIR_LOCAL / example_data_dirname(example_idx)
        if (
            example_idx == 0
            and not example_dir.exists()
            and (EXAMPLE_DATA_DIR_LOCAL / "image.jpg").exists()
        ):
            example_dir = EXAMPLE_DATA_DIR_LOCAL
        default_runtime_config = LingbotRuntimeConfig()
        text_events = default_runtime_config.text_events
        first_frame_filename = "image.jpg"
        default_image_url = default_runtime_config.default_image_url
        default_preset_id = None
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
        first_frame_filename=first_frame_filename,
        default_image_url=default_image_url,
        default_preset_id=default_preset_id,
        text_events=text_events,
    )


def _read_preset_assets(
    preset_assets_dir: _PresetResource,
    *,
    source: str,
) -> tuple[bytes, str, tuple[TextEventSpec, ...]]:
    missing = [
        filename
        for filename in PRESET_ASSET_FILENAMES
        if not preset_assets_dir.joinpath(filename).is_file()
    ]
    if missing:
        raise ValueError(f"{source} is missing: {', '.join(missing)}")

    first_frame_path = preset_assets_dir.joinpath(PRESET_FIRST_FRAME_FILENAME)
    try:
        first_frame_bytes = first_frame_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Failed to read preset first frame from {first_frame_path}: {exc}"
        ) from exc
    if not first_frame_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Preset first frame is not a PNG file: {first_frame_path}")

    prompt_path = preset_assets_dir.joinpath(PRESET_PROMPT_FILENAME)
    try:
        prompt = normalize_prompt_text(prompt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Failed to read preset prompt from {prompt_path}: {exc}"
        ) from exc

    events_path = preset_assets_dir.joinpath(PRESET_TEXT_EVENTS_FILENAME)
    try:
        raw_events: object = json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to read preset text events from {events_path}: {exc}"
        ) from exc

    if isinstance(raw_events, dict):
        raw_events = raw_events.get("text_events", raw_events.get("events"))
        if raw_events is None:
            raise ValueError(
                f"Preset text events in {events_path} must be a list or contain "
                "an 'events' or 'text_events' list."
            )
    try:
        text_events = normalize_text_events(raw_events)
    except ValueError as exc:
        raise ValueError(f"Invalid preset text events in {events_path}: {exc}") from exc
    return first_frame_bytes, prompt, text_events


def load_preset_assets(
    preset_assets_dir: Path,
) -> tuple[Path, tuple[TextEventSpec, ...]]:
    """Validate a preset-assets directory and load its text-event catalog."""
    preset_assets_dir = preset_assets_dir.expanduser().resolve()
    if not preset_assets_dir.is_dir():
        raise ValueError(
            f"--preset-assets-dir must be an existing directory: {preset_assets_dir}"
        )
    _, _, text_events = _read_preset_assets(
        preset_assets_dir,
        source=f"Preset assets directory {preset_assets_dir}",
    )
    return preset_assets_dir, text_events


@lru_cache(maxsize=1)
def load_bundled_presets() -> tuple[PresetAsset, ...]:
    """Load bundled preset assets for the WebRTC picker."""
    presets = []
    for preset_id in BUNDLED_PRESET_IDS:
        preset_dir = cast(
            _PresetResource,
            PRESET_DIR_RESOURCE.joinpath(preset_id),
        )
        first_frame_bytes, prompt, text_events = _read_preset_assets(
            preset_dir,
            source=f"Bundled preset {preset_id}",
        )
        presets.append(
            PresetAsset(
                preset_id=preset_id,
                label=preset_id.replace("-", " ").title(),
                prompt=prompt,
                first_frame=LingbotImagePayload(
                    data=first_frame_bytes,
                    content_type="image/png",
                ),
                text_events=text_events,
            )
        )
    return tuple(presets)


def _bundled_preset_id_for_path(preset_assets_dir: Path) -> str | None:
    preset_id = preset_assets_dir.name
    if preset_id not in BUNDLED_PRESET_IDS:
        return None
    bundled_path = Path(__file__).resolve().parent / "presets" / preset_id
    return preset_id if preset_assets_dir == bundled_path.resolve() else None


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

    # Without a local preset, mirror the offline runner's example-data
    # preflight so the WebRTC entry point works on a fresh checkout.
    if getattr(args, "preset_assets_dir", None) is None:
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
