# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP upload routes for FlashVSR WebRTC sessions."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote

from aiohttp import web
from aiohttp.multipart import BodyPartReader

from flashdreams.runtime.demo import DemoSpec
from flashdreams.serving.webrtc.server import SessionBusyError

from .adapter import FlashVSRDemoAdapter
from .spec import PreparedFlashVSRVideo

MAX_UPLOAD_VIDEO_BYTES = 512 * 1024 * 1024
"""Maximum accepted MP4 upload size."""

_UPLOAD_ROUTE = "/api/session/input"
_OFFER_ROUTE = "/api/webrtc/offer"
_ACCEPTED_VIDEO_CONTENT_TYPES = frozenset(
    {
        "application/mp4",
        "application/octet-stream",
        "video/mp4",
    }
)


class FlashVSRSessionManager(Protocol):
    """Manager surface needed by the upload controller."""

    @property
    def pending_session_input(self) -> Any: ...

    def has_active_session(self) -> bool: ...

    def set_pending_session_input(self, session_input: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class FlashVSRWebRTCSessionInput:
    """Decoded input staged for one WebRTC offer."""

    prepared_video: PreparedFlashVSRVideo
    """CPU video consumed by the native model input provider."""

    original_name: str
    """Sanitized browser filename used for display metadata."""


class FlashVSRUploadController:
    """Validate, decode, and stage uploaded videos for the next session."""

    def __init__(
        self,
        *,
        manager: FlashVSRSessionManager,
        adapter: FlashVSRDemoAdapter,
        spec: DemoSpec,
        default_video: PreparedFlashVSRVideo | None,
    ) -> None:
        self.manager = manager
        self.adapter = adapter
        self.spec = spec
        self.default_video = default_video

    def has_session_input(self) -> bool:
        """Return whether an upload or server-side fallback is available."""
        return (
            isinstance(
                self.manager.pending_session_input,
                FlashVSRWebRTCSessionInput,
            )
            or self.default_video is not None
        )

    def status_payload(self) -> dict[str, Any]:
        """Return browser-facing input availability and video metadata."""
        pending = self.manager.pending_session_input
        if isinstance(pending, FlashVSRWebRTCSessionInput):
            return {
                "upload_required": False,
                "has_default_input": self.default_video is not None,
                "input_source": "uploaded",
                "filename": pending.original_name,
                **_video_metadata(pending.prepared_video),
            }
        payload: dict[str, Any] = {
            "upload_required": self.default_video is None,
            "has_default_input": self.default_video is not None,
            "input_source": "server" if self.default_video is not None else None,
        }
        if self.default_video is not None:
            payload.update(_video_metadata(self.default_video))
        return payload

    def stage_uploaded_video(
        self,
        *,
        upload_path: Path,
        original_name: str,
    ) -> dict[str, Any]:
        """Decode an uploaded MP4 and stage it for the next WebRTC offer."""
        if self.manager.has_active_session():
            raise SessionBusyError("A FlashVSR session is already active.")
        prepared = self.adapter.prepare_uploaded_video(
            self.spec,
            upload_path=upload_path,
            original_name=original_name,
        )
        session_input = FlashVSRWebRTCSessionInput(
            prepared_video=prepared,
            original_name=original_name,
        )
        self.manager.set_pending_session_input(session_input)
        return {
            "upload_required": False,
            "has_default_input": self.default_video is not None,
            "input_source": "uploaded",
            "filename": original_name,
            **_video_metadata(prepared),
        }


FLASHVSR_UPLOAD_CONTROLLER_KEY = web.AppKey(
    "flashvsr_upload_controller",
    FlashVSRUploadController,
)


def configure_flashvsr_webrtc_app(
    app: web.Application,
    *,
    controller: FlashVSRUploadController,
) -> None:
    """Register FlashVSR upload routes and offer validation."""
    app[FLASHVSR_UPLOAD_CONTROLLER_KEY] = controller
    app.router.add_get(_UPLOAD_ROUTE, _session_input_status)
    app.router.add_post(_UPLOAD_ROUTE, _session_input_upload)
    app.middlewares.append(_require_session_input)


@web.middleware
async def _require_session_input(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    controller = request.app[FLASHVSR_UPLOAD_CONTROLLER_KEY]
    if (
        request.path == _OFFER_ROUTE
        and not controller.manager.has_active_session()
        and not controller.has_session_input()
    ):
        raise web.HTTPBadRequest(
            reason="Upload an MP4 before connecting the FlashVSR session."
        )
    return await handler(request)


async def _session_input_status(request: web.Request) -> web.StreamResponse:
    controller = request.app[FLASHVSR_UPLOAD_CONTROLLER_KEY]
    return web.json_response(controller.status_payload())


async def _session_input_upload(request: web.Request) -> web.StreamResponse:
    if not request.content_type.startswith("multipart/"):
        raise web.HTTPBadRequest(reason="Expected a multipart MP4 upload.")
    try:
        reader = await request.multipart()
    except Exception as exc:
        raise web.HTTPBadRequest(reason="Expected a multipart MP4 upload.") from exc

    video_field: BodyPartReader | None = None
    while True:
        field = await reader.next()
        if field is None:
            break
        if not isinstance(field, BodyPartReader):
            continue
        if field.name != "video":
            await field.release()
            continue
        if video_field is not None:
            raise web.HTTPBadRequest(reason="Upload exactly one MP4 video.")
        video_field = field
        break

    if video_field is None or not video_field.filename:
        raise web.HTTPBadRequest(reason="Upload an MP4 in the 'video' field.")
    original_name = _sanitize_filename(video_field.filename)
    if Path(original_name).suffix.lower() != ".mp4":
        raise web.HTTPBadRequest(reason="Uploaded video must use the .mp4 extension.")
    content_type = (
        video_field.headers.get(
            "Content-Type",
            "application/octet-stream",
        )
        .partition(";")[0]
        .strip()
        .lower()
    )
    if content_type not in _ACCEPTED_VIDEO_CONTENT_TYPES:
        raise web.HTTPBadRequest(reason="Uploaded video must be an MP4.")

    controller = request.app[FLASHVSR_UPLOAD_CONTROLLER_KEY]
    if controller.manager.has_active_session():
        raise web.HTTPConflict(reason="A FlashVSR session is already active.")

    with tempfile.TemporaryDirectory(prefix="flashvsr-upload-") as temp_dir:
        upload_path = Path(temp_dir) / "input.mp4"
        await _stream_uploaded_video(video_field, upload_path)
        try:
            payload = await asyncio.to_thread(
                controller.stage_uploaded_video,
                upload_path=upload_path,
                original_name=original_name,
            )
        except SessionBusyError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Invalid uploaded MP4: {exc}") from exc
    return web.json_response(payload)


async def _stream_uploaded_video(
    field: BodyPartReader,
    upload_path: Path,
) -> None:
    total_bytes = 0
    with upload_path.open("wb") as stream:
        while True:
            chunk = await field.read_chunk(size=1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_VIDEO_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_UPLOAD_VIDEO_BYTES,
                    actual_size=total_bytes,
                )
            stream.write(chunk)
    if total_bytes == 0:
        raise web.HTTPBadRequest(reason="Uploaded MP4 is empty.")


def _sanitize_filename(value: str) -> str:
    filename = Path(unquote(value).replace("\\", "/")).name.strip()
    return filename or "upload.mp4"


def _video_metadata(prepared: PreparedFlashVSRVideo) -> dict[str, Any]:
    return {
        "fps": prepared.fps,
        "num_frames": prepared.total_frames,
        "input_resolution": {
            "width": prepared.input_width,
            "height": prepared.input_height,
        },
        "resolution": {
            "width": prepared.target_width,
            "height": prepared.target_height,
        },
    }


__all__ = [
    "FLASHVSR_UPLOAD_CONTROLLER_KEY",
    "MAX_UPLOAD_VIDEO_BYTES",
    "FlashVSRUploadController",
    "FlashVSRWebRTCSessionInput",
    "configure_flashvsr_webrtc_app",
]
