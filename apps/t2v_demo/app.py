# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2V-specific WebRTC session and browser routes."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from aiohttp import web

from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import WebRTCSessionConfig

from .backends import backend_metadata
from .runtime import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
)


class T2VWebRTCSessionManager(BaseWebRTCSessionManager[Any, WebRTCSessionConfig]):
    """Shared manager with a prompt update for the next browser session."""

    def update_prompt(self, prompt: str, duration_s: float) -> None:
        if not prompt.strip():
            raise ValueError("Prompt must be non-empty.")
        if not 0 < duration_s <= 60:
            raise ValueError("Duration must be greater than 0 and at most 60 seconds.")
        scenario = dict(self._shared_spec.scenario or {})
        scenario[FIELD_PROMPT] = prompt.strip()
        scenario[FIELD_TOTAL_BLOCKS] = self.runtime.blocks_for_duration(
            duration_s, fps=int(scenario[FIELD_FPS])
        )
        self._shared_spec = replace(self._shared_spec, scenario=scenario)
        self._shared_scenario = self._shared_adapter.prepare_scenario(self._shared_spec)


def _configure_app(
    app: web.Application,
    *,
    manager: T2VWebRTCSessionManager,
    backend: str,
) -> None:
    async def app_config(_: web.Request) -> web.StreamResponse:
        return web.json_response(
            {"backends": backend_metadata(), "selected_backend": backend}
        )

    async def update_prompt(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            raise web.HTTPBadRequest(reason="Expected a JSON prompt.")
        duration_s = payload.get("duration_s")
        if not isinstance(duration_s, int | float):
            raise web.HTTPBadRequest(reason="Expected numeric duration_s.")
        try:
            manager.update_prompt(payload["prompt"], float(duration_s))
        except (RuntimeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"status": "ok"})

    async def download(_: web.Request) -> web.StreamResponse:
        artifact = manager.runtime.latest_artifact
        if artifact is None:
            raise web.HTTPNotFound(reason="No completed generation is available yet.")
        video_path, scenario = artifact
        if not video_path.is_file():
            raise web.HTTPNotFound(reason="Generated MP4 is no longer available.")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(video_path, "video.mp4")
            archive.writestr(
                "prompt.json",
                json.dumps(
                    {
                        "prompt": scenario.prompt,
                        "total_blocks": scenario.total_blocks,
                        "fps": scenario.fps,
                        "width": scenario.pixel_width,
                        "height": scenario.pixel_height,
                    },
                    indent=2,
                ),
            )
        return web.Response(
            body=buffer.getvalue(),
            headers={
                "Content-Disposition": "attachment; filename=flashdreams-generation.zip"
            },
            content_type="application/zip",
        )

    async def playback(_: web.Request) -> web.StreamResponse:
        artifact = manager.runtime.latest_artifact
        if artifact is None or not artifact[0].is_file():
            raise web.HTTPNotFound(reason="No completed MP4 is available yet.")
        return web.FileResponse(artifact[0])

    app.router.add_get("/api/t2v/config", app_config)
    app.router.add_post("/api/t2v/prompt", update_prompt)
    app.router.add_get("/api/t2v/download", download)
    app.router.add_get("/api/t2v/playback", playback)


__all__ = ["T2VWebRTCSessionManager", "_configure_app"]
