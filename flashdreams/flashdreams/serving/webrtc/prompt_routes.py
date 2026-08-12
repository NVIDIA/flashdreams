# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic HTTP routes for prompt-driven WebRTC video generation."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aiohttp import web


def configure_prompt_generation_routes(
    app: web.Application,
    *,
    manager: Any,
    config: Mapping[str, object] | None = None,
) -> None:
    """Attach generic prompt, config, playback, and download endpoints.

    Args:
        app: WebRTC application receiving the routes.
        manager: Session manager exposing ``update_prompt`` and a runtime with
            ``latest_artifact``.
        config: Browser-safe model metadata returned by ``GET /api/config``;
            ``None`` returns an empty object.
    """
    config = config or {}

    async def get_config(_: web.Request) -> web.StreamResponse:
        return web.json_response(config)

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
        video_path, scenario = _latest_artifact(manager)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(video_path, "video.mp4")
            archive.writestr(
                "scenario.json",
                json.dumps(_scenario_metadata(scenario), indent=2, default=str),
            )
        return web.Response(
            body=buffer.getvalue(),
            headers={
                "Content-Disposition": "attachment; filename=flashdreams-generation.zip"
            },
            content_type="application/zip",
        )

    async def playback(_: web.Request) -> web.StreamResponse:
        video_path, _ = _latest_artifact(manager)
        return web.FileResponse(video_path)

    app.router.add_get("/api/config", get_config)
    app.router.add_post("/api/prompt", update_prompt)
    app.router.add_get("/api/download", download)
    app.router.add_get("/api/playback", playback)


def _latest_artifact(manager: Any) -> tuple[Path, object]:
    """Return the manager's completed video artifact or raise a matching HTTP error."""
    artifact = manager.runtime.latest_artifact
    if artifact is None:
        raise web.HTTPNotFound(reason="No completed generation is available yet.")
    video_path, scenario = artifact
    video_path = Path(video_path)
    if not video_path.is_file():
        raise web.HTTPNotFound(reason="Generated MP4 is no longer available.")
    return video_path, scenario


def _scenario_metadata(scenario: object) -> Mapping[str, object]:
    """Convert a scenario object into browser-downloadable metadata."""
    if is_dataclass(scenario) and not isinstance(scenario, type):
        return asdict(scenario)
    if isinstance(scenario, Mapping):
        return scenario
    return {"scenario": str(scenario)}


__all__ = ["configure_prompt_generation_routes"]
