# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams WebRTC hooks for the shared demo API."""

from __future__ import annotations

from typing import Protocol, cast

from aiohttp import web
from omnidreams.webrtc.postprocess import validate_requested_postprocess_preset
from omnidreams.webrtc.session import OmnidreamsRuntimeConfig, OmnidreamsSessionInput

from flashdreams.plugins.registry import resolve_postprocess_preset
from flashdreams.runtime.demo import DemoSpec, WebRTCAppResources
from flashdreams.serving.webrtc.server import (
    SESSION_MANAGER_KEY,
    SessionBusyError,
    WebRTCSessionManager,
)


class _OmnidreamsSessionManager(WebRTCSessionManager, Protocol):
    runtime_config: OmnidreamsRuntimeConfig

    def set_pending_session_input(
        self,
        session_input: OmnidreamsSessionInput,
    ) -> None: ...


async def postprocess_options(request: web.Request) -> web.StreamResponse:
    """Return the postprocess preset selected at server launch."""
    manager = _get_omnidreams_manager(request.app)
    configured_preset = manager.runtime_config.postprocess.preset
    presets = [configured_preset] if configured_preset else []
    return web.json_response(
        {
            "default_preset": configured_preset,
            "presets": presets,
        }
    )


async def session_input(request: web.Request) -> web.StreamResponse:
    """Apply browser-selected settings to the next WebRTC rollout."""
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
        if preset:
            validate_requested_postprocess_preset(
                requested_preset=preset,
                configured_preset=manager.runtime_config.postprocess.preset,
            )
        manager.set_pending_session_input(
            OmnidreamsSessionInput(postprocess_preset=preset)
        )
    except SessionBusyError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=str(exc)) from exc
    return web.json_response({"postprocess_preset": preset})


def configure_omnidreams_webrtc_app(app: web.Application) -> None:
    """Register OmniDreams browser support routes on a shared WebRTC app."""
    app.router.add_get("/api/postprocess/options", postprocess_options)
    app.router.add_post("/api/session/input", session_input)


def omnidreams_webrtc_app_resources(spec: DemoSpec) -> WebRTCAppResources:
    """Return OmniDreams assets and routes for the shared WebRTC app."""
    from importlib.resources import files

    del spec
    return WebRTCAppResources(
        model_web_resource=files("omnidreams.webrtc").joinpath("web"),
        preload_name="Omnidreams",
        configure_app=configure_omnidreams_webrtc_app,
    )


def validate_postprocess_preset(preset: str) -> None:
    """Validate a configured preset without enabling the output system broadly."""
    if preset:
        resolve_postprocess_preset(preset)


def _get_omnidreams_manager(app: web.Application) -> _OmnidreamsSessionManager:
    return cast(_OmnidreamsSessionManager, app[SESSION_MANAGER_KEY])


__all__ = [
    "configure_omnidreams_webrtc_app",
    "omnidreams_webrtc_app_resources",
    "postprocess_options",
    "session_input",
    "validate_postprocess_preset",
]
