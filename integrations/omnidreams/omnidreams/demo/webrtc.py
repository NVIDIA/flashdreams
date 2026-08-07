# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams WebRTC hooks for the shared demo API."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Protocol, cast

from aiohttp import web
from omnidreams.webrtc.session import (
    OmnidreamsRuntimeConfig,
    OmnidreamsRuntimeError,
    OmnidreamsSessionInput,
    _validate_requested_postprocess_preset,
)

from flashdreams.plugins.registry import resolve_postprocess_preset
from flashdreams.runtime.demo import DemoSpec
from flashdreams.runtime.demo.webrtc import WebRTCAppExtension, WebRTCManagerOptions
from flashdreams.serving.webrtc.controls import WSAD_SUPPORTED_KEYS
from flashdreams.serving.webrtc.server import (
    SESSION_MANAGER_KEY,
    SessionBusyError,
)

_BUSY_MESSAGE = "An Omnidreams session is already active."
_WARMUP_LABEL = "Omnidreams WebRTC"


class _OmnidreamsSessionManager(Protocol):
    runtime_config: OmnidreamsRuntimeConfig

    def set_pending_session_input(
        self, session_input: OmnidreamsSessionInput
    ) -> None: ...


@dataclass(slots=True)
class _OmnidreamsPendingSessionInputState:
    runtime_config: OmnidreamsRuntimeConfig
    pending_session_input: OmnidreamsSessionInput | None = None

    def peek(self) -> OmnidreamsSessionInput | None:
        return self.pending_session_input

    def clear(self) -> None:
        self.pending_session_input = None

    def set(self, manager: Any, session_input: Any) -> None:
        if not isinstance(session_input, OmnidreamsSessionInput):
            raise TypeError("Expected OmnidreamsSessionInput.")
        if manager.has_active_session():
            raise SessionBusyError(_BUSY_MESSAGE)
        preset = session_input.postprocess_preset
        if preset:
            _validate_requested_postprocess_preset(
                requested_preset=preset,
                configured_preset=self.runtime_config.postprocess.preset,
            )
        self.pending_session_input = session_input


def create_omnidreams_webrtc_manager_options(
    *,
    runtime_config: OmnidreamsRuntimeConfig,
) -> WebRTCManagerOptions:
    """Build shared manager options for OmniDreams WebRTC semantics."""
    session_input_state = _OmnidreamsPendingSessionInputState(
        runtime_config=runtime_config
    )

    async def reset_runtime(runtime: Any, session_input: Any) -> None:
        await runtime.reset_for_new_session(session_input=session_input)

    def chunk_done_extra(runtime: Any, config: Any) -> dict[str, Any]:
        runtime_config = cast(OmnidreamsRuntimeConfig, config)
        return {
            "stream": "hdmap" if runtime_config.debug_serve_hdmaps else "rgb",
            "postprocess_preset": runtime.postprocess_preset,
        }

    return WebRTCManagerOptions(
        model_name=runtime_config.pipeline_config_name,
        busy_message=_BUSY_MESSAGE,
        warmup_label=_WARMUP_LABEL,
        runtime_error_types=(OmnidreamsRuntimeError,),
        close_session_on_generation_error=True,
        supported_keys=WSAD_SUPPORTED_KEYS,
        peek_pending_session_input=session_input_state.peek,
        clear_pending_session_input=session_input_state.clear,
        set_pending_session_input=session_input_state.set,
        reset_runtime_for_session=reset_runtime,
        chunk_done_extra=chunk_done_extra,
    )


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


def create_omnidreams_webrtc_app_extension(
    *,
    spec: DemoSpec,
    session_manager: Any,
    request_session_url: str,
) -> WebRTCAppExtension:
    """Describe OmniDreams browser assets and routes for the shared builder."""
    del session_manager, request_session_url
    output_preload_name = getattr(spec.output, "preload_name", None)
    preload_name = output_preload_name if isinstance(output_preload_name, str) else ""
    return WebRTCAppExtension(
        web_resource=files("omnidreams.webrtc").joinpath("web"),
        preload_name=preload_name or "Omnidreams",
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
    "create_omnidreams_webrtc_app_extension",
    "create_omnidreams_webrtc_manager_options",
    "postprocess_options",
    "session_input",
    "validate_postprocess_preset",
]
