# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams WebRTC hooks for the shared demo API."""

from __future__ import annotations

from typing import Any, cast

from aiohttp import web
from omnidreams.webrtc.session import (
    OmnidreamsRuntimeConfig,
    OmnidreamsRuntimeError,
    OmnidreamsSessionInput,
    _validate_requested_postprocess_preset,
)

from flashdreams.plugins.registry import resolve_postprocess_preset
from flashdreams.runtime.demo import DemoSpec
from flashdreams.runtime.demo.webrtc import SharedDemoWebRTCSessionManager
from flashdreams.serving.webrtc.controls import WSAD_SUPPORTED_KEYS
from flashdreams.serving.webrtc.manager import DEFAULT_CLIENT_LIVENESS_TIMEOUT_S
from flashdreams.serving.webrtc.server import (
    SESSION_MANAGER_KEY,
    SessionBusyError,
    create_packaged_webrtc_app,
)
from flashdreams.serving.webrtc.server import (
    close_package_resources as _close_package_resources,
)


class OmnidreamsDemoWebRTCSessionManager(SharedDemoWebRTCSessionManager):
    """Shared WebRTC manager customized for OmniDreams session semantics."""

    _busy_message = "An Omnidreams session is already active."
    _warmup_label = "Omnidreams WebRTC"
    _runtime_error_types = (OmnidreamsRuntimeError,)
    _close_session_on_generation_error = True
    _resampler_supported_keys = WSAD_SUPPORTED_KEYS

    runtime_config: OmnidreamsRuntimeConfig
    _runtime: Any

    def __init__(
        self,
        *,
        runtime: Any,
        runtime_config: OmnidreamsRuntimeConfig,
        fps: int,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        super().__init__(
            model_name=runtime_config.pipeline_config_name,
            runtime=runtime,
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )
        self._pending_session_input: OmnidreamsSessionInput | None = None

    def _model_name(self) -> str:
        return self.runtime_config.pipeline_config_name

    def _chunk_done_extra(self) -> dict[str, Any]:
        return {
            "stream": "hdmap" if self.runtime_config.debug_serve_hdmaps else "rgb",
            "postprocess_preset": self._runtime.postprocess_preset,
        }

    def _peek_pending_session_input(self) -> OmnidreamsSessionInput | None:
        return self._pending_session_input

    def _clear_pending_session_input(self) -> None:
        self._pending_session_input = None

    async def _reset_runtime_for_session(
        self, session_input: OmnidreamsSessionInput | None
    ) -> None:
        await self._runtime.reset_for_new_session(session_input=session_input)

    def set_pending_session_input(self, session_input: OmnidreamsSessionInput) -> None:
        if self.has_active_session():
            raise SessionBusyError(self._busy_message)
        preset = session_input.postprocess_preset
        if preset:
            _validate_requested_postprocess_preset(
                requested_preset=preset,
                configured_preset=self.runtime_config.postprocess.preset,
            )
        self._pending_session_input = session_input


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


def create_omnidreams_webrtc_app(
    *,
    spec: DemoSpec,
    session_manager: Any,
    request_session_url: str,
) -> web.Application:
    """Create the packaged OmniDreams browser app through shared serving glue."""
    from importlib.resources import as_file, files

    output_preload_name = getattr(spec.output, "preload_name", None)
    preload_name = output_preload_name if isinstance(output_preload_name, str) else ""
    return create_packaged_webrtc_app(
        web_resource=files("flashdreams.serving.webrtc").joinpath("web"),
        session_manager=session_manager,
        preload_name=preload_name or "Omnidreams",
        request_session_url=request_session_url,
        configure_app=configure_omnidreams_webrtc_app,
        as_file_fn=as_file,
        cleanup_callback=_close_package_resources,
    )


def validate_postprocess_preset(preset: str) -> None:
    """Validate a configured preset without enabling the output system broadly."""
    if preset:
        resolve_postprocess_preset(preset)


def _get_omnidreams_manager(app: web.Application) -> OmnidreamsDemoWebRTCSessionManager:
    return cast(OmnidreamsDemoWebRTCSessionManager, app[SESSION_MANAGER_KEY])


__all__ = [
    "OmnidreamsDemoWebRTCSessionManager",
    "configure_omnidreams_webrtc_app",
    "create_omnidreams_webrtc_app",
    "postprocess_options",
    "session_input",
    "validate_postprocess_preset",
]
