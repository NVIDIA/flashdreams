# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams WebRTC hooks for the shared demo API."""

from __future__ import annotations

from importlib.resources import files
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
from flashdreams.runtime.demo.webrtc import (
    PendingSessionInputState,
    WebRTCAppExtension,
    WebRTCManagerOptions,
    WebRTCRoute,
    json_get_route,
    session_input_route,
)
from flashdreams.serving.webrtc.controls import WSAD_SUPPORTED_KEYS

_BUSY_MESSAGE = "An Omnidreams session is already active."
_WARMUP_LABEL = "Omnidreams WebRTC"


def create_omnidreams_webrtc_manager_options(
    *,
    runtime_config: OmnidreamsRuntimeConfig,
) -> WebRTCManagerOptions:
    """Build shared manager options for OmniDreams WebRTC semantics."""
    session_input_state = PendingSessionInputState(
        busy_message=_BUSY_MESSAGE,
        input_type=OmnidreamsSessionInput,
        validate_input=lambda session_input: validate_omnidreams_session_input(
            session_input,
            runtime_config=runtime_config,
        ),
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


def build_postprocess_options_payload(manager: Any) -> dict[str, Any]:
    """Return the postprocess preset selected at server launch."""
    configured_preset = manager.runtime_config.postprocess.preset
    presets = [configured_preset] if configured_preset else []
    return {
        "default_preset": configured_preset,
        "presets": presets,
    }


async def parse_omnidreams_session_input(
    request: web.Request,
    manager: Any,
) -> OmnidreamsSessionInput:
    """Parse browser-selected settings for the next WebRTC rollout."""
    del manager
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Expected JSON session input.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Session input must be a JSON object.")
    preset = payload.get("postprocess_preset")
    if not isinstance(preset, str):
        raise ValueError(
            "Session input must include string 'postprocess_preset'."
        )
    return OmnidreamsSessionInput(postprocess_preset=preset)


def build_session_input_response(
    session_input: Any,
    manager: Any,
) -> dict[str, Any]:
    """Return the browser-visible settings accepted for the next rollout."""
    del manager
    omnidreams_input = cast(OmnidreamsSessionInput, session_input)
    return {"postprocess_preset": omnidreams_input.postprocess_preset}


def validate_omnidreams_session_input(
    session_input: Any,
    *,
    runtime_config: OmnidreamsRuntimeConfig,
) -> None:
    """Validate OmniDreams session input before storing it for next reset."""
    omnidreams_input = cast(OmnidreamsSessionInput, session_input)
    preset = omnidreams_input.postprocess_preset
    if preset:
        _validate_requested_postprocess_preset(
            requested_preset=preset,
            configured_preset=runtime_config.postprocess.preset,
        )


def create_omnidreams_webrtc_routes() -> tuple[WebRTCRoute, ...]:
    """Describe OmniDreams browser support routes for the shared app builder."""
    return (
        json_get_route(
            "/api/postprocess/options",
            build_postprocess_options_payload,
        ),
        session_input_route(
            "/api/session/input",
            parse_input=parse_omnidreams_session_input,
            build_response=build_session_input_response,
        ),
    )


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
        routes=create_omnidreams_webrtc_routes(),
    )


def validate_postprocess_preset(preset: str) -> None:
    """Validate a configured preset without enabling the output system broadly."""
    if preset:
        resolve_postprocess_preset(preset)


__all__ = [
    "build_postprocess_options_payload",
    "build_session_input_response",
    "create_omnidreams_webrtc_app_extension",
    "create_omnidreams_webrtc_manager_options",
    "create_omnidreams_webrtc_routes",
    "parse_omnidreams_session_input",
    "validate_omnidreams_session_input",
    "validate_postprocess_preset",
]
