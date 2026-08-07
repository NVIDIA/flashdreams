# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot WebRTC hooks for the shared demo API."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Any

from aiohttp import web

from flashdreams.runtime.demo import DemoSpec
from flashdreams.serving.webrtc.server import (
    close_package_resources,
    create_packaged_webrtc_app,
)
from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
    LingbotWebRTCSessionManager,
)
from lingbot.webrtc.server import configure_lingbot_webrtc_app


class LingbotDemoWebRTCSessionManager(LingbotWebRTCSessionManager):
    """Shared demo session manager using Lingbot's existing WebRTC semantics."""

    def __init__(
        self,
        *,
        runtime: LingbotInferenceRuntime,
        runtime_config: LingbotRuntimeConfig,
        fps: int,
        client_liveness_timeout_s: float,
    ) -> None:
        super().__init__(
            runtime=runtime,
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )


def create_lingbot_webrtc_app(
    *,
    spec: DemoSpec,
    session_manager: Any,
    request_session_url: str,
) -> web.Application:
    """Create Lingbot's shared browser app through generic serving glue."""
    del spec
    return create_packaged_webrtc_app(
        web_resource=files("flashdreams.serving.webrtc").joinpath("web"),
        model_web_resource=files("lingbot.webrtc").joinpath("web"),
        session_manager=session_manager,
        preload_name="Lingbot",
        request_session_url=request_session_url,
        configure_app=configure_lingbot_webrtc_app,
        as_file_fn=as_file,
        cleanup_callback=close_package_resources,
    )


__all__ = [
    "LingbotDemoWebRTCSessionManager",
    "create_lingbot_webrtc_app",
]
