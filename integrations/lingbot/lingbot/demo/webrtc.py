# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot WebRTC hooks for the shared demo API."""

from __future__ import annotations

from typing import Any

from aiohttp import web

from flashdreams.runtime.demo import DemoSpec
from lingbot.webrtc.server import create_app
from lingbot.webrtc.session import (
    LingbotInferenceRuntime,
    LingbotRuntimeConfig,
    LingbotWebRTCSessionManager,
)


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
    """Create the packaged Lingbot browser app through existing serving glue."""
    del spec
    return create_app(
        session_manager=session_manager,
        request_session_url=request_session_url,
    )


__all__ = [
    "LingbotDemoWebRTCSessionManager",
    "create_lingbot_webrtc_app",
]
