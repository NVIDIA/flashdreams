# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot WebRTC hooks for the shared demo API."""

from __future__ import annotations

from importlib.resources import files

from flashdreams.runtime.demo import DemoSpec, WebRTCAppResources
from lingbot.webrtc.server import configure_lingbot_webrtc_app


def lingbot_webrtc_app_resources(spec: DemoSpec) -> WebRTCAppResources:
    """Return Lingbot assets and routes for the shared WebRTC app."""
    del spec
    return WebRTCAppResources(
        model_web_resource=files("lingbot.webrtc").joinpath("web"),
        preload_name="Lingbot",
        configure_app=configure_lingbot_webrtc_app,
    )


__all__ = [
    "lingbot_webrtc_app_resources",
]
