# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared demo entrypoints."""

from __future__ import annotations

from typing import Any

from .replay import run_replay_demo
from .spec import DemoAdapter, DemoSpec


def run_flashdreams_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    **kwargs: Any,
) -> object:
    """Run a synchronous replay demo through the shared runtime runner."""
    return run_replay_demo(spec=spec, adapter=adapter, **kwargs)


def serve_flashdreams_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    **kwargs: Any,
) -> object:
    """Serve a WebRTC demo through the shared serving manager."""
    from .webrtc import serve_webrtc_demo

    return serve_webrtc_demo(spec=spec, adapter=adapter, **kwargs)


__all__ = ["run_flashdreams_demo", "serve_flashdreams_demo"]
