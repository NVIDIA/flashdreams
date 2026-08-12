# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2V-specific WebRTC session and browser routes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager
from flashdreams.serving.webrtc.runtime import WebRTCSessionConfig

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


__all__ = ["T2VWebRTCSessionManager"]
